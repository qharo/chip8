"""Training script for CHIP-8 transformer with temperature annealing.

Supports CUDA (mixed precision, torch.compile, full parallel attention),
MPS (gradient checkpointing, chunked attention), and CPU.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tokenizer import Tokenizer
from data.generator import generate_dataset
from data.dataset import TraceDataset
from model.config import ModelConfig
from model.transformer import Chip8Transformer




def get_temperature(step: int, total_steps: int, temp_start: float,
                    temp_end: float) -> float:
    """Reach temp_end at 75% of training, then stay there."""
    cooldown_step = int(total_steps * 0.75)
    if step >= cooldown_step:
        return temp_end

    ratio = step / cooldown_step
    return temp_start * (temp_end / temp_start) ** ratio

def get_device(args) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.device:
        return torch.device(args.device)
    return ModelConfig.detect_device()


def train(args):
    device = get_device(args)
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and not args.no_amp
    amp_dtype = torch.bfloat16 if is_cuda else torch.float32

    print(f"Device: {device}")
    if is_cuda:
        gpu_name = torch.cuda.get_device_properties(0).name
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.0f} GB)")
    print(f"Mixed precision: {use_amp} (dtype={amp_dtype})")

    # 1. Generate data — per-ROM traces for proper train/val splitting
    print("\nGenerating training data...")
    t0 = time.time()
    rom_dir = args.rom_dir if args.rom_dir else None
    rom_traces = generate_dataset(
        num_random_roms=args.num_roms,
        instructions_per_rom=args.inst_per_rom,
        cycles_per_rom=args.cycles_per_rom,
        rom_dir=rom_dir,
        seed=args.seed,
        per_rom=True,
    )
    total_lines = sum(len(t) for t in rom_traces)
    print(f"Generated {len(rom_traces)} ROM traces, {total_lines:,} total lines in {time.time()-t0:.1f}s")

    # 2. Tokenize
    print("Tokenizing...")
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Tokenize each ROM trace separately (for ROM-level split)
    rom_token_lists = []
    for trace in rom_traces:
        rom_token_lists.append(tokenizer.encode_trace(trace))
    total_tokens = sum(len(t) for t in rom_token_lists)
    print(f"Total tokens: {total_tokens:,}")

    # 3. ROM-level train/val split — no leakage between splits
    import random as _random
    _random.Random(args.seed).shuffle(rom_token_lists)

    val_rom_count = max(1, int(len(rom_token_lists) * 0.05))
    train_rom_tokens = [tok for rom in rom_token_lists[val_rom_count:] for tok in rom]
    val_rom_tokens = [tok for rom in rom_token_lists[:val_rom_count] for tok in rom]

    print(f"Train ROMs: {len(rom_token_lists) - val_rom_count}, Val ROMs: {val_rom_count}")
    print(f"Train tokens: {len(train_rom_tokens):,}, Val tokens: {len(val_rom_tokens):,}")

    # 4. Datasets — overlapping windows for train, non-overlapping for val
    seq_len = args.seq_len
    train_ds = TraceDataset(train_rom_tokens, seq_len=seq_len,
                            stride=seq_len,  # 75% overlap
                            pad_id=tokenizer.pad_id)
    val_ds = TraceDataset(val_rom_tokens, seq_len=seq_len,
                          stride=seq_len,  # no overlap for honest eval
                          pad_id=tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 4. Model
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=seq_len,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_id,
        use_checkpoint=args.checkpoint,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
    )
    model = Chip8Transformer(config).to(device)

    # torch.compile on CUDA (PyTorch 2.x)
    if is_cuda and not args.no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Gradient checkpointing: {config.use_checkpoint}")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    # Learning rate schedule: warmup + cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler (not needed for bfloat16, but for safety)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # 6. Resume from checkpoint
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "latest":
            step_ckpts = sorted(Path(args.output_dir).glob("checkpoint_step*.pt"))
            if step_ckpts:
                ckpt_path = str(step_ckpts[-1])
            else:
                ckpt_path = os.path.join(args.output_dir, "final_model.pt")

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resumed from {ckpt_path} (epoch {start_epoch}, step {global_step})")
        else:
            print(f"Checkpoint not found: {ckpt_path}, starting fresh")

    # 7. Training loop
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    leave=True, dynamic_ncols=True)
        t_epoch = time.time()
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Current temperature
            temp = get_temperature(global_step, total_steps,
                                   config.temp_start, config.temp_end)

            optimizer.zero_grad()

            # Mixed precision forward
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype, enabled=use_amp):
                temp_tensor = torch.tensor(temp, device=device, dtype=torch.float32)
                logits = model(x, temperature=temp_tensor)
                loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))

            # Backward (with scaler for fp16, no-op for bf16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Accuracy (non-pad tokens)
            preds = logits.argmax(dim=-1)
            mask = y != tokenizer.pad_id
            correct = ((preds == y) & mask).sum().item()
            total = mask.sum().item()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total
            global_step += 1

            # Update progress bar
            lr = optimizer.param_groups[0]["lr"]
            acc = correct / max(total, 1) * 100
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.1f}%",
                             temp=f"{temp:.3f}", lr=f"{lr:.1e}",
                             step=global_step)

            # Step-based checkpoint
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                pbar.write(f"  Saved checkpoint at step {global_step}")
                save_path = os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt")
                torch.save({
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                }, save_path)

        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / max(epoch_total, 1) * 100

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype, enabled=use_amp):
                for x, y in tqdm(val_loader, desc="  Validating",
                                 leave=False, dynamic_ncols=True):
                    x, y = x.to(device), y.to(device)
                    logits = model(x, temperature=config.temp_end)
                    loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                    preds = logits.argmax(dim=-1)
                    mask = y != tokenizer.pad_id
                    val_correct += ((preds == y) & mask).sum().item()
                    val_total += mask.sum().item()
                    val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1) * 100

        elapsed = time.time() - t_epoch
        samples_per_sec = (len(train_ds) * seq_len) / elapsed
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s, "
              f"{samples_per_sec:,.0f} tok/s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            }, save_path)

    # Save final model
    save_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, save_path)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CHIP-8 Transformer")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (e.g., cuda, mps, cpu)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU")

    # Data
    parser.add_argument("--num-roms", type=int, default=500)
    parser.add_argument("--inst-per-rom", type=int, default=64)
    parser.add_argument("--cycles-per-rom", type=int, default=2000)
    parser.add_argument("--rom-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)

    # Temperature
    parser.add_argument("--temp-start", type=float, default=10.0)
    parser.add_argument("--temp-end", type=float, default=0.01)

    # CUDA optimizations (auto-disabled on MPS/CPU)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-every-steps", type=int, default=100,
                        help="Save checkpoint every N steps (0 to disable)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from, or 'latest' to auto-find")

    args = parser.parse_args()

    # Auto-set checkpoint flag
    if args.no_checkpoint:
        args.checkpoint = False
    else:
        args.checkpoint = True

    train(args)


if __name__ == "__main__":
    main()
