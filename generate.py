"""Game runner: load a ROM, run it through the trained transformer, render with pygame."""

from __future__ import annotations

import argparse
import sys
import time

import torch

from data.tokenizer import Tokenizer
from data.generator import load_rom_file, generate_random_rom
from emulator.tracer import format_mem_snapshot, format_reg_snapshot
from model.config import ModelConfig
from model.transformer import Chip8Transformer


KEY_MAP = {
    # pygame key -> CHIP-8 key
    # Standard CHIP-8 keypad mapping:
    # 1 2 3 4     -> 1 2 3 C
    # Q W E R     -> 4 5 6 D
    # A S D F     -> 7 8 9 E
    # Z X C V     -> A 0 B F
}


def build_key_map():
    """Build pygame key -> CHIP-8 mapping."""
    try:
        import pygame
    except ImportError:
        return {}

    return {
        pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0xC,
        pygame.K_q: 0x4, pygame.K_w: 0x5, pygame.K_e: 0x6, pygame.K_r: 0xD,
        pygame.K_a: 0x7, pygame.K_s: 0x8, pygame.K_d: 0x9, pygame.K_f: 0xE,
        pygame.K_z: 0xA, pygame.K_x: 0x0, pygame.K_c: 0xB, pygame.K_v: 0xF,
    }


def load_model(checkpoint_path: str, device: torch.device) -> tuple[Chip8Transformer, ModelConfig]:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = Chip8Transformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def run_headless(model: Chip8Transformer, tokenizer: Tokenizer,
                 rom: bytes, max_cycles: int, device: torch.device):
    """Run the transformer emulator without display (headless mode)."""
    # Build initial prompt
    from emulator.cpu import CPU
    cpu = CPU(seed=0)
    cpu.load_rom(rom)

    prompt_lines = []
    prompt_lines.append(format_mem_snapshot(cpu.memory))
    prompt_lines.append(format_reg_snapshot(cpu.state))
    prompt_lines.append("<TRACE_START>")

    prompt_text = " ".join(prompt_lines)
    input_ids = tokenizer.encode_line(prompt_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print("Prompt tokens:", len(input_ids))
    print("Generating trace...")

    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_cycles,
                                temperature=0.01, use_hardmax=True)

    generated = output[0, len(input_ids):].tolist()
    decoded = tokenizer.decode(generated)
    print(decoded[:5000])
    print(f"\n... ({len(generated)} tokens generated)")


def run_with_display(model: Chip8Transformer, tokenizer: Tokenizer,
                     rom: bytes, max_cycles: int, device: torch.device):
    """Run the transformer emulator with pygame display."""
    try:
        import pygame
    except ImportError:
        print("pygame not installed. Run: pip install pygame")
        print("Falling back to headless mode.")
        run_headless(model, tokenizer, rom, max_cycles, device)
        return

    SCALE = 10
    WIDTH, HEIGHT = 64 * SCALE, 32 * SCALE
    FPS = 60
    CYCLES_PER_FRAME = 10  # ~600 Hz

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CHIP-8 Transformer Emulator")
    clock = pygame.time.Clock()

    key_map = build_key_map()

    # Build initial prompt from emulator state
    from emulator.cpu import CPU
    cpu = CPU(seed=0)
    cpu.load_rom(rom)

    prompt_lines = []
    prompt_lines.append(format_mem_snapshot(cpu.memory))
    prompt_lines.append(format_reg_snapshot(cpu.state))
    prompt_lines.append("<TRACE_START>")

    prompt_text = " ".join(prompt_lines)
    input_ids = tokenizer.encode_line(prompt_text)
    token_buffer = list(input_ids)

    # Display state (parsed from generated tokens)
    display_pixels = [[0] * 64 for _ in range(32)]

    running = True
    frame_count = 0

    print("Running CHIP-8 Transformer Emulator")
    print("Controls: 1-4, Q-W-E-R, A-S-D-F, Z-X-C-V")
    print("Press ESC to quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_map:
                    chip8_key = key_map[event.key]
                    token = f"<KEY_{chip8_key:X}>"
                    if token in tokenizer.token_to_id:
                        token_buffer.append(tokenizer.token_to_id[token])
            elif event.type == pygame.KEYUP:
                if event.key in key_map:
                    chip8_key = key_map[event.key]
                    token = f"<NO_KEY>"
                    if token in tokenizer.token_to_id:
                        token_buffer.append(tokenizer.token_to_id[token])

        # Generate N cycles worth of tokens
        input_tensor = torch.tensor([token_buffer], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=CYCLES_PER_FRAME * 10,  # rough estimate
                temperature=0.01,
                use_hardmax=True,
            )

        new_tokens = output[0, len(token_buffer):].tolist()
        token_buffer = output[0].tolist()

        # Parse generated tokens for display updates
        decoded = tokenizer.decode(new_tokens)
        # Simple parsing: look for DRW instructions
        # (Full implementation would maintain display state)

        # Render
        screen.fill((0, 0, 0))
        for y in range(32):
            for x in range(64):
                if display_pixels[y][x]:
                    pygame.draw.rect(screen, (255, 255, 255),
                                     (x * SCALE, y * SCALE, SCALE, SCALE))

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

        # Trim buffer to prevent unbounded growth
        if len(token_buffer) > 4096:
            # Keep last 2048 tokens
            token_buffer = token_buffer[-2048:]

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Run CHIP-8 Transformer Emulator")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--rom", type=str, default=None,
                        help="Path to .ch8 ROM file")
    parser.add_argument("--random-rom", action="store_true",
                        help="Generate a random ROM")
    parser.add_argument("--cycles", type=int, default=2000,
                        help="Max cycles to generate")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU device")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {model.count_parameters():,} parameters")

    tokenizer = Tokenizer()

    # Load ROM
    if args.rom:
        rom = load_rom_file(args.rom)
        print(f"Loaded ROM: {args.rom} ({len(rom)} bytes)")
    elif args.random_rom:
        import random
        rom = generate_random_rom(random.Random(42), num_instructions=64)
        print(f"Generated random ROM ({len(rom)} bytes)")
    else:
        print("No ROM specified. Use --rom or --random-rom")
        sys.exit(1)

    if args.headless:
        run_headless(model, tokenizer, rom, args.cycles, device)
    else:
        run_with_display(model, tokenizer, rom, args.cycles, device)


if __name__ == "__main__":
    main()
