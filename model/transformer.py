"""Causal transformer with 2D attention heads and RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import Attention2D, apply_rope

try:
    from torch.utils.checkpoint import checkpoint
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention2D(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                use_hardmax: bool = False, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), temperature=temperature,
                          use_hardmax=use_hardmax, start_pos=start_pos)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_step(self, x: torch.Tensor, kv_cache: tuple | None = None,
                     temperature: float = 1.0,
                     use_hardmax: bool = False, seq_pos: int = 0) -> tuple[torch.Tensor, tuple]:
        normed = self.ln1(x)
        attn_out, new_kv = self.attn.forward_step(
            normed, kv_cache=kv_cache,
            temperature=temperature, use_hardmax=use_hardmax, seq_pos=seq_pos
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class Chip8Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_checkpoint = config.use_checkpoint

        # Removed pos_emb to strictly enforce relative learning via RoPE
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_token_id)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor,
                temperature: float = 1.0,
                use_hardmax: bool = False, start_pos: int = 0) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            if self.use_checkpoint and self.training and HAS_CHECKPOINT:
                x = checkpoint(block, x, temperature, use_hardmax, start_pos,
                               use_reentrant=False)
            else:
                x = block(x, temperature=temperature, use_hardmax=use_hardmax, start_pos=start_pos)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0,
                 use_hardmax: bool = False) -> torch.Tensor:
        self.eval()
        cur = input_ids
        B, T = cur.shape
        
        x = self.token_emb(cur)
        x = self.emb_dropout(x)

        kv_caches = []
        for block in self.blocks:
            normed = block.ln1(x)
            K = block.attn.W_k(normed).view(B, T, block.attn.n_heads,
                                              block.attn.d_head).transpose(1, 2)
            V = block.attn.W_v(normed).view(B, T, block.attn.n_heads,
                                              block.attn.d_head).transpose(1, 2)
            
            # Apply RoPE to prefill K cache
            cos, sin = block.attn._get_cos_sin(0, T, x.device)
            _, K = apply_rope(torch.zeros_like(K), K, cos, sin)
            
            kv_caches.append((K, V))
            attn_out = block.attn(normed, temperature=temperature,
                                  use_hardmax=use_hardmax, start_pos=0)
            x = x + attn_out
            x = x + block.ffn(block.ln2(x))

        for step in range(max_new_tokens):
            last_id = cur[:, -1:]
            seq_pos = cur.shape[1] - 1  # Get exact position for RoPE
            
            x_step = self.token_emb(last_id)
            new_kvs = []
            
            for i, block in enumerate(self.blocks):
                normed = block.ln1(x_step)
                attn_out, new_kv = block.attn.forward_step(
                    normed, kv_cache=kv_caches[i],
                    temperature=temperature, use_hardmax=use_hardmax,
                    seq_pos=seq_pos
                )
                new_kvs.append(new_kv)
                x_step = x_step + attn_out
                x_step = x_step + block.ffn(block.ln2(x_step))

            kv_caches = new_kvs
            x_last = self.ln_f(x_step)
            logits = self.head(x_last)
            next_token = logits.argmax(dim=-1)
            cur = torch.cat([cur, next_token], dim=1)

        return cur

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
