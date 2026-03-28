"""Causal transformer with 2D attention heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import Attention2D

try:
    from torch.utils.checkpoint import checkpoint
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> Attention -> residual -> LayerNorm -> FFN -> residual."""

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
                use_hardmax: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), temperature=temperature,
                          use_hardmax=use_hardmax)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_step(self, x: torch.Tensor, kv_cache: tuple | None = None,
                     temperature: float = 1.0,
                     use_hardmax: bool = False) -> tuple[torch.Tensor, tuple]:
        """Single-step forward for autoregressive generation."""
        normed = self.ln1(x)
        attn_out, new_kv = self.attn.forward_step(
            normed, kv_cache=kv_cache,
            temperature=temperature, use_hardmax=use_hardmax,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class Chip8Transformer(nn.Module):
    """Causal transformer for CHIP-8 emulation with 2D attention heads.

    ~5M parameters. Designed for next-token prediction on CHIP-8 execution traces.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_checkpoint = config.use_checkpoint

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: output head shares weights with token embeddings
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
                use_hardmax: bool = False) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token IDs
            temperature: attention temperature
            use_hardmax: if True, use hardmax attention

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        # Token + position embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training and HAS_CHECKPOINT:
                x = checkpoint(block, x, temperature, use_hardmax,
                               use_reentrant=False)
            else:
                x = block(x, temperature=temperature, use_hardmax=use_hardmax)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0,
                 use_hardmax: bool = False) -> torch.Tensor:
        """Autoregressive generation with KV-cache.

        Args:
            input_ids: (B, T) initial tokens (the prompt)
            max_new_tokens: number of tokens to generate
            temperature: attention temperature
            use_hardmax: if True, use hardmax attention

        Returns:
            (B, T + max_new_tokens) full sequence including prompt
        """
        self.eval()
        cur = input_ids

        # Prefill: process the entire prompt, build KV cache
        B, T = cur.shape
        positions = torch.arange(T, device=cur.device).unsqueeze(0)
        x = self.token_emb(cur) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        kv_caches = []
        for block in self.blocks:
            normed = block.ln1(x)
            # Build K, V for the full prompt
            K = block.attn.W_k(normed).view(B, T, block.attn.n_heads,
                                              block.attn.d_head).transpose(1, 2)
            V = block.attn.W_v(normed).view(B, T, block.attn.n_heads,
                                              block.attn.d_head).transpose(1, 2)
            kv_caches.append((K, V))

            # Standard attention forward for prefill
            attn_out = block.attn(normed, temperature=temperature,
                                  use_hardmax=use_hardmax)
            x = x + attn_out
            x = x + block.ffn(block.ln2(x))

        # Generate tokens one at a time
        for step in range(max_new_tokens):
            # Embed just the last token
            last_id = cur[:, -1:]
            pos = torch.tensor([[cur.shape[1] - 1]], device=cur.device)
            x_step = self.token_emb(last_id) + self.pos_emb(pos)

            new_kvs = []
            for i, block in enumerate(self.blocks):
                normed = block.ln1(x_step)
                attn_out, new_kv = block.attn.forward_step(
                    normed, kv_cache=kv_caches[i],
                    temperature=temperature, use_hardmax=use_hardmax,
                )
                new_kvs.append(new_kv)
                x_step = x_step + attn_out
                x_step = x_step + block.ffn(block.ln2(x_step))

            kv_caches = new_kvs

            x_last = self.ln_f(x_step)
            logits = self.head(x_last)  # (B, 1, vocab_size)
            next_token = logits.argmax(dim=-1)  # greedy

            cur = torch.cat([cur, next_token], dim=1)

        return cur

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
