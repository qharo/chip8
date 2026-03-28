"""Hardmax attention for inference-time O(log N) lookups.

Instead of softmax, we find the single key with maximum dot product
for each query, then take that key's value directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def hardmax_attention_step(Q: torch.Tensor, K: torch.Tensor,
                           V: torch.Tensor) -> torch.Tensor:
    """Compute hardmax attention for a single query position.

    Args:
        Q: (B, n_heads, 1, d_head) — query for current position
        K: (B, n_heads, T, d_head) — all cached keys
        V: (B, n_heads, T, d_head) — all cached values

    Returns:
        output: (B, n_heads, 1, d_head) — the value of the nearest key
        indices: (B, n_heads, 1) — which key was selected
    """
    # Dot product: (B, n_heads, 1, T)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Argmax: which key has highest dot product
    idx = scores.argmax(dim=-1)  # (B, n_heads, 1)

    # Gather the corresponding value
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, Q.shape[-1])
    out = torch.gather(V, dim=2, index=idx_expanded)

    return out, idx


def hardmax_attention_full(Q: torch.Tensor, K: torch.Tensor,
                           V: torch.Tensor) -> torch.Tensor:
    """Compute hardmax attention for full sequence.

    Args:
        Q: (B, n_heads, T, d_head)
        K: (B, n_heads, T, d_head)
        V: (B, n_heads, T, d_head)

    Returns:
        output: (B, n_heads, T, d_head)
    """
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, T, T)

    # Apply causal mask (lower triangular = allowed)
    T = Q.shape[2]
    mask = torch.triu(torch.ones(T, T, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    idx = scores.argmax(dim=-1)  # (B, n_heads, T)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, Q.shape[-1])
    out = torch.gather(V, dim=2, index=idx_expanded)
    return out
