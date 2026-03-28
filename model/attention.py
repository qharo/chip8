"""2D attention with temperature annealing.

Two paths:
  - CUDA fast path: full parallel attention across all 128 heads (single matmul)
  - MPS/CPU safe path: chunked attention to stay within memory limits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2D(nn.Module):
    """Multi-head attention where each head operates in 2D.

    Key innovation: d_head = 2, so each head's Q/K/V are 2D vectors.
    During training: standard softmax with temperature scaling.
    During inference: can switch to hardmax (argmax dot product).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.scale = math.sqrt(d_head)

        # Q, K, V projections (output: d_model = n_heads * d_head)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                use_hardmax: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            temperature: attention temperature (lower = sharper)
            use_hardmax: if True, use argmax instead of softmax

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        mask = self.causal_mask[:T, :T]  # (T, T)

        is_cuda = x.device.type == "cuda"

        if is_cuda and not use_hardmax:
            # CUDA fast path: full parallel attention, all 128 heads at once
            out = self._cuda_attention(Q, K, V, T, temperature)
        elif use_hardmax:
            if is_cuda:
                out = self._cuda_hardmax(Q, K, V, T)
            else:
                out = self._chunked_hardmax(Q, K, V, T, B, mask)
        else:
            # MPS/CPU safe path: chunked heads
            out = self._chunked_softmax(Q, K, V, T, B, mask, temperature)

        # (B, n_heads, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        out = self.resid_dropout(out)
        return out

    def _cuda_attention(self, Q: torch.Tensor, K: torch.Tensor,
                        V: torch.Tensor, T: int,
                        temperature: float) -> torch.Tensor:
        """Full parallel attention on CUDA. (B, n_heads, T, T) score matrix
        fits in 24GB+ VRAM."""
        # Try Flash Attention via scaled_dot_product_attention
        # Flash Attention doesn't support custom temperature, so we
        # fall back to manual if temperature != 1.0
        if temperature == 1.0:
            try:
                mask_bool = self.causal_mask[:T, :T]
                out = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask_bool,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,  # we pass the mask explicitly
                )
                return out
            except Exception:
                pass

        # Manual attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores / temperature
        mask_expanded = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_expanded, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, V)

    def _cuda_hardmax(self, Q: torch.Tensor, K: torch.Tensor,
                      V: torch.Tensor, T: int) -> torch.Tensor:
        """Full parallel hardmax on CUDA."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        mask_expanded = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_expanded, float("-inf"))
        idx = scores.argmax(dim=-1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        return torch.gather(V, dim=2, index=idx_expanded)

    def _chunked_softmax(self, Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, T: int, B: int,
                         mask: torch.Tensor,
                         temperature: float) -> torch.Tensor:
        """Chunked head processing for MPS/CPU. Keeps peak memory low."""
        out = torch.empty(B, self.n_heads, T, self.d_head,
                          device=Q.device, dtype=Q.dtype)

        # Dynamic chunk size based on available memory estimate
        chunk_size = min(self.n_heads, max(1, 32 * 512 * 512 // (B * T * T)))
        chunk_size = max(1, min(chunk_size, self.n_heads))

        for start in range(0, self.n_heads, chunk_size):
            end = min(start + chunk_size, self.n_heads)
            Qc = Q[:, start:end, :, :]
            Kc = K[:, start:end, :, :]
            Vc = V[:, start:end, :, :]

            # Reshape for batched matmul
            nc = end - start
            Qr = Qc.reshape(B * nc, T, self.d_head)
            Kr = Kc.reshape(B * nc, T, self.d_head)
            Vr = Vc.reshape(B * nc, T, self.d_head)

            scores = torch.bmm(Qr, Kr.transpose(1, 2)) / self.scale
            scores = scores / temperature
            scores = scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            chunk_out = torch.bmm(attn_weights, Vr)

            out[:, start:end, :, :] = chunk_out.reshape(B, nc, T, self.d_head)
            del scores, Qc, Kc, Vc, Qr, Kr, Vr

        return out

    def _chunked_hardmax(self, Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, T: int, B: int,
                         mask: torch.Tensor) -> torch.Tensor:
        """Chunked hardmax for MPS/CPU."""
        out = torch.empty(B, self.n_heads, T, self.d_head,
                          device=Q.device, dtype=Q.dtype)

        chunk_size = min(self.n_heads, max(1, 32 * 512 * 512 // (B * T * T)))
        chunk_size = max(1, min(chunk_size, self.n_heads))

        for start in range(0, self.n_heads, chunk_size):
            end = min(start + chunk_size, self.n_heads)
            nc = end - start

            Qr = Q[:, start:end, :, :].reshape(B * nc, T, self.d_head)
            Kr = K[:, start:end, :, :].reshape(B * nc, T, self.d_head)
            Vr = V[:, start:end, :, :].reshape(B * nc, T, self.d_head)

            scores = torch.bmm(Qr, Kr.transpose(1, 2)) / self.scale
            scores = scores.masked_fill(mask, float("-inf"))
            idx = scores.argmax(dim=-1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.d_head)
            chunk_out = torch.gather(Vr, dim=1, index=idx_exp)

            out[:, start:end, :, :] = chunk_out.reshape(B, nc, T, self.d_head)
            del scores, Qr, Kr, Vr

        return out

    def forward_step(self, x: torch.Tensor, kv_cache: tuple | None = None,
                     temperature: float = 1.0,
                     use_hardmax: bool = False) -> tuple[torch.Tensor, tuple]:
        """Single-step forward for autoregressive generation.

        Args:
            x: (B, 1, d_model) — single new token embedding
            kv_cache: (K_cache, V_cache) each (B, n_heads, T_cache, d_head)
            temperature: attention temperature
            use_hardmax: if True, use argmax

        Returns:
            output: (B, 1, d_model)
            new_cache: (K_cache, V_cache) updated
        """
        B = x.shape[0]

        Q = self.W_q(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        K_new = self.W_k(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        V_new = self.W_v(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        if use_hardmax:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            idx = scores.argmax(dim=-1)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            out = torch.gather(V, dim=2, index=idx_expanded)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            scores = scores / temperature
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        out = self.W_o(out)

        return out, (K, V)
