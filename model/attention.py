"""2D attention with temperature annealing and Rotary Position Embeddings (RoPE)."""

from __future__ import annotations
import torch._dynamo
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies Rotary Position Embedding to 2D query and key vectors."""
    q_0, q_1 = q[..., 0:1], q[..., 1:2]
    k_0, k_1 = k[..., 0:1], k[..., 1:2]
    
    # 2D Rotation Matrix
    q_out_0 = q_0 * cos - q_1 * sin
    q_out_1 = q_0 * sin + q_1 * cos
    q_out = torch.cat([q_out_0, q_out_1], dim=-1)
    
    k_out_0 = k_0 * cos - k_1 * sin
    k_out_1 = k_0 * sin + k_1 * cos
    k_out = torch.cat([k_out_0, k_out_1], dim=-1)
    
    return q_out, k_out


class Attention2D(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.scale = math.sqrt(d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

        # --- CUSTOM 2D RoPE PRE-COMPUTATION ---
        # Since d_head=2, we distribute the frequencies across the heads!
        inv_freq = 1.0 / (10000 ** (torch.arange(0, n_heads, dtype=torch.float32) / n_heads))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(inv_freq, t)  # (n_heads, max_seq_len)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(-1)) # (1, n_heads, T, 1)
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(-1)) # (1, n_heads, T, 1)

    def _get_cos_sin(self, start_pos: int, seq_len: int, device: torch.device):
        if start_pos + seq_len <= self.cos_cached.shape[2]:
            return (self.cos_cached[:, :, start_pos:start_pos+seq_len, :],
                    self.sin_cached[:, :, start_pos:start_pos+seq_len, :])
        
        # Fallback for dynamic extended generation
        t = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(self.inv_freq, t)
        cos = freqs.cos().unsqueeze(0).unsqueeze(-1)
        sin = freqs.sin().unsqueeze(0).unsqueeze(-1)
        return cos, sin

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                use_hardmax: bool = False, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self._get_cos_sin(start_pos, T, Q.device)
        Q, K = apply_rope(Q, K, cos, sin)

        mask = self.causal_mask[:T, :T]
        is_cuda = x.device.type == "cuda"

        if is_cuda and not use_hardmax:
            out = self._cuda_attention(Q, K, V, T, temperature)
        elif use_hardmax:
            if is_cuda:
                out = self._cuda_hardmax(Q, K, V, T)
            else:
                out = self._chunked_hardmax(Q, K, V, T, B, mask)
        else:
            out = self._chunked_softmax(Q, K, V, T, B, mask, temperature)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        out = self.resid_dropout(out)
        return out

    @torch._dynamo.disable
    def _cuda_attention(self, Q: torch.Tensor, K: torch.Tensor,
                        V: torch.Tensor, T: int,
                        temperature: torch.Tensor) -> torch.Tensor:
        """Full parallel attention on CUDA using Flash Attention."""
        
        pad_size = 16 - self.d_head
        scale_correction = math.sqrt(16) / (self.scale * temperature)
        Q_scaled = Q * scale_correction
        
        # Pad and FORCE contiguity (Flash Attention strictly requires this to avoid Math fallback)
        Q_pad = F.pad(Q_scaled, (0, pad_size)).contiguous()
        K_pad = F.pad(K, (0, pad_size)).contiguous()
        V_pad = F.pad(V, (0, pad_size)).contiguous()
        
        out_pad = F.scaled_dot_product_attention(
            Q_pad, K_pad, V_pad,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        
        # Slice the 14 zeros off the end to return perfectly to d_head=2
        return out_pad[..., :self.d_head].contiguous()
   

    def _cuda_hardmax(self, Q: torch.Tensor, K: torch.Tensor,
                      V: torch.Tensor, T: int) -> torch.Tensor:
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
        out = torch.empty(B, self.n_heads, T, self.d_head,
                          device=Q.device, dtype=Q.dtype)

        chunk_size = min(self.n_heads, max(1, 32 * 512 * 512 // (B * T * T)))
        chunk_size = max(1, min(chunk_size, self.n_heads))

        for start in range(0, self.n_heads, chunk_size):
            end = min(start + chunk_size, self.n_heads)
            Qc = Q[:, start:end, :, :]
            Kc = K[:, start:end, :, :]
            Vc = V[:, start:end, :, :]

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

        return out

    def _chunked_hardmax(self, Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, T: int, B: int,
                         mask: torch.Tensor) -> torch.Tensor:
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

        return out

    def forward_step(self, x: torch.Tensor, kv_cache: tuple | None = None,
                     temperature: float = 1.0,
                     use_hardmax: bool = False, seq_pos: int = 0) -> tuple[torch.Tensor, tuple]:
        B = x.shape[0]

        Q = self.W_q(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        K_new = self.W_k(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        V_new = self.W_v(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to the new K and Q
        cos, sin = self._get_cos_sin(seq_pos, 1, Q.device)
        Q, K_new = apply_rope(Q, K_new, cos, sin)

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
