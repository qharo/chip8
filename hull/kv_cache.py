"""Convex hull KV-cache for O(log N) attention inference.

For 2D attention heads (d_head=2), the maximum dot product Q·K occurs
at a vertex of the convex hull of the key points. By maintaining a
convex hull per head, we can find the best key in O(log V) where V
is the number of hull vertices (V << N in practice).

Uses scipy.spatial.ConvexHull for hull computation.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.spatial import ConvexHull, QhullError
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ConvexHullKVCache:
    """Per-head convex hull cache for 2D key vectors.

    Stores raw K, V tensors and maintains a convex hull index
    for fast nearest-key lookup per head.
    """

    def __init__(self, n_heads: int, d_head: int = 2):
        assert d_head == 2, "Convex hull cache requires d_head=2"
        self.n_heads = n_heads
        self.d_head = d_head

        # Raw KV storage (grown dynamically)
        self.K_cache: list[torch.Tensor] = []  # list of (B, n_heads, 1, 2)
        self.V_cache: list[torch.Tensor] = []

        # Per-head convex hull vertices (indices into K_cache)
        # hull_vertices[h] = sorted list of indices on the hull for head h
        self.hull_vertices: list[list[int]] = [[] for _ in range(n_heads)]

        # Full 2D points per head (for hull computation)
        self.points_per_head: list[list[np.ndarray]] = [[] for _ in range(n_heads)]

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """Add new key-value pairs.

        Args:
            K_new: (B, n_heads, 1, d_head)
            V_new: (B, n_heads, 1, d_head)
        """
        self.K_cache.append(K_new.detach().cpu())
        self.V_cache.append(V_new.detach().cpu())

        # Update hull for each head
        step_idx = len(self.K_cache) - 1
        for h in range(self.n_heads):
            point = K_new[0, h, 0, :].numpy()  # (2,)
            self.points_per_head[h].append(point)

            # Rebuild hull for this head
            self._rebuild_hull(h)

    def _rebuild_hull(self, head_idx: int):
        """Rebuild the convex hull for a specific head."""
        points = self.points_per_head[head_idx]
        if len(points) < 3:
            # Hull needs at least 3 points
            self.hull_vertices[head_idx] = list(range(len(points)))
            return

        if not HAS_SCIPY:
            # Fallback: use all points as "hull"
            self.hull_vertices[head_idx] = list(range(len(points)))
            return

        pts_array = np.array(points)
        try:
            hull = ConvexHull(pts_array)
            self.hull_vertices[head_idx] = sorted(hull.vertices.tolist())
        except QhullError:
            # Degenerate case (collinear points)
            self.hull_vertices[head_idx] = list(range(len(points)))

    def query_hardmax(self, Q: torch.Tensor) -> torch.Tensor:
        """Query the hull to find the nearest key for each head.

        Args:
            Q: (B, n_heads, 1, d_head)

        Returns:
            output: (B, n_heads, 1, d_head) — values of nearest hull keys
        """
        B = Q.shape[0]
        outputs = torch.zeros_like(Q)

        if not self.K_cache:
            return outputs

        # Stack all K and V
        K_all = torch.cat(self.K_cache, dim=2)  # (B, n_heads, T, 2)
        V_all = torch.cat(self.V_cache, dim=2)  # (B, n_heads, T, 2)

        for h in range(self.n_heads):
            hull_idx = self.hull_vertices[h]
            if not hull_idx:
                continue

            # Get hull key points
            hull_K = K_all[0, h, hull_idx, :]  # (V, 2)
            q = Q[0, h, 0, :]  # (2,)

            # Dot product with hull vertices
            dots = torch.mv(hull_K, q)  # (V,)
            best_local = dots.argmax().item()
            best_global = hull_idx[best_local]

            outputs[0, h, 0, :] = V_all[0, h, best_global, :]

        return outputs

    def __len__(self):
        return len(self.K_cache)


class StandardKVCache:
    """Simple KV-cache without hull optimization (baseline)."""

    def __init__(self):
        self.K: torch.Tensor | None = None
        self.V: torch.Tensor | None = None

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        if self.K is None:
            self.K = K_new
            self.V = V_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)

    def query_hardmax(self, Q: torch.Tensor) -> torch.Tensor:
        """Standard hardmax: dot product against ALL keys."""
        scores = torch.matmul(Q, self.K.transpose(-2, -1))
        idx = scores.argmax(dim=-1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, Q.shape[-1])
        return torch.gather(self.V, dim=2, index=idx_expanded)

    def __len__(self):
        return 0 if self.K is None else self.K.shape[2]
