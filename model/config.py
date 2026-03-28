"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    vocab_size: int = 370
    d_model: int = 256
    n_heads: int = 128
    d_head: int = 2         # 256 / 128 = 2D attention
    n_layers: int = 6
    d_ff: int = 1024        # 4 * d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    use_checkpoint: bool = True   # gradient checkpointing

    # Temperature annealing
    temp_start: float = 10.0
    temp_end: float = 0.01

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.d_head, \
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"

    @staticmethod
    def detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def for_device(cls, device: torch.device, **kwargs) -> "ModelConfig":
        """Create config with device-appropriate defaults.

        CUDA (24GB+): seq_len=1024, checkpointing, full parallel attention
        MPS (9GB):    seq_len=512,  checkpointing, chunked attention
        CPU:          seq_len=256,  checkpointing, chunked attention
        """
        defaults = {}
        if device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 40:
                defaults = {"max_seq_len": 1024, "use_checkpoint": True}
            elif vram_gb >= 20:
                defaults = {"max_seq_len": 1024, "use_checkpoint": True}
            else:
                defaults = {"max_seq_len": 512, "use_checkpoint": True}
        elif device.type == "mps":
            defaults = {"max_seq_len": 512, "use_checkpoint": True}
        else:
            defaults = {"max_seq_len": 256, "use_checkpoint": True}

        defaults.update(kwargs)
        return cls(**defaults)
