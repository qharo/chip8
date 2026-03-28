"""PyTorch Dataset for CHIP-8 traces."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from data.tokenizer import Tokenizer


class TraceDataset(Dataset):
    """Tokenized trace sequences for next-token prediction.

    Each sample is a window of `seq_len + 1` tokens.
    Input X = tokens[0:seq_len], Target Y = tokens[1:seq_len+1].
    """

    def __init__(self, token_ids: list[int], seq_len: int = 1024,
                 pad_id: int = 0):
        self.seq_len = seq_len
        self.pad_id = pad_id

        # Pre-compute all valid starting positions
        self.starts = list(range(0, len(token_ids) - seq_len, seq_len))

        # Store as tensor for fast indexing
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        chunk = self.data[start:start + self.seq_len + 1]

        # Pad if necessary
        if len(chunk) < self.seq_len + 1:
            padding = torch.full((self.seq_len + 1 - len(chunk),),
                                 self.pad_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding])

        x = chunk[:self.seq_len]
        y = chunk[1:self.seq_len + 1]
        return x, y
