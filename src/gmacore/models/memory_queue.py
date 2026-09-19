"""FIFO memory queue of negative representations with age tracking.

The queue stores graph-level key representations produced during earlier
optimization steps, providing a larger pool of negatives than a single
mini-batch. Each slot additionally records the age of its occupant, measured in
optimization steps since insertion, which supports the age-based weighting of
Section 4.2.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryQueue(nn.Module):
    """Fixed-capacity queue of L2-normalized negative keys.

    Buffers are registered on the module so that the queue moves with
    `.to(device)` and is captured in checkpoints.
    """

    def __init__(self, size: int, dim: int, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")

        self.size = size
        self.dim = dim
        self.decay = decay

        self.register_buffer("keys", F.normalize(torch.randn(size, dim), dim=1))
        self.register_buffer("age", torch.zeros(size, dtype=torch.float))
        self.register_buffer("valid", torch.zeros(size, dtype=torch.bool))
        self.register_buffer("pointer", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    @property
    def num_valid(self) -> int:
        """Number of slots that hold a genuine key rather than initial noise."""
        return int(self.valid.sum().item())

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """Insert a batch of keys, evicting the oldest entries when full."""
        keys = F.normalize(keys.detach(), dim=1)
        batch_size = keys.size(0)
        if batch_size == 0:
            return
        if batch_size > self.size:
            raise ValueError(
                f"Batch of {batch_size} keys exceeds queue capacity {self.size}"
            )

        # Every existing entry becomes one optimization step older.
        self.age += 1.0

        pointer = int(self.pointer.item())
        indices = (torch.arange(batch_size, device=keys.device) + pointer) % self.size

        self.keys[indices] = keys
        self.age[indices] = 0.0
        self.valid[indices] = True
        self.pointer[0] = (pointer + batch_size) % self.size

    # ------------------------------------------------------------------
    def decay_weights(self, enabled: bool = True) -> torch.Tensor:
        """Return the age-dependent weights gamma_i of Equation 23.

        With `enabled=False` every valid entry receives unit weight, which
        reproduces the GMACORE behaviour in which all stored negatives are
        treated as equally relevant.
        """
        if not enabled:
            return torch.ones_like(self.age)
        return torch.pow(
            torch.tensor(self.decay, device=self.age.device, dtype=self.age.dtype), self.age
        )

    def negatives(self) -> torch.Tensor:
        """Return the currently valid negative keys and their decay weights."""
        return self.keys[self.valid]

    def valid_mask(self) -> torch.Tensor:
        return self.valid

    def mean_age(self) -> float:
        """Average age of the valid entries, reported for diagnostic logging."""
        if self.num_valid == 0:
            return 0.0
        return float(self.age[self.valid].mean().item())
