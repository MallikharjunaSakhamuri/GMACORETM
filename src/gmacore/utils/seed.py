"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    With `deterministic=True` cuDNN is placed in deterministic mode. This makes
    repeated runs bitwise reproducible at some cost in throughput, and is
    recommended when reproducing the reported numbers rather than for routine
    development.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(preference: str = "cuda") -> torch.device:
    """Return the requested device, falling back to CPU when CUDA is absent."""
    if preference.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)
