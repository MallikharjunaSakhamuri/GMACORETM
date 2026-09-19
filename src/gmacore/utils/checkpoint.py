"""Checkpoint serialization.

Checkpoints store the full framework state together with the configuration
needed to rebuild the architecture, so that a downstream run can reconstruct
the encoder without also being given the training script's arguments.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model,
    epoch: int,
    metrics: Optional[Dict] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
) -> None:
    """Write a checkpoint to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "framework_config": model.config.to_dict(),
        "metrics": metrics or {},
    }
    if optimizers:
        payload["optimizer_state_dicts"] = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }

    torch.save(payload, path)
    logger.debug("Saved checkpoint to %s", path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    """Read a checkpoint from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
