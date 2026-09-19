"""Shared utilities: configuration, logging, seeding and checkpoints.

`config` and `logging` depend only on the standard library and PyYAML;
`seed` and `checkpoint` require PyTorch and are exposed lazily so that the
former two remain importable without it.
"""

from typing import TYPE_CHECKING

from .config import apply_overrides, load_yaml, merge
from .logging import MetricHistory, configure_logging

__all__ = [
    "MetricHistory",
    "apply_overrides",
    "configure_logging",
    "load_checkpoint",
    "load_yaml",
    "merge",
    "resolve_device",
    "save_checkpoint",
    "set_seed",
]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .checkpoint import load_checkpoint, save_checkpoint
    from .seed import resolve_device, set_seed

_LAZY = {
    "load_checkpoint": "checkpoint",
    "save_checkpoint": "checkpoint",
    "resolve_device": "seed",
    "set_seed": "seed",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
