"""GMACORE-TM: stability-aware generative adversarial contrastive learning
for self-supervised molecular graph representation.

Reference implementation accompanying the manuscript
"Self-supervised molecular graph representation learning via stability-aware
generative adversarial contrastive learning".

The model classes are exposed lazily so that the pure-NumPy submodules, such as
`gmacore.evaluation.metrics` and `gmacore.utils.config`, remain importable in
an environment without PyTorch installed.
"""

from typing import TYPE_CHECKING

__version__ = "1.0.0"

__all__ = ["FrameworkConfig", "GMACoreFramework", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .models.framework import FrameworkConfig, GMACoreFramework


def __getattr__(name: str):
    if name in {"FrameworkConfig", "GMACoreFramework"}:
        from .models import framework

        return getattr(framework, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
