"""Downstream metrics and representation-space analysis.

Both submodules depend only on NumPy, SciPy and scikit-learn, so they are
importable without PyTorch.
"""

from .metrics import compute_metrics, primary_metric, summarize_runs
from .representation import (
    centroid_trajectory,
    encoder_key_alignment,
    representation_drift,
    umap_projection,
)

__all__ = [
    "centroid_trajectory",
    "compute_metrics",
    "encoder_key_alignment",
    "primary_metric",
    "representation_drift",
    "summarize_runs",
    "umap_projection",
]
