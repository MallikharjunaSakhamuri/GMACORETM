"""Quantitative analysis of the pretrained representation space.

Consumes the per-epoch embedding checkpoints written during pretraining and
reports representation drift, encoder-key alignment, centroid trajectory
consistency and embedding dispersion. These are the numerical counterparts to
the UMAP figures.

Example
-------
    python scripts/analyze_representations.py \
        --embedding-dir runs/gmacore_tm/embeddings \
        --output runs/gmacore_tm/representation_analysis.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gmacore.evaluation.representation import (
    centroid_trajectory,
    embedding_dispersion,
    encoder_key_alignment,
    representation_drift,
)


def load_epoch_embeddings(directory: str) -> Dict[int, np.ndarray]:
    """Load `embeddings_epoch_NNN.pt` files keyed by epoch number."""
    embeddings: Dict[int, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(directory, "embeddings_epoch_*.pt"))):
        match = re.search(r"embeddings_epoch_(\d+)\.pt$", path)
        if match is None:
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        embeddings[int(match.group(1))] = tensor.numpy()
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze representation stability.")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_epoch = load_epoch_embeddings(args.embedding_dir)
    if len(by_epoch) < 3:
        raise SystemExit(
            f"Found {len(by_epoch)} embedding checkpoints in {args.embedding_dir}; "
            "at least three are required."
        )

    epochs: List[int] = sorted(by_epoch)
    sequence = [by_epoch[epoch] for epoch in epochs]

    report = {
        "epochs": epochs,
        "consecutive_drift": {},
        "consecutive_alignment": {},
        "final_dispersion": embedding_dispersion(sequence[-1]),
        "centroid_trajectory": centroid_trajectory(sequence),
        "total_drift": representation_drift(sequence[0], sequence[-1]),
    }

    for earlier, later in zip(epochs[:-1], epochs[1:]):
        key = f"{earlier}->{later}"
        report["consecutive_drift"][key] = representation_drift(by_epoch[earlier], by_epoch[later])
        report["consecutive_alignment"][key] = encoder_key_alignment(
            by_epoch[earlier], by_epoch[later]
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Analysis written to {args.output}")
    print("Centroid directional consistency: "
          f"{report['centroid_trajectory']['directional_consistency']:.4f}")
    print(f"Mean drift (first to last): {report['total_drift']['drift_mean']:.6f}")


if __name__ == "__main__":
    main()
