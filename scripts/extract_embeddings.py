"""Extract graph-level embeddings from a pretrained checkpoint.

Writes an `.npz` archive containing the embeddings and the corresponding SMILES
strings, which is the input format expected by `scripts/analyze_representations.py`.

Example
-------
    python scripts/extract_embeddings.py \
        --checkpoint runs/gmacore_tm/best_model.pt \
        --smiles-file data/pubchem/pubchem_10k.txt \
        --output runs/gmacore_tm/embeddings.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gmacore.data import FeaturizerConfig, PretrainGraphDataset
from gmacore.models import FrameworkConfig, GMACoreFramework
from gmacore.models.manual_augment import ManualAugmentConfig
from gmacore.utils import configure_logging, load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use-projection",
        action="store_true",
        help="Return the projection head output rather than the pooled representation.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    framework_config = dict(payload["framework_config"])
    manual_cfg = framework_config.pop("manual_augment", {}) or {}

    model = GMACoreFramework(
        FrameworkConfig(manual_augment=ManualAugmentConfig(**manual_cfg), **framework_config)
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()

    dataset = PretrainGraphDataset.from_smiles_file(
        path=args.smiles_file, limit=args.limit, featurizer_config=FeaturizerConfig()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            chunks.append(model.embed(batch, use_projection=args.use_projection).cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0)
    smiles = np.array([graph.smiles for graph in dataset.graphs], dtype=object)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, embeddings=embeddings, smiles=smiles)
    print(
        f"Wrote {embeddings.shape[0]} embeddings of dimension "
        f"{embeddings.shape[1]} to {args.output}"
    )


if __name__ == "__main__":
    main()
