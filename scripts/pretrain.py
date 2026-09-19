"""Self-supervised pretraining entry point.

Examples
--------
    python scripts/pretrain.py --config configs/pretrain_gmacore_tm.yaml
    python scripts/pretrain.py --config configs/pretrain_gmacore.yaml \
        --set pretrain.epochs=50 data.limit=10000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gmacore.data import FeaturizerConfig, PretrainGraphDataset, feature_dimensions
from gmacore.models import FrameworkConfig, GMACoreFramework
from gmacore.models.manual_augment import ManualAugmentConfig
from gmacore.training import PretrainConfig, Pretrainer
from gmacore.utils import apply_overrides, configure_logging, load_yaml, resolve_device, set_seed

logger = logging.getLogger("pretrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a molecular graph encoder.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Dotted configuration overrides, for example pretrain.epochs=50.",
    )
    parser.add_argument("--output-dir", default=None, help="Override the run output directory.")
    parser.add_argument("--device", default=None, help="Override the compute device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args.overrides)

    data_cfg = config.get("data", {})
    framework_cfg = config.get("framework", {})
    pretrain_cfg = config.get("pretrain", {})

    if args.output_dir is not None:
        pretrain_cfg["output_dir"] = args.output_dir
    if args.device is not None:
        pretrain_cfg["device"] = args.device

    output_dir = pretrain_cfg.get("output_dir", "runs/pretrain")
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(log_file=os.path.join(output_dir, "pretrain.log"))

    seed = pretrain_cfg.get("seed", 42)
    set_seed(seed, deterministic=pretrain_cfg.get("deterministic", False))
    device = resolve_device(pretrain_cfg.get("device", "cuda"))
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    featurizer_config = FeaturizerConfig(
        add_explicit_hydrogens=data_cfg.get("add_explicit_hydrogens", False),
        generate_conformer=data_cfg.get("generate_conformer", False),
        max_heavy_atoms=data_cfg.get("max_heavy_atoms"),
        min_heavy_atoms=data_cfg.get("min_heavy_atoms", 2),
    )

    dataset = PretrainGraphDataset.from_smiles_file(
        path=data_cfg["smiles_file"],
        cache_path=data_cfg.get("cache_path"),
        limit=data_cfg.get("limit"),
        featurizer_config=featurizer_config,
    )
    logger.info("Pretraining corpus: %d molecular graphs", len(dataset))

    # ------------------------------------------------------------------
    node_dim, edge_dim = feature_dimensions()
    manual_cfg = framework_cfg.pop("manual_augment", {}) or {}

    framework = GMACoreFramework(
        FrameworkConfig(
            node_dim=node_dim,
            edge_dim=edge_dim,
            manual_augment=ManualAugmentConfig(**manual_cfg),
            **framework_cfg,
        )
    )
    logger.info("Variant: %s", framework.config.variant)
    logger.info(
        "Trainable parameters: %d",
        sum(p.numel() for p in framework.parameters() if p.requires_grad),
    )

    trainer = Pretrainer(framework, PretrainConfig(**pretrain_cfg), device=device)
    trainer.fit(dataset)
    logger.info("Pretraining complete. Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
