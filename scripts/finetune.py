"""Downstream evaluation on the MoleculeNet benchmarks.

Runs the requested benchmarks with a fixed set of seeds and reports
`mean +/- standard deviation` over the runs, which is the format used in
Table 4.

Examples
--------
    python scripts/finetune.py --config configs/finetune_classification.yaml \
        --checkpoint runs/gmacore_tm/best_model.pt

    python scripts/finetune.py --config configs/finetune_regression.yaml \
        --checkpoint runs/gmacore_tm/best_model.pt --datasets esol qm9

    python scripts/finetune.py --config configs/finetune_classification.yaml \
        --no-pretrain    # randomly initialized encoder control
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gmacore.data import FeaturizerConfig, MoleculeNetDataset, feature_dimensions, get_split
from gmacore.data.moleculenet import BENCHMARKS
from gmacore.evaluation.metrics import primary_metric, summarize_runs
from gmacore.training.finetune import (
    DownstreamModel,
    FinetuneConfig,
    Finetuner,
    TargetScaler,
    build_loaders,
    build_random_encoder,
    load_pretrained_encoder,
)
from gmacore.utils import apply_overrides, configure_logging, load_yaml, resolve_device, set_seed

logger = logging.getLogger("finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained encoder downstream.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Pretraining checkpoint to transfer.")
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Use a randomly initialized encoder as the no-pretraining control.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Override the benchmark list.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    return parser.parse_args()


def run_single(
    dataset: MoleculeNetDataset,
    split,
    encoder_factory,
    finetune_config: FinetuneConfig,
    device,
    seed: int,
) -> Dict[str, float]:
    """Train and evaluate one benchmark under one seed."""
    set_seed(seed)

    train_loader, valid_loader, test_loader = build_loaders(
        dataset, split, finetune_config.batch_size, finetune_config.num_workers
    )

    scaler = None
    if dataset.task_type == "regression" and finetune_config.normalize_targets:
        train_targets = np.stack(
            [dataset[i].y.view(-1).numpy() for i in split[0]], axis=0
        )
        scaler = TargetScaler().fit(train_targets)

    model = DownstreamModel(
        encoder=encoder_factory(),
        num_tasks=dataset.num_tasks,
        dropout=finetune_config.dropout,
        head_hidden_dim=finetune_config.head_hidden_dim,
        freeze_encoder=finetune_config.protocol == "linear_probe",
    )

    trainer = Finetuner(
        model=model,
        config=finetune_config,
        task_type=dataset.task_type,
        scaler=scaler,
        device=device,
    )
    return trainer.fit(train_loader, valid_loader, test_loader)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args.overrides)

    data_cfg = config.get("data", {})
    finetune_cfg = config.get("finetune", {})

    datasets: List[str] = args.datasets or config.get("datasets", [])
    seeds: List[int] = args.seeds or config.get("seeds", [0, 1, 2])
    output_dir = args.output_dir or config.get("output_dir", "runs/finetune")
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(log_file=os.path.join(output_dir, "finetune.log"))

    if not datasets:
        raise SystemExit("No benchmarks requested. Use --datasets or set `datasets` in the config.")
    if args.checkpoint is None and not args.no_pretrain:
        raise SystemExit("Provide --checkpoint, or pass --no-pretrain for the control run.")

    finetune_config = FinetuneConfig(**finetune_cfg)
    device = resolve_device(finetune_config.device)
    logger.info("Device: %s | protocol: %s", device, finetune_config.protocol)

    featurizer_config = FeaturizerConfig(
        add_explicit_hydrogens=data_cfg.get("add_explicit_hydrogens", False),
        generate_conformer=data_cfg.get("generate_conformer", False),
    )

    # The encoder is rebuilt from the checkpoint for every run so that each
    # seed starts from the pretrained weights rather than from the previous
    # benchmark's fine-tuned state.
    if args.no_pretrain:
        node_dim, edge_dim = feature_dimensions()
        encoder_kwargs = config.get("encoder", {})

        def encoder_factory():
            return build_random_encoder(node_dim, edge_dim, **encoder_kwargs)

    else:

        def encoder_factory():
            encoder, _ = load_pretrained_encoder(args.checkpoint, device="cpu")
            return encoder

    all_results: Dict[str, Dict] = {}

    for name in datasets:
        spec = BENCHMARKS[name.lower()]
        logger.info("Benchmark %s (%s)", spec.name, spec.task_type)

        dataset = MoleculeNetDataset.load(
            name=name,
            root=data_cfg.get("root", "data/moleculenet"),
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            featurizer_config=featurizer_config,
        )
        split = get_split(
            strategy=data_cfg.get("split", spec.default_split),
            smiles=dataset.smiles,
            seed=data_cfg.get("split_seed", 42),
        )

        runs = []
        for seed in seeds:
            result = run_single(
                dataset, split, encoder_factory, finetune_config, device, seed
            )
            logger.info(
                "  seed %d | %s = %.4f",
                seed,
                primary_metric(dataset.task_type),
                result[primary_metric(dataset.task_type)],
            )
            runs.append(result)

        all_results[spec.name] = {
            "task_type": spec.task_type,
            "num_tasks": dataset.num_tasks,
            "seeds": seeds,
            "runs": runs,
            "summary": summarize_runs(runs),
        }
        logger.info("  %s summary: %s", spec.name, all_results[spec.name]["summary"])

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "protocol": finetune_config.protocol,
                "results": all_results,
            },
            handle,
            indent=2,
        )
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
