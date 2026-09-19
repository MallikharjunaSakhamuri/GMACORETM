"""MoleculeNet downstream benchmarks.

Five benchmarks are used in the manuscript: BACE, BBBP and ClinTox for binary
classification (ROC-AUC), and ESOL and QM9 for regression (RMSE). Two further
benchmarks, HIV and Lipophilicity, are registered here for the extended
evaluation described in the response to reviewers.

CSV files are expected under `data/moleculenet/`. Use
`scripts/download_moleculenet.py` to retrieve them from the DeepChem mirror.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .featurizer import FeaturizerConfig, MolecularFeaturizer

logger = logging.getLogger(__name__)

QM9_DEFAULT_TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "cv"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static description of a MoleculeNet benchmark."""

    name: str
    filename: str
    smiles_column: str
    task_type: str  # "classification" or "regression"
    target_columns: List[str] = field(default_factory=list)
    default_metric: str = "roc_auc"
    default_split: str = "scaffold"
    url: Optional[str] = None


_DEEPCHEM = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets"

BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "bace": BenchmarkSpec(
        name="bace",
        filename="bace.csv",
        smiles_column="mol",
        task_type="classification",
        target_columns=["Class"],
        default_metric="roc_auc",
        url=f"{_DEEPCHEM}/bace.csv",
    ),
    "bbbp": BenchmarkSpec(
        name="bbbp",
        filename="BBBP.csv",
        smiles_column="smiles",
        task_type="classification",
        target_columns=["p_np"],
        default_metric="roc_auc",
        url=f"{_DEEPCHEM}/BBBP.csv",
    ),
    "clintox": BenchmarkSpec(
        name="clintox",
        filename="clintox.csv",
        smiles_column="smiles",
        task_type="classification",
        target_columns=["FDA_APPROVED", "CT_TOX"],
        default_metric="roc_auc",
        url=f"{_DEEPCHEM}/clintox.csv.gz",
    ),
    "hiv": BenchmarkSpec(
        name="hiv",
        filename="HIV.csv",
        smiles_column="smiles",
        task_type="classification",
        target_columns=["HIV_active"],
        default_metric="roc_auc",
        url=f"{_DEEPCHEM}/HIV.csv",
    ),
    "esol": BenchmarkSpec(
        name="esol",
        filename="delaney-processed.csv",
        smiles_column="smiles",
        task_type="regression",
        target_columns=["measured log solubility in mols per litre"],
        default_metric="rmse",
        url=f"{_DEEPCHEM}/delaney-processed.csv",
    ),
    "lipophilicity": BenchmarkSpec(
        name="lipophilicity",
        filename="Lipophilicity.csv",
        smiles_column="smiles",
        task_type="regression",
        target_columns=["exp"],
        default_metric="rmse",
        url=f"{_DEEPCHEM}/Lipophilicity.csv",
    ),
    "qm9": BenchmarkSpec(
        name="qm9",
        filename="qm9.csv",
        smiles_column="smiles",
        task_type="regression",
        target_columns=QM9_DEFAULT_TARGETS,
        default_metric="rmse",
        default_split="random",
        url=f"{_DEEPCHEM}/qm9.csv",
    ),
}


class MoleculeNetDataset(Dataset):
    """Featurized MoleculeNet benchmark with graph-level targets.

    Missing labels are represented as NaN and are masked out of the loss and of
    the per-task metrics, which is the standard protocol for the multi-task
    MoleculeNet benchmarks.
    """

    def __init__(self, graphs: List[Data], spec: BenchmarkSpec) -> None:
        self.graphs = graphs
        self.spec = spec

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]

    @property
    def num_tasks(self) -> int:
        return len(self.spec.target_columns)

    @property
    def task_type(self) -> str:
        return self.spec.task_type

    @property
    def smiles(self) -> List[str]:
        return [graph.smiles for graph in self.graphs]

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        name: str,
        root: str = "data/moleculenet",
        target_columns: Optional[Sequence[str]] = None,
        cache_dir: Optional[str] = "data/cache",
        featurizer_config: Optional[FeaturizerConfig] = None,
    ) -> MoleculeNetDataset:
        key = name.lower()
        if key not in BENCHMARKS:
            raise KeyError(f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARKS)}")

        spec = BENCHMARKS[key]
        if target_columns is not None:
            spec = BenchmarkSpec(
                name=spec.name,
                filename=spec.filename,
                smiles_column=spec.smiles_column,
                task_type=spec.task_type,
                target_columns=list(target_columns),
                default_metric=spec.default_metric,
                default_split=spec.default_split,
                url=spec.url,
            )

        cache_path = None
        if cache_dir is not None:
            tag = "-".join(spec.target_columns)[:64]
            cache_path = os.path.join(cache_dir, f"{spec.name}_{tag}.pt")
            if os.path.exists(cache_path):
                logger.info("Loading cached %s graphs from %s", spec.name, cache_path)
                return cls(torch.load(cache_path, weights_only=False), spec)

        csv_path = os.path.join(root, spec.filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run `python scripts/download_moleculenet.py "
                f"--datasets {spec.name}` first."
            )

        frame = pd.read_csv(csv_path)
        missing = [c for c in spec.target_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Columns {missing} not present in {csv_path}")

        smiles = frame[spec.smiles_column].astype(str).tolist()
        labels = frame[spec.target_columns].astype(float).values.tolist()

        featurizer = MolecularFeaturizer(featurizer_config)
        graphs, failed = featurizer.featurize_many(smiles, labels)
        logger.info(
            "%s: featurized %d molecules (%d skipped)", spec.name, len(graphs), len(failed)
        )

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(graphs, cache_path)

        return cls(graphs, spec)
