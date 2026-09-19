"""Unlabeled pretraining corpus of SMILES-derived molecular graphs.

The corpus used in the manuscript is the ChemBERTa-compiled PubChem SMILES
collection. The loader accepts any newline-delimited SMILES file and caches the
featurized graphs so that repeated runs do not repeat RDKit preprocessing.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .featurizer import FeaturizerConfig, MolecularFeaturizer

logger = logging.getLogger(__name__)


def read_smiles_file(path: str, limit: Optional[int] = None) -> List[str]:
    """Read newline-delimited SMILES, ignoring blanks and comment lines."""
    smiles: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            # Tolerate simple delimited files by taking the first column.
            smiles.append(value.split()[0].split(",")[0])
            if limit is not None and len(smiles) >= limit:
                break
    return smiles


class PretrainGraphDataset(Dataset):
    """In-memory dataset of unlabeled molecular graphs."""

    def __init__(self, graphs: List[Data]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]

    # ------------------------------------------------------------------
    @classmethod
    def from_smiles_file(
        cls,
        path: str,
        cache_path: Optional[str] = None,
        limit: Optional[int] = None,
        featurizer_config: Optional[FeaturizerConfig] = None,
    ) -> PretrainGraphDataset:
        """Build the dataset, reusing a cached featurization when available."""
        if cache_path is not None and os.path.exists(cache_path):
            logger.info("Loading cached pretraining graphs from %s", cache_path)
            graphs = torch.load(cache_path, weights_only=False)
            return cls(graphs)

        smiles = read_smiles_file(path, limit=limit)
        logger.info("Read %d SMILES strings from %s", len(smiles), path)

        featurizer = MolecularFeaturizer(featurizer_config)
        graphs, failed = featurizer.featurize_many(smiles)
        logger.info("Featurized %d graphs (%d skipped)", len(graphs), len(failed))

        if not graphs:
            raise RuntimeError(f"No valid molecular graphs were produced from {path}")

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(graphs, cache_path)
            logger.info("Cached pretraining graphs to %s", cache_path)

        return cls(graphs)
