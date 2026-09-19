"""Dataset splitting utilities for the downstream benchmarks.

Scaffold splitting is the standard protocol for the MoleculeNet classification
benchmarks and is used for BACE, BBBP, ClinTox, HIV and ESOL. Random splitting
is used for QM9, following common practice for that benchmark.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("RDKit is required for scaffold splitting.") from exc

logger = logging.getLogger(__name__)

Indices = Tuple[List[int], List[int], List[int]]


def _validate_fractions(train: float, valid: float, test: float) -> None:
    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, received {total}")


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold of a molecule as a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    smiles: Sequence[str],
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    include_chirality: bool = False,
) -> Indices:
    """Deterministic scaffold split with the largest scaffold sets in training."""
    _validate_fractions(train_fraction, valid_fraction, test_fraction)

    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for index, value in enumerate(smiles):
        scaffolds[murcko_scaffold(value, include_chirality)].append(index)

    # Largest scaffold groups are assigned first, so the test set is enriched
    # for rare scaffolds and the split measures structural generalization.
    ordered = sorted(scaffolds.values(), key=lambda group: (len(group), group[0]), reverse=True)

    total = len(smiles)
    train_cutoff = train_fraction * total
    valid_cutoff = (train_fraction + valid_fraction) * total

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for group in ordered:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(valid_idx) + len(group) <= valid_cutoff:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    logger.info(
        "Scaffold split: %d train / %d valid / %d test over %d scaffolds",
        len(train_idx),
        len(valid_idx),
        len(test_idx),
        len(scaffolds),
    )
    return train_idx, valid_idx, test_idx


def random_split(
    num_samples: int,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Indices:
    """Shuffled split with a fixed seed for reproducibility."""
    _validate_fractions(train_fraction, valid_fraction, test_fraction)

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(num_samples)

    train_end = int(train_fraction * num_samples)
    valid_end = int((train_fraction + valid_fraction) * num_samples)

    return (
        permutation[:train_end].tolist(),
        permutation[train_end:valid_end].tolist(),
        permutation[valid_end:].tolist(),
    )


def get_split(
    strategy: str,
    smiles: Sequence[str],
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Indices:
    """Dispatch to the requested splitting strategy."""
    strategy = strategy.lower()
    if strategy == "scaffold":
        return scaffold_split(smiles, train_fraction, valid_fraction, test_fraction)
    if strategy == "random":
        return random_split(len(smiles), train_fraction, valid_fraction, test_fraction, seed)
    raise ValueError(f"Unknown split strategy '{strategy}'. Use 'scaffold' or 'random'.")
