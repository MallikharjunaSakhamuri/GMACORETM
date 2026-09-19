"""Tests for featurization and dataset splitting."""

import pytest

pytest.importorskip("rdkit")

from gmacore.data.featurizer import (
    FeaturizerConfig,
    MolecularFeaturizer,
    feature_dimensions,
)
from gmacore.data.splits import murcko_scaffold, random_split, scaffold_split

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "C1CCCCC1",
    "CCN(CC)CC",
]


def test_declared_dimensions_match_produced_tensors():
    featurizer = MolecularFeaturizer()
    graph = featurizer("CCO")
    node_dim, edge_dim = feature_dimensions()

    assert graph is not None
    assert graph.x_cat.size(1) + graph.x_phys.size(1) == node_dim
    assert graph.edge_attr.size(1) == edge_dim


def test_edges_are_stored_in_both_directions():
    graph = MolecularFeaturizer()("CCO")
    assert graph.edge_index.size(1) % 2 == 0
    assert graph.edge_index.size(1) == 4  # two bonds, both directions


def test_invalid_smiles_returns_none():
    assert MolecularFeaturizer()("not_a_molecule") is None


def test_single_atom_molecule_is_rejected():
    """Isolated atoms have no edges and carry no message-passing signal."""
    assert MolecularFeaturizer()("C") is None


def test_labels_are_attached():
    graph = MolecularFeaturizer()("CCO", label=[1.0, 0.0])
    assert graph.y.shape == (1, 2)


def test_bond_length_is_zero_without_a_conformer():
    """The default featurization is 2D, so the geometric channel must be zero."""
    graph = MolecularFeaturizer(FeaturizerConfig(generate_conformer=False))("CCO")
    assert graph.edge_attr[:, 4].abs().sum() == 0


def test_featurize_many_reports_failures():
    graphs, failed = MolecularFeaturizer().featurize_many(
        SMILES + ["invalid"], progress=False
    )
    assert len(graphs) == len(SMILES)
    assert failed == ["invalid"]


# ----------------------------------------------------------------------
def test_scaffold_split_is_a_partition():
    train, valid, test = scaffold_split(SMILES)
    combined = sorted(train + valid + test)
    assert combined == list(range(len(SMILES)))
    assert not (set(train) & set(valid) & set(test))


def test_scaffold_split_is_deterministic():
    assert scaffold_split(SMILES) == scaffold_split(SMILES)


def test_random_split_is_seed_stable():
    assert random_split(100, seed=7) == random_split(100, seed=7)
    assert random_split(100, seed=7) != random_split(100, seed=8)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        scaffold_split(SMILES, 0.5, 0.2, 0.2)


def test_scaffold_of_an_acyclic_molecule_is_empty():
    assert murcko_scaffold("CCO") == ""
