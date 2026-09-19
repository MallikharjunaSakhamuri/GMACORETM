"""Shared fixtures for the test suite.

Torch and PyTorch Geometric are imported lazily rather than at module scope. A
module-level `importorskip` in a conftest aborts the whole session when the
dependency is absent, which would hide the results of the tests that do not
need it. Collection-time skipping is handled per module by
`collect_ignore_glob` below.
"""

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

NODE_DIM = 10
EDGE_DIM = 5


def _has(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = _has("torch")
HAS_PYG = HAS_TORCH and _has("torch_geometric")
HAS_RDKIT = _has("rdkit")

# Modules whose every test needs an optional dependency are skipped at
# collection time, so a partial environment still reports the tests it can run.
collect_ignore_glob = []
if not HAS_PYG:
    collect_ignore_glob += ["test_models.py", "test_memory_queue.py", "test_losses.py"]
if not HAS_RDKIT:
    collect_ignore_glob += ["test_data.py"]


def make_graph(num_nodes: int = 6, seed: int = 0):
    """Build a synthetic connected molecular graph with the expected schema."""
    import torch
    from torch_geometric.data import Data

    generator = torch.Generator().manual_seed(seed)

    x_cat = torch.randint(0, 20, (num_nodes, 2), generator=generator)
    x_phys = torch.randn(num_nodes, 8, generator=generator)

    # A path graph, stored in both directions as the featurizer does.
    rows, cols = [], []
    for index in range(num_nodes - 1):
        rows += [index, index + 1]
        cols += [index + 1, index]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), EDGE_DIM, generator=generator)

    data = Data(
        x_cat=x_cat,
        x_phys=x_phys,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )
    data.smiles = "C" * num_nodes
    return data


@pytest.fixture
def batch():
    """A batch of four synthetic graphs of differing sizes."""
    from torch_geometric.data import Batch

    return Batch.from_data_list([make_graph(4 + i, seed=i) for i in range(4)])


@pytest.fixture
def feature_dims():
    return NODE_DIM, EDGE_DIM
