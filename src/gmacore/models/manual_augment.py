"""Handcrafted molecular graph augmentations for the ManualAug-CL baseline.

These are the predefined transformations described in Section 2: node dropping,
edge perturbation and node feature masking. They operate on a batched PyG
object and are applied stochastically, one operation sampled per call, which is
the standard protocol for manual graph contrastive augmentation.

This module contains the baseline against which the generative augmentation of
GMACORE is compared. It is deliberately simple and involves no learned
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Data


@dataclass
class ManualAugmentConfig:
    """Probabilities and rates governing the handcrafted augmentations."""

    node_drop_rate: float = 0.20
    edge_drop_rate: float = 0.20
    feature_mask_rate: float = 0.20
    operations: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.operations is None:
            self.operations = ["node_drop", "edge_drop", "feature_mask"]


def _clone(data) -> Data:
    augmented = Data(
        x_cat=data.x_cat.clone().float(),
        x_phys=data.x_phys.clone().float(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone().float(),
    )
    augmented.batch = getattr(data, "batch", None)
    augmented.num_nodes = data.x_cat.size(0)
    return augmented


def node_drop(data, rate: float) -> Data:
    """Zero the features of a random subset of nodes.

    Node indices are retained rather than removed, so that `batch` assignment
    and `edge_index` remain valid. Masking the features of a dropped node is
    equivalent for a mean-pooled message-passing encoder while avoiding a
    costly reindexing of the batch.
    """
    augmented = _clone(data)
    num_nodes = augmented.x_cat.size(0)
    keep = (torch.rand(num_nodes, 1, device=augmented.x_cat.device) >= rate).float()
    augmented.x_cat = augmented.x_cat * keep
    augmented.x_phys = augmented.x_phys * keep
    return augmented


def edge_drop(data, rate: float) -> Data:
    """Remove a random subset of edges.

    Both directions of an undirected bond are stored consecutively, so the mask
    is generated per bond and repeated, keeping the augmented graph symmetric.
    """
    augmented = _clone(data)
    num_directed = augmented.edge_index.size(1)
    if num_directed == 0:
        return augmented

    num_bonds = num_directed // 2
    bond_mask = torch.rand(num_bonds, device=augmented.edge_index.device) >= rate
    mask = bond_mask.repeat_interleave(2)
    if mask.numel() < num_directed:  # odd edge count, retain the remainder
        pad = torch.ones(num_directed - mask.numel(), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, pad])

    if mask.sum() == 0:  # never return an edgeless graph
        return augmented

    augmented.edge_index = augmented.edge_index[:, mask]
    augmented.edge_attr = augmented.edge_attr[mask]
    return augmented


def feature_mask(data, rate: float) -> Data:
    """Zero a random subset of the physicochemical node feature channels."""
    augmented = _clone(data)
    num_channels = augmented.x_phys.size(1)
    mask = (torch.rand(1, num_channels, device=augmented.x_phys.device) >= rate).float()
    augmented.x_phys = augmented.x_phys * mask
    return augmented


class ManualAugmenter:
    """Samples and applies one handcrafted augmentation per call."""

    def __init__(self, config: ManualAugmentConfig = None) -> None:
        self.config = config or ManualAugmentConfig()

    def __call__(self, data) -> Data:
        operations = self.config.operations
        choice = operations[int(torch.randint(len(operations), (1,)).item())]

        if choice == "node_drop":
            return node_drop(data, self.config.node_drop_rate)
        if choice == "edge_drop":
            return edge_drop(data, self.config.edge_drop_rate)
        if choice == "feature_mask":
            return feature_mask(data, self.config.feature_mask_rate)
        raise ValueError(f"Unknown augmentation operation '{choice}'")
