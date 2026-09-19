"""Generative augmentation module for molecular graphs.

The generator implements Equations 8 to 10 of the manuscript. Node and edge
importance scores are produced by scoring MLPs conditioned on a shared latent
noise vector, and are used to modulate the input features. Components whose
score falls below a threshold are dropped.

The latent noise vector `z_noise ~ N(0, I)` is sampled once per graph in the
batch and broadcast to that graph's nodes and edges. Sampling per graph rather
than per node ensures the perturbation is coherent across a single molecule
while remaining stochastic across the batch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from .encoder import init_linear_weights, normalize_node_features


class _StraightThroughThreshold(torch.autograd.Function):
    """Hard threshold in the forward pass, identity gradient in the backward pass.

    Dropping a component is a discrete decision and is not differentiable. The
    straight-through estimator lets the discrete mask be applied to the
    augmented view while still propagating gradients into the scoring MLPs, so
    the generator can be trained by the adversarial objective.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, threshold: float) -> torch.Tensor:  # noqa: D102
        return (scores >= threshold).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        return grad_output, None


def straight_through_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return _StraightThroughThreshold.apply(scores, threshold)


class GraphGenerator(nn.Module):
    """Assigns importance scores to nodes and edges and perturbs the graph."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        noise_dim: int = 32,
        num_layers: int = 3,
        drop_threshold: float = 0.25,
        hard_drop: bool = True,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.drop_threshold = drop_threshold
        self.hard_drop = hard_drop

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Equation 9: scores conditioned on the shared latent noise vector.
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_linear_weights)

    # ------------------------------------------------------------------
    def _sample_noise(self, data, device: torch.device) -> torch.Tensor:
        """Sample one noise vector per graph and expand it to node resolution."""
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x_cat.size(0), dtype=torch.long, device=device)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        graph_noise = torch.randn(num_graphs, self.noise_dim, device=device)
        return graph_noise[batch]

    def forward(
        self, data, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-node and per-edge importance scores in (0, 1)."""
        device = data.x_cat.device

        x = normalize_node_features(data.x_cat, data.x_phys)
        x = self.node_encoder(x)
        _ = self.edge_encoder(data.edge_attr.float())

        edge_index = data.edge_index
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)

        node_noise = self._sample_noise(data, device) if noise is None else noise

        node_scores = torch.sigmoid(self.node_scorer(torch.cat([x, node_noise], dim=-1)))

        source, target = edge_index[0], edge_index[1]
        edge_input = torch.cat([x[source], x[target], node_noise[source]], dim=-1)
        edge_scores = torch.sigmoid(self.edge_scorer(edge_input))

        return node_scores, edge_scores

    # ------------------------------------------------------------------
    def perturb(
        self, data, node_scores: torch.Tensor, edge_scores: torch.Tensor
    ) -> Data:
        """Apply the importance scores to produce an augmented view.

        Features are modulated by their score (Equation 10). When `hard_drop` is
        enabled, components scoring below `drop_threshold` are additionally
        zeroed through a straight-through mask, matching the description of
        dropping low-importance graph components.
        """
        node_weight = node_scores
        edge_weight = edge_scores

        if self.hard_drop:
            node_weight = node_weight * straight_through_threshold(
                node_scores, self.drop_threshold
            )
            edge_weight = edge_weight * straight_through_threshold(
                edge_scores, self.drop_threshold
            )

        augmented = Data(
            x_cat=data.x_cat.float() * node_weight,
            x_phys=data.x_phys.float() * node_weight,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float() * edge_weight,
        )
        augmented.batch = getattr(data, "batch", None)
        augmented.num_nodes = data.x_cat.size(0)
        return augmented
