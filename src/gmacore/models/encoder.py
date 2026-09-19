"""Graph encoder producing graph-level molecular representations.

The encoder implements Equations 16 to 18 of the manuscript: a shared MLP node
encoder over the concatenated categorical and physicochemical node features, an
edge encoder, `L` graph convolution layers, mean pooling over nodes, and a
projection head onto the contrastive embedding space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def normalize_node_features(
    x_cat: torch.Tensor, x_phys: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Concatenate the two node feature groups into a single float tensor.

    Categorical indices are cast to float. Physicochemical descriptors are
    standardized across the nodes of the current batch, which keeps the widely
    differing descriptor scales (atomic weight against formal charge, for
    instance) within a comparable range.
    """
    x_cat = x_cat.float()
    x_phys = x_phys.float()
    if x_phys.size(0) > 1:
        x_phys = (x_phys - x_phys.mean(dim=0)) / (x_phys.std(dim=0) + eps)
    return torch.cat([x_cat, x_phys], dim=-1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def init_linear_weights(module: nn.Module) -> None:
    """Xavier initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)


class GraphEncoder(nn.Module):
    """Message-passing encoder with a contrastive projection head."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_encoder = _mlp(node_dim, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            out_channels = output_dim if layer == num_layers - 1 else hidden_dim
            self.convs.append(GCNConv(hidden_dim, out_channels))

        self.dropout = dropout
        self.projection = _mlp(output_dim, hidden_dim, output_dim)
        self.apply(init_linear_weights)

    def encode(self, data) -> torch.Tensor:
        """Return the pooled graph representation before projection."""
        x = normalize_node_features(data.x_cat, data.x_phys)
        x = self.node_encoder(x)

        # Edge attributes are encoded for architectural symmetry with the
        # generator. GCNConv consumes scalar edge weights only, so the encoded
        # edge representation is not passed into the convolutions.
        _ = self.edge_encoder(data.edge_attr.float())

        edge_index = data.edge_index
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)

    def forward(self, data) -> torch.Tensor:
        """Return the projected contrastive embedding."""
        return self.projection(self.encode(data))


class PredictionHead(nn.Module):
    """Downstream head attached to the frozen or fine-tuned encoder."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_tasks))
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_tasks),
            )
        self.apply(init_linear_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
