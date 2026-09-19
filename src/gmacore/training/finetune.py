"""Downstream fine-tuning on the MoleculeNet benchmarks.

Two transfer protocols are supported:

`finetune`
    The pretrained encoder and a fresh prediction head are trained jointly,
    typically with a lower learning rate on the encoder. This is the protocol
    used for the results reported in Table 4.

`linear_probe`
    The encoder is frozen and only the prediction head is trained. This
    isolates the quality of the pretrained representation from the capacity
    gained during fine-tuning, and is the appropriate protocol for the
    limited-labelled-data transfer study.

Model selection uses the validation split; the reported score is taken from the
test split at the selected epoch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics, is_higher_better, primary_metric
from ..models.encoder import GraphEncoder, PredictionHead

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Optimization settings for downstream transfer."""

    epochs: int = 100
    batch_size: int = 32
    encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-6
    dropout: float = 0.20
    head_hidden_dim: Optional[int] = None
    protocol: str = "finetune"  # "finetune" or "linear_probe"
    patience: int = 30
    grad_clip: Optional[float] = 5.0
    num_workers: int = 0
    device: str = "cuda"
    normalize_targets: bool = True


class DownstreamModel(nn.Module):
    """Pretrained encoder with a task-specific prediction head."""

    def __init__(
        self,
        encoder: GraphEncoder,
        num_tasks: int,
        dropout: float = 0.2,
        head_hidden_dim: Optional[int] = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # The pooled representation, not the projection head output, is the
        # transfer target: the projection head is discarded after pretraining.
        embedding_dim = encoder.convs[-1].out_channels
        self.head = PredictionHead(
            input_dim=embedding_dim,
            num_tasks=num_tasks,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(self, data) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                self.encoder.eval()
                features = self.encoder.encode(data)
        else:
            features = self.encoder.encode(data)
        return self.head(features)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over the observed labels only."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the observed labels only."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return predictions.sum() * 0.0
    return F.mse_loss(predictions[mask], targets[mask])


class TargetScaler:
    """Standardizes regression targets using training split statistics.

    QM9 targets span several orders of magnitude, so an unnormalized
    multi-task regression loss is dominated by the largest-scale property.
    Predictions are inverse-transformed before metrics are computed, so the
    reported RMSE remains in the original units.
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, targets: np.ndarray) -> TargetScaler:
        self.mean = np.nanmean(targets, axis=0)
        self.std = np.nanstd(targets, axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=targets.dtype, device=targets.device)
        std = torch.tensor(self.std, dtype=targets.dtype, device=targets.device)
        return (targets - mean) / std

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        return predictions * self.std + self.mean


class Finetuner:
    """Trains and evaluates a downstream model on one benchmark."""

    def __init__(
        self,
        model: DownstreamModel,
        config: FinetuneConfig,
        task_type: str,
        scaler: Optional[TargetScaler] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.task_type = task_type
        self.scaler = scaler
        self.device = device or torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        parameter_groups = [{"params": self.model.head.parameters(), "lr": config.head_lr}]
        if config.protocol != "linear_probe":
            parameter_groups.append(
                {"params": self.model.encoder.parameters(), "lr": config.encoder_lr}
            )

        self.optimizer = torch.optim.Adam(parameter_groups, weight_decay=config.weight_decay)
        self.loss_fn = masked_bce_loss if task_type == "classification" else masked_mse_loss

    # ------------------------------------------------------------------
    def _targets(self, batch) -> torch.Tensor:
        targets = batch.y.view(batch.num_graphs, -1).float()
        if self.task_type == "regression" and self.scaler is not None:
            targets = self.scaler.transform(targets)
        return targets

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        if self.model.freeze_encoder:
            self.model.encoder.eval()

        total, batches = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(batch)
            loss = self.loss_fn(outputs, self._targets(batch))
            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total += float(loss.item())
            batches += 1

        return total / max(batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_outputs, all_targets = [], []

        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch).cpu().numpy()
            targets = batch.y.view(batch.num_graphs, -1).float().cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(targets)

        outputs = np.concatenate(all_outputs, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        if self.task_type == "regression" and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)

        return compute_metrics(self.task_type, targets, outputs)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        metric_name = primary_metric(self.task_type)
        higher_better = is_higher_better(metric_name)

        best_valid = -np.inf if higher_better else np.inf
        best_test: Dict[str, float] = {}
        best_epoch = 0
        epochs_without_improvement = 0

        progress = tqdm(range(1, self.config.epochs + 1), desc="Fine-tuning", leave=False)
        for epoch in progress:
            train_loss = self.train_epoch(train_loader)
            valid_metrics = self.evaluate(valid_loader)
            score = valid_metrics[metric_name]

            improved = score > best_valid if higher_better else score < best_valid
            if improved and not np.isnan(score):
                best_valid = score
                best_test = self.evaluate(test_loader)
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            progress.set_postfix(loss=f"{train_loss:.4f}", valid=f"{score:.4f}")

            if self.config.patience > 0 and epochs_without_improvement >= self.config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        result = dict(best_test)
        result[f"valid_{metric_name}"] = float(best_valid)
        result["best_epoch"] = best_epoch
        return result


def build_loaders(
    dataset,
    split: Tuple[List[int], List[int], List[int]],
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test loaders from index lists."""
    train_idx, valid_idx, test_idx = split
    return (
        DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(Subset(dataset, valid_idx), batch_size=batch_size, shuffle=False),
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False),
    )


def load_pretrained_encoder(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[GraphEncoder, Dict]:
    """Reconstruct the pretrained encoder from a pretraining checkpoint."""
    from ..utils.checkpoint import load_checkpoint

    payload = load_checkpoint(checkpoint_path, map_location=device)
    config = payload["framework_config"]

    encoder = GraphEncoder(
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.0),
    )

    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in payload["model_state_dict"].items()
        if key.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        logger.warning("Missing encoder keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected encoder keys: %s", unexpected)

    return encoder, config


def build_random_encoder(node_dim: int, edge_dim: int, **kwargs) -> GraphEncoder:
    """Randomly initialized encoder, used as the no-pretraining control."""
    return GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, **kwargs)
