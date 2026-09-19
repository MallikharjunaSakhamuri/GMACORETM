"""Self-supervised pretraining loop.

Per optimization step the order of operations follows Section 5.3:

    1. discriminator update on a detached augmented view
    2. encoder update on the contrastive objective with age-weighted negatives
    3. EMA synchronization of the momentum encoder
    4. generator update through the frozen discriminator
    5. FIFO enqueue of the new keys with age reset

Steps 1 and 4 are skipped for the ManualAug-CL baseline, and step 3 applies
only to GMACORE-TM.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..models.framework import FrameworkConfig, GMACoreFramework
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import MetricHistory

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig:
    """Optimization settings for pretraining."""

    epochs: int = 150
    batch_size: int = 64
    encoder_lr: float = 5e-4
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 5.0
    num_workers: int = 0
    checkpoint_every: int = 10
    embedding_every: int = 10
    output_dir: str = "runs/pretrain"
    seed: int = 42
    device: str = "cuda"
    # Places cuDNN in deterministic mode. Consumed by the entry point when
    # seeding, and retained here so that it round-trips through the saved
    # configuration rather than being silently dropped.
    deterministic: bool = False
    embedding_epochs: List[int] = field(default_factory=list)


class Pretrainer:
    """Drives self-supervised pretraining and records training diagnostics."""

    def __init__(
        self,
        model: GMACoreFramework,
        config: PretrainConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        self.optimizer_encoder = torch.optim.Adam(
            self.model.encoder.parameters(),
            lr=config.encoder_lr,
            weight_decay=config.weight_decay,
        )
        self.optimizer_generator = None
        self.optimizer_discriminator = None
        if self.model.config.uses_generator:
            self.optimizer_generator = torch.optim.Adam(
                self.model.generator.parameters(), lr=config.generator_lr
            )
            self.optimizer_discriminator = torch.optim.Adam(
                self.model.discriminator.parameters(), lr=config.discriminator_lr
            )

        self.history = MetricHistory()
        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _clip(self, module: torch.nn.Module) -> None:
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_clip)

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {"contrastive": 0.0, "similarity": 0.0, "adversarial": 0.0, "discriminator": 0.0}
        num_batches = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(self.device)

            # 1. Discriminator
            if self.optimizer_discriminator is not None:
                self.optimizer_discriminator.zero_grad(set_to_none=True)
                loss_disc = self.model.discriminator_step(batch)
                loss_disc.backward()
                self._clip(self.model.discriminator)
                self.optimizer_discriminator.step()
                totals["discriminator"] += float(loss_disc.item())

            # 2. Encoder
            self.optimizer_encoder.zero_grad(set_to_none=True)
            outputs = self.model.contrastive_step(batch)
            outputs["encoder_total"].backward()
            self._clip(self.model.encoder)
            self.optimizer_encoder.step()

            totals["contrastive"] += float(outputs["contrastive"].item())
            totals["similarity"] += float(outputs["similarity"].item())

            # 3. Momentum encoder
            self.model.momentum_update()

            # 4. Generator
            if self.optimizer_generator is not None:
                self.optimizer_generator.zero_grad(set_to_none=True)
                loss_gen = self.model.generator_step(batch)
                loss_gen.backward()
                self._clip(self.model.generator)
                self.optimizer_generator.step()
                totals["adversarial"] += float(loss_gen.item())

            # 5. Enqueue keys. Performed after the encoder update so that the
            # queue holds representations of the encoder state that produced
            # them, and their recorded age is measured from that step.
            self.model.memory_queue.enqueue(outputs["keys"])

            num_batches += 1
            progress.set_postfix(
                contrastive=totals["contrastive"] / max(num_batches, 1),
                adversarial=totals["adversarial"] / max(num_batches, 1),
            )

        averages = {key: value / max(num_batches, 1) for key, value in totals.items()}
        averages["queue_mean_age"] = self.model.memory_queue.mean_age()
        averages["queue_occupancy"] = self.model.memory_queue.num_valid
        return averages

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader) -> torch.Tensor:
        """Graph-level embeddings for the full loader, in loader order."""
        self.model.eval()
        chunks = []
        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
            batch = batch.to(self.device)
            chunks.append(self.model.embed(batch).cpu())
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    def fit(self, dataset, eval_loader: Optional[DataLoader] = None) -> MetricHistory:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        if eval_loader is None:
            eval_loader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=False
            )

        embedding_dir = os.path.join(self.config.output_dir, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        torch.save(
            self.extract_embeddings(eval_loader),
            os.path.join(embedding_dir, "embeddings_epoch_000.pt"),
        )

        best_loss = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            metrics = self.train_epoch(loader, epoch)
            self.history.append(epoch, metrics)
            logger.info(
                "Epoch %d | contrastive %.4f | adversarial %.4f | mean queue age %.1f",
                epoch,
                metrics["contrastive"],
                metrics["adversarial"],
                metrics["queue_mean_age"],
            )

            should_save_embeddings = (
                epoch in self.config.embedding_epochs
                or (
                    self.config.embedding_every > 0
                    and epoch % self.config.embedding_every == 0
                )
            )
            if should_save_embeddings:
                torch.save(
                    self.extract_embeddings(eval_loader),
                    os.path.join(embedding_dir, f"embeddings_epoch_{epoch:03d}.pt"),
                )

            if self.config.checkpoint_every > 0 and epoch % self.config.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch:03d}.pt"),
                    model=self.model,
                    epoch=epoch,
                    metrics=metrics,
                )

            if metrics["contrastive"] < best_loss:
                best_loss = metrics["contrastive"]
                save_checkpoint(
                    os.path.join(self.config.output_dir, "best_model.pt"),
                    model=self.model,
                    epoch=epoch,
                    metrics=metrics,
                )

        save_checkpoint(
            os.path.join(self.config.output_dir, "final_model.pt"),
            model=self.model,
            epoch=self.config.epochs,
            metrics=self.history.last(),
        )
        self.history.to_json(os.path.join(self.config.output_dir, "training_history.json"))

        with open(
            os.path.join(self.config.output_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "framework": self.model.config.to_dict(),
                    "pretrain": self.config.__dict__,
                },
                handle,
                indent=2,
            )

        return self.history


def build_framework(node_dim: int, edge_dim: int, **kwargs) -> GMACoreFramework:
    """Convenience constructor used by the training scripts."""
    return GMACoreFramework(FrameworkConfig(node_dim=node_dim, edge_dim=edge_dim, **kwargs))
