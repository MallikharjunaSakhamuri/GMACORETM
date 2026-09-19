"""Unified pretraining framework covering the three model variants.

A single class implements all three configurations reported in the manuscript,
so that the variants differ only in which components are switched on and share
an identical encoder, queue and optimization path.

    variant            augmentation   key encoder        negative weighting
    -----------------  -------------  -----------------  ------------------
    manualaug_cl       handcrafted    query encoder      uniform
    gmacore            generative     query encoder      uniform
    gmacore_tm         generative     momentum encoder   age-based decay

This makes the ablation exact: any difference between `gmacore` and
`gmacore_tm` is attributable to the momentum encoder and the age-based
weighting alone.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from ..losses.adversarial import (
    adversarial_alignment_loss,
    discriminator_loss,
    similarity_regularizer,
)
from ..losses.contrastive import contrastive_loss
from .encoder import GraphEncoder
from .generator import GraphGenerator
from .manual_augment import ManualAugmentConfig, ManualAugmenter
from .memory_queue import MemoryQueue

VARIANTS = ("manualaug_cl", "gmacore", "gmacore_tm")


@dataclass
class FrameworkConfig:
    """Architecture and objective hyperparameters.

    Defaults reproduce Table 3 of the manuscript.
    """

    node_dim: int
    edge_dim: int
    variant: str = "gmacore_tm"

    # Encoder
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10

    # Generator
    generator_hidden_dim: int = 256
    noise_dim: int = 32
    drop_threshold: float = 0.25
    hard_drop: bool = True

    # Contrastive objective
    temperature: float = 0.07
    queue_size: int = 50_000

    # Stability regularization
    momentum: float = 0.999
    decay: float = 0.99

    # Loss weights
    similarity_weight: float = 0.10
    minimax_discriminator: bool = False

    manual_augment: ManualAugmentConfig = field(default_factory=ManualAugmentConfig)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, received '{self.variant}'")

    @property
    def uses_generator(self) -> bool:
        return self.variant in ("gmacore", "gmacore_tm")

    @property
    def uses_momentum_encoder(self) -> bool:
        return self.variant == "gmacore_tm"

    @property
    def uses_age_weighting(self) -> bool:
        return self.variant == "gmacore_tm"

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["manual_augment"] = asdict(self.manual_augment)
        return payload


class GMACoreFramework(nn.Module):
    """Self-supervised pretraining framework for molecular graphs."""

    def __init__(self, config: FrameworkConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = GraphEncoder(
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # The key encoder. For GMACORE-TM it is an EMA copy updated without
        # gradients; for the other variants the query encoder itself supplies
        # the keys, matching an end-to-end contrastive architecture.
        if config.uses_momentum_encoder:
            self.momentum_encoder = copy.deepcopy(self.encoder)
            for parameter in self.momentum_encoder.parameters():
                parameter.requires_grad = False
        else:
            self.momentum_encoder = None

        if config.uses_generator:
            self.generator = GraphGenerator(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                hidden_dim=config.generator_hidden_dim,
                noise_dim=config.noise_dim,
                drop_threshold=config.drop_threshold,
                hard_drop=config.hard_drop,
            )
            # A dedicated discriminator evaluates semantic consistency between
            # the original and augmented graphs (Equation 11). It is distinct
            # from the contrastive encoder so that the adversarial signal does
            # not directly reshape the representation space.
            self.discriminator = GraphEncoder(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
            )
            self.manual_augmenter = None
        else:
            self.generator = None
            self.discriminator = None
            self.manual_augmenter = ManualAugmenter(config.manual_augment)

        self.memory_queue = MemoryQueue(
            size=config.queue_size, dim=config.output_dim, decay=config.decay
        )

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    @torch.no_grad()
    def momentum_update(self) -> None:
        """EMA synchronization of the key encoder (Equation 19)."""
        if self.momentum_encoder is None:
            return
        m = self.config.momentum
        for query_param, key_param in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            key_param.data.mul_(m).add_(query_param.data, alpha=1.0 - m)
        for query_buf, key_buf in zip(self.encoder.buffers(), self.momentum_encoder.buffers()):
            key_buf.data.copy_(query_buf.data)

    @torch.no_grad()
    def encode_keys(self, data) -> torch.Tensor:
        """Encode the key view without gradients."""
        encoder = self.momentum_encoder if self.momentum_encoder is not None else self.encoder
        was_training = encoder.training
        encoder.eval()
        keys = encoder(data)
        if was_training:
            encoder.train()
        return keys

    def augment(self, data):
        """Produce the augmented query view for the configured variant."""
        if self.config.uses_generator:
            node_scores, edge_scores = self.generator(data)
            return self.generator.perturb(data, node_scores, edge_scores)
        return self.manual_augmenter(data)

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------
    def contrastive_step(self, data) -> Dict[str, torch.Tensor]:
        """Compute the contrastive and similarity terms for the encoder update.

        The generator is held fixed here; it is updated separately by
        `generator_step` so that the two objectives do not share a backward
        pass through the same graph.
        """
        if self.config.uses_generator:
            with torch.no_grad():
                node_scores, edge_scores = self.generator(data)
            augmented = self.generator.perturb(
                data, node_scores.detach(), edge_scores.detach()
            )
        else:
            augmented = self.manual_augmenter(data)

        query = self.encoder(augmented)
        keys = self.encode_keys(data)

        negatives = self.memory_queue.negatives()
        if negatives.size(0) == 0:
            # Queue is empty on the first step; fall back to in-batch negatives.
            negatives = keys.detach()
            weights = None
        elif self.config.uses_age_weighting:
            weights = self.memory_queue.decay_weights(enabled=True)[
                self.memory_queue.valid_mask()
            ]
        else:
            weights = None

        loss_contrastive = contrastive_loss(
            query=query,
            positive_key=keys,
            negatives=negatives,
            temperature=self.config.temperature,
            weights=weights,
        )

        with torch.no_grad():
            reference = self.encoder(data)
        loss_similarity = similarity_regularizer(query, reference)

        total = loss_contrastive + self.config.similarity_weight * loss_similarity

        return {
            "contrastive": loss_contrastive,
            "similarity": loss_similarity,
            "encoder_total": total,
            "keys": keys,
        }

    def discriminator_step(self, data) -> torch.Tensor:
        """Discriminator update on a detached augmented view (Equation 14)."""
        if not self.config.uses_generator:
            raise RuntimeError("The active variant has no generator or discriminator.")

        with torch.no_grad():
            node_scores, edge_scores = self.generator(data)
            augmented = self.generator.perturb(data, node_scores, edge_scores)

        original_embedding = self.discriminator(data)
        augmented_embedding = self.discriminator(augmented)
        return discriminator_loss(
            original_embedding,
            augmented_embedding,
            minimax=self.config.minimax_discriminator,
        )

    def generator_step(self, data) -> torch.Tensor:
        """Generator update through the frozen discriminator (Equation 13)."""
        if not self.config.uses_generator:
            raise RuntimeError("The active variant has no generator.")

        node_scores, edge_scores = self.generator(data)
        augmented = self.generator.perturb(data, node_scores, edge_scores)

        with torch.no_grad():
            original_embedding = self.discriminator(data)
        augmented_embedding = self.discriminator(augmented)
        return adversarial_alignment_loss(original_embedding, augmented_embedding)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, data, use_projection: bool = False) -> torch.Tensor:
        """Graph-level representation for downstream use.

        The pooled representation before the projection head is the standard
        transfer target in contrastive pretraining and is the default here.
        """
        self.encoder.eval()
        return self.encoder(data) if use_projection else self.encoder.encode(data)
