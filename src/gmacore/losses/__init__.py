"""Contrastive and adversarial objectives."""

from .adversarial import (
    adversarial_alignment_loss,
    discriminator_loss,
    similarity_regularizer,
)
from .contrastive import age_weighted_info_nce_loss, contrastive_loss, info_nce_loss

__all__ = [
    "adversarial_alignment_loss",
    "age_weighted_info_nce_loss",
    "contrastive_loss",
    "discriminator_loss",
    "info_nce_loss",
    "similarity_regularizer",
]
