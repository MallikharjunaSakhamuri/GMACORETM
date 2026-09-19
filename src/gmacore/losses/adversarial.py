"""Adversarial alignment objective for the generative augmentation module.

Equation 12 defines the adversarial loss as the squared distance between the
discriminator representations of the original and augmented graphs:

    L_adv = || D(G) - D(G') ||^2

Equations 13 and 14 then update the generator and the discriminator
alternately. Note that both updates descend the same quantity, so the
formulation is an alignment objective rather than a minimax game: the generator
is pushed towards augmentations that the discriminator embeds close to the
original graph, which is what makes the resulting views usable as positive
pairs for contrastive learning.

The zero-sum variant, in which the discriminator instead maximizes the
distance, is provided behind `adversarial_alignment_loss(..., minimax=True)`
for ablation. It is not the configuration used for the reported results.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adversarial_alignment_loss(
    original_embedding: torch.Tensor,
    augmented_embedding: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance between original and augmented representations."""
    return F.mse_loss(augmented_embedding, original_embedding)


def discriminator_loss(
    original_embedding: torch.Tensor,
    augmented_embedding: torch.Tensor,
    minimax: bool = False,
) -> torch.Tensor:
    """Discriminator objective.

    With `minimax=False` the discriminator aligns the two representations,
    matching Equation 14. With `minimax=True` it separates them, giving the
    conventional zero-sum adversarial game.
    """
    loss = adversarial_alignment_loss(original_embedding, augmented_embedding)
    return -loss if minimax else loss


def similarity_regularizer(
    query_embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
) -> torch.Tensor:
    """Penalty keeping the augmented view close to the original in encoder space.

    This term prevents the learned augmentation from collapsing onto views that
    are trivially separable from the source molecule.
    """
    return F.mse_loss(query_embedding, reference_embedding.detach())
