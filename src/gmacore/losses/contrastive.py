"""Contrastive objectives.

Two forms are provided:

`info_nce_loss`
    The standard queue-based InfoNCE objective of Equations 15 and 21, in which
    every stored negative contributes with equal weight. This is the objective
    used by the ManualAug-CL baseline and by GMACORE.

`age_weighted_info_nce_loss`
    The training-aware objective of Equation 24, in which each stored negative
    is weighted by gamma_i = gamma ** delta_t_i according to its residence time
    in the memory queue. This is the objective used by GMACORE-TM.

Both return the mean negative log-likelihood of the positive pair, matching the
leading negative sign of the manuscript formulation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_LOG_EPS = 1e-12


def _prepare(query: torch.Tensor, positive_key: torch.Tensor, negatives: torch.Tensor):
    query = F.normalize(query, dim=1)
    positive_key = F.normalize(positive_key.detach(), dim=1)
    negatives = F.normalize(negatives.detach(), dim=1)
    return query, positive_key, negatives


def info_nce_loss(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Queue-based InfoNCE with uniformly weighted negatives."""
    query, positive_key, negatives = _prepare(query, positive_key, negatives)

    logits_pos = (query * positive_key).sum(dim=1, keepdim=True)
    logits_neg = query @ negatives.t()

    logits = torch.cat([logits_pos, logits_neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def age_weighted_info_nce_loss(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negatives: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Age-weighted InfoNCE corresponding to Equation 24.

    The weight gamma_i multiplies the exponentiated similarity of negative i in
    the denominator. Because the loss is evaluated in log-space for numerical
    stability, this is implemented by adding log(gamma_i) to the corresponding
    logit, which is algebraically identical:

        gamma_i * exp(s_i / tau) == exp(s_i / tau + log gamma_i)

    Scaling the similarity itself by gamma_i, rather than the exponential, is a
    different and incorrect objective: it shrinks the similarity of aged
    negatives towards zero instead of reducing their weight in the partition
    function.
    """
    query, positive_key, negatives = _prepare(query, positive_key, negatives)

    if weights.numel() != negatives.size(0):
        raise ValueError(
            f"Expected {negatives.size(0)} decay weights, received {weights.numel()}"
        )

    logits_pos = (query * positive_key).sum(dim=1, keepdim=True) / temperature
    logits_neg = (query @ negatives.t()) / temperature

    log_weights = torch.log(weights.detach().clamp_min(_LOG_EPS)).unsqueeze(0)
    logits_neg = logits_neg + log_weights

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    # The positive logit occupies index 0, so the cross-entropy target is 0.
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def contrastive_loss(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch to the weighted or unweighted objective."""
    if weights is None:
        return info_nce_loss(query, positive_key, negatives, temperature)
    return age_weighted_info_nce_loss(query, positive_key, negatives, weights, temperature)
