"""Tests for the contrastive objectives."""

import math

import pytest
import torch

from gmacore.losses.contrastive import (
    age_weighted_info_nce_loss,
    contrastive_loss,
    info_nce_loss,
)


@pytest.fixture
def triplet():
    torch.manual_seed(0)
    query = torch.randn(8, 16)
    positive = torch.randn(8, 16)
    negatives = torch.randn(64, 16)
    return query, positive, negatives


def test_info_nce_is_positive_and_finite(triplet):
    loss = info_nce_loss(*triplet)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_info_nce_rewards_aligned_positives(triplet):
    query, _, negatives = triplet
    aligned = info_nce_loss(query, query.clone(), negatives)
    misaligned = info_nce_loss(query, -query.clone(), negatives)
    assert aligned.item() < misaligned.item()


def test_unit_weights_recover_unweighted_loss(triplet):
    query, positive, negatives = triplet
    weights = torch.ones(negatives.size(0))

    weighted = age_weighted_info_nce_loss(query, positive, negatives, weights)
    unweighted = info_nce_loss(query, positive, negatives)
    assert torch.allclose(weighted, unweighted, atol=1e-6)


def test_decaying_weights_reduce_the_loss(triplet):
    """Down-weighting negatives shrinks the partition function, so the negative
    log-likelihood of the positive pair decreases."""
    query, positive, negatives = triplet
    decayed = torch.full((negatives.size(0),), 0.1)

    weighted = age_weighted_info_nce_loss(query, positive, negatives, decayed)
    unweighted = info_nce_loss(query, positive, negatives)
    assert weighted.item() < unweighted.item()


def test_weight_applied_to_exponential_not_similarity():
    """gamma * exp(s/tau) must equal exp(s/tau + log gamma)."""
    torch.manual_seed(1)
    query = torch.randn(1, 8)
    positive = torch.randn(1, 8)
    negatives = torch.randn(3, 8)
    gamma = 0.5
    temperature = 0.07

    weights = torch.full((3,), gamma)
    actual = age_weighted_info_nce_loss(query, positive, negatives, weights, temperature)

    q = torch.nn.functional.normalize(query, dim=1)
    p = torch.nn.functional.normalize(positive, dim=1)
    n = torch.nn.functional.normalize(negatives, dim=1)

    pos_term = math.exp(float((q * p).sum()) / temperature)
    neg_term = sum(gamma * math.exp(float(q @ n[i]) / temperature) for i in range(3))
    expected = -math.log(pos_term / (pos_term + neg_term))

    assert actual.item() == pytest.approx(expected, rel=1e-5)


def test_weight_count_must_match_negatives(triplet):
    query, positive, negatives = triplet
    with pytest.raises(ValueError):
        age_weighted_info_nce_loss(query, positive, negatives, torch.ones(3))


def test_dispatch_matches_direct_calls(triplet):
    query, positive, negatives = triplet
    assert torch.allclose(
        contrastive_loss(query, positive, negatives),
        info_nce_loss(query, positive, negatives),
    )
