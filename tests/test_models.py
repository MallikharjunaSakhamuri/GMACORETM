"""Tests for the encoder, generator and the combined framework."""

import pytest
import torch

from gmacore.models import FrameworkConfig, GMACoreFramework, GraphEncoder, GraphGenerator
from gmacore.models.manual_augment import ManualAugmenter, edge_drop, node_drop

NODE_DIM, EDGE_DIM = 10, 5


def build(variant: str, **kwargs) -> GMACoreFramework:
    return GMACoreFramework(
        FrameworkConfig(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            variant=variant,
            hidden_dim=16,
            output_dim=16,
            queue_size=32,
            **kwargs,
        )
    )


# ----------------------------------------------------------------------
def test_encoder_returns_one_embedding_per_graph(batch):
    encoder = GraphEncoder(NODE_DIM, EDGE_DIM, hidden_dim=16, output_dim=16)
    embeddings = encoder(batch)
    assert embeddings.shape == (batch.num_graphs, 16)
    assert torch.isfinite(embeddings).all()


def test_generator_scores_lie_in_unit_interval(batch):
    generator = GraphGenerator(NODE_DIM, EDGE_DIM, hidden_dim=16, noise_dim=4)
    node_scores, edge_scores = generator(batch)

    assert node_scores.shape == (batch.x_cat.size(0), 1)
    assert edge_scores.shape == (batch.edge_index.size(1), 1)
    assert (node_scores >= 0).all() and (node_scores <= 1).all()
    assert (edge_scores >= 0).all() and (edge_scores <= 1).all()


def test_generator_noise_makes_augmentation_stochastic(batch):
    """Two calls must differ, otherwise the latent noise is not being used."""
    torch.manual_seed(0)
    generator = GraphGenerator(NODE_DIM, EDGE_DIM, hidden_dim=16, noise_dim=4)
    first, _ = generator(batch)
    second, _ = generator(batch)
    assert not torch.allclose(first, second)


def test_generator_gradients_reach_scoring_layers(batch):
    """The straight-through threshold must not sever the generator gradient."""
    generator = GraphGenerator(NODE_DIM, EDGE_DIM, hidden_dim=16, noise_dim=4, hard_drop=True)
    node_scores, edge_scores = generator(batch)
    augmented = generator.perturb(batch, node_scores, edge_scores)

    (augmented.x_phys.sum() + augmented.edge_attr.sum()).backward()

    gradient = generator.node_scorer[0].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_perturbation_preserves_graph_shape(batch):
    generator = GraphGenerator(NODE_DIM, EDGE_DIM, hidden_dim=16, noise_dim=4)
    node_scores, edge_scores = generator(batch)
    augmented = generator.perturb(batch, node_scores, edge_scores)

    assert augmented.x_cat.shape == batch.x_cat.shape
    assert augmented.edge_index.shape == batch.edge_index.shape
    assert augmented.batch is not None


# ----------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["manualaug_cl", "gmacore", "gmacore_tm"])
def test_variant_components(variant):
    model = build(variant)

    assert (model.generator is not None) == (variant != "manualaug_cl")
    assert (model.discriminator is not None) == (variant != "manualaug_cl")
    assert (model.momentum_encoder is not None) == (variant == "gmacore_tm")
    assert model.config.uses_age_weighting == (variant == "gmacore_tm")


@pytest.mark.parametrize("variant", ["manualaug_cl", "gmacore", "gmacore_tm"])
def test_contrastive_step_produces_a_backward_pass(batch, variant):
    model = build(variant)
    model.train()

    outputs = model.contrastive_step(batch)
    assert torch.isfinite(outputs["encoder_total"])

    outputs["encoder_total"].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_momentum_encoder_receives_no_gradient(batch):
    model = build("gmacore_tm")
    model.contrastive_step(batch)["encoder_total"].backward()
    assert all(p.grad is None for p in model.momentum_encoder.parameters())


def test_momentum_update_moves_key_encoder_towards_query():
    model = build("gmacore_tm", momentum=0.5)

    query_param = next(model.encoder.parameters())
    key_param = next(model.momentum_encoder.parameters())

    with torch.no_grad():
        query_param.add_(1.0)
    before = key_param.detach().clone()

    model.momentum_update()
    expected = 0.5 * before + 0.5 * query_param.detach()
    assert torch.allclose(key_param, expected, atol=1e-6)


def test_non_tm_variants_use_the_query_encoder_for_keys(batch):
    model = build("gmacore")
    model.eval()
    keys = model.encode_keys(batch)
    assert torch.allclose(keys, model.encoder(batch), atol=1e-5)


def test_generator_and_discriminator_steps_are_finite(batch):
    model = build("gmacore_tm")
    assert torch.isfinite(model.discriminator_step(batch))
    assert torch.isfinite(model.generator_step(batch))


def test_manualaug_variant_rejects_generator_steps(batch):
    model = build("manualaug_cl")
    with pytest.raises(RuntimeError):
        model.generator_step(batch)


def test_embed_returns_pooled_representation(batch):
    model = build("gmacore_tm")
    pooled = model.embed(batch, use_projection=False)
    projected = model.embed(batch, use_projection=True)
    assert pooled.shape == projected.shape == (batch.num_graphs, 16)
    assert not torch.allclose(pooled, projected)


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError):
        FrameworkConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, variant="not_a_variant")


# ----------------------------------------------------------------------
def test_manual_augmentations_change_the_graph(batch):
    dropped = node_drop(batch, rate=1.0)
    assert dropped.x_phys.abs().sum() == 0

    thinned = edge_drop(batch, rate=0.5)
    assert thinned.edge_index.size(1) <= batch.edge_index.size(1)


def test_manual_augmenter_keeps_the_batch_assignment(batch):
    augmented = ManualAugmenter()(batch)
    assert augmented.batch is not None
    assert augmented.x_cat.size(0) == batch.x_cat.size(0)
