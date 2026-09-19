# Differences from the exploratory notebook

The original development notebook (`notebooks/legacy_gmacore_tm.ipynb`) is
retained for provenance. This document records where the packaged
implementation departs from it, and why. Items marked **correctness** change
numerical behaviour and require rerunning any affected experiment; items marked
**consistency** align the code with the manuscript without changing results;
items marked **engineering** are structural only.

---

## 1. Age weighting applied to the exponential, not the similarity

**Correctness.** The notebook computed

```python
l_neg = l_neg * decay_weights.unsqueeze(0)
logits = torch.cat([l_pos, l_neg], dim=1) / temperature
```

which scales the similarity of each negative by `gamma_i` before
exponentiation. Equation 24 of the manuscript specifies

```
gamma_i * exp(q . z_i / tau)
```

that is, the weight multiplies the exponentiated similarity in the denominator.
The two are not equivalent. Scaling the similarity pulls an aged negative's
similarity towards zero, which makes it look like an *orthogonal* negative
rather than a *less influential* one; a negative with a strongly negative
similarity is actually pushed towards zero from below, increasing its
contribution.

The implementation in `losses/contrastive.py` adds `log(gamma_i)` to the
negative logit, which is algebraically identical to weighting the exponential
and is numerically stable:

```python
logits_neg = (query @ negatives.t()) / temperature + log_weights
```

`tests/test_losses.py::test_weight_applied_to_exponential_not_similarity`
checks this against a closed-form evaluation.

## 2. Base decay coefficient

**Consistency.** The notebook used `decay = 0.99999`, which over a queue of
50,000 entries yields a weight of approximately 0.61 for the oldest entry and
is effectively no decay at all over the first few thousand steps. Table 3
specifies `gamma = 0.99`. The configuration files use 0.99. If the reported
numbers were produced with 0.99999, either the manuscript table or the
experiments need to be brought into agreement; this is flagged in
`docs/DESIGN.md` as an open item for the authors.

## 3. Latent noise conditioning in the generator

**Correctness.** Equation 9 conditions the node and edge importance scores on a
latent noise vector `z_noise ~ N(0, I)`:

```
s_v_i = sigma(psi_v([h_i || z_noise]))
s_e_ij = sigma(psi_e([h_i || h_j || z_noise]))
```

The notebook's generator had no noise input, so the augmentation was fully
deterministic given the input graph. This removes the stochasticity the paper
relies on for producing multiple plausible variants of the same molecule, and
means the positive pair is identical at every epoch.

`models/generator.py` samples one noise vector per graph and broadcasts it
across that graph's nodes and edges. Sampling per graph rather than per node
keeps the perturbation coherent within a molecule.

## 4. Threshold-based component dropping

**Correctness.** The manuscript states that graph components scoring below a
threshold are dropped. The notebook only multiplied features by their
continuous score, so nothing was ever removed. The commented-out alternative in
the notebook used a hard mask that would have severed the gradient to the
scoring MLPs entirely, leaving the generator untrainable.

The implementation uses a straight-through estimator: the hard mask is applied
in the forward pass and the gradient passes through unchanged in the backward
pass. Set `framework.hard_drop: false` to recover the purely continuous
modulation.

## 5. Separate discriminator and encoder

**Consistency.** The notebook defined a discriminator but, in an earlier
revision, computed the adversarial loss against the contrastive encoder. The
adversarial signal then directly reshaped the representation space. Equation 11
defines the discriminator as a distinct network, and the framework keeps the
two separate.

## 6. Queue enqueue ordering

**Correctness.** The notebook enqueued keys at various points relative to the
encoder update depending on the training phase. Because the recorded age is
meant to measure the number of encoder updates elapsed since a key was
produced, keys must be enqueued after the encoder step that generated them. The
training loop fixes this ordering, and it is documented inline.

## 7. Queue validity tracking

**Engineering.** The notebook initialized the queue with random normalized
vectors and treated all 65,536 slots as valid negatives from the first step. At
the start of training almost all negatives were therefore pure noise. The
`MemoryQueue` tracks which slots have been written and exposes only those, with
an in-batch fallback on the very first step.

## 8. Queue registered as buffers

**Engineering.** The notebook stored the queue as plain tensor attributes and
moved them to the device manually. They were therefore absent from
`state_dict()` and lost on checkpoint reload. They are now registered buffers,
so they move with `.to(device)` and are captured in checkpoints.

## 9. Temperature

**Consistency.** The notebook defined a temperature annealing schedule from
0.10 to 0.05 in `forward`, but the training loop called
`compute_contrastive_loss` with the fixed `config.temperature` of 0.07, so the
schedule never took effect. Table 3 specifies a fixed `tau = 0.07`. The
annealing code has been removed rather than left as dead code.

## 10. Three-dimensional conformer generation

**Consistency.** The notebook called `AllChem.EmbedMolecule` and MMFF or UFF
optimization for every molecule, and derived a bond-length feature from the
resulting conformer. The manuscript's Limitations section states explicitly
that three-dimensional conformational information is not incorporated. These
are in direct contradiction.

Conformer generation is now off by default, so the featurization is
two-dimensional and the bond-length channel is zero, matching the paper. It can
be re-enabled with `data.generate_conformer: true`. Conformer embedding was
also the dominant preprocessing cost.

## 11. Explicit hydrogens

**Consistency.** The notebook called `Chem.AddHs`, adding every hydrogen as a
graph node, which roughly doubles graph size and is not mentioned in the
manuscript. This is off by default and available as
`data.add_explicit_hydrogens`.

## 12. Undocumented Phase 1 pretraining

**Correctness.** The notebook ran ten epochs of contrastive learning without
the generator before the main loop. This phase does not appear in the
manuscript, and it means the reported GMACORE results include ten epochs of
what is effectively an unaugmented baseline. The packaged training loop has a
single phase. If the warm-up was intentional it should be described in the
manuscript and reintroduced as an explicit configuration option.

## 13. Duplicated and unreachable code in the training loop

**Engineering.** The periodic embedding-extraction block in the notebook sat
outside the epoch loop, so it executed once after training rather than every
ten epochs, and `model.train()` was called twice in succession. Embedding
extraction is now controlled by `pretrain.embedding_every` and
`pretrain.embedding_epochs`.

## 14. Missing downstream code

**Engineering.** The notebook contained no fine-tuning, evaluation or metric
code, so none of Table 4 could be reproduced from it. `training/finetune.py`,
`evaluation/metrics.py` and `scripts/finetune.py` supply this, including
scaffold splitting, masked multi-task losses, target standardization for QM9,
early stopping on a validation split and multi-seed aggregation.

## 15. Unused metadata extraction

**Engineering.** The notebook's `extract_molecule_metadata` computed graph
diameter, assortativity, clustering coefficients and ring statistics for every
molecule. None of these quantities is used by the model or reported in the
manuscript, and the computation is expensive on large corpora. It has been
removed from the training path. The descriptors actually used for Figure 13 are
computed in the analysis stage instead.

## 16. Hyperparameters aligned with Table 3

**Consistency.** The notebook used batch size 32, encoder learning rate 3e-4,
50 epochs and a queue size of 65,536. Table 3 specifies batch size 64,
learning rate 5e-4, 150 epochs and a queue size of 50,000. The configuration
files follow Table 3.

## 17. Attention heads in Table 3

**Open item.** Table 3 lists "Number of attention heads: 4" and "Feed-forward
dimension: 256" under the GMACORE encoder. The implemented encoder uses
`GCNConv` layers, which have neither attention heads nor a feed-forward
sublayer. Either the table describes a different architecture than the one
used, or the encoder should be a graph attention network. This must be
resolved before release, since a reviewer comparing the table against the code
will notice it. See `docs/DESIGN.md`.
