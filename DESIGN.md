# Design notes

This document maps the manuscript onto the implementation and records the
decisions that are not obvious from the code alone.

---

## Equation to code map

| Manuscript | Location |
| --- | --- |
| Eq. 7, node encoding of concatenated features | `models/encoder.py::normalize_node_features`, `GraphEncoder.node_encoder` |
| Eq. 8, graph convolution in the generator | `models/generator.py::GraphGenerator.forward` |
| Eq. 9, noise-conditioned importance scores | `models/generator.py::GraphGenerator.forward` |
| Eq. 10, feature modulation and dropping | `models/generator.py::GraphGenerator.perturb` |
| Eq. 11, discriminator | `models/framework.py`, a second `GraphEncoder` |
| Eq. 12, adversarial loss | `losses/adversarial.py::adversarial_alignment_loss` |
| Eq. 13, 14, alternating updates | `training/pretrain.py::Pretrainer.train_epoch` |
| Eq. 15, contrastive loss (GMACORE) | `losses/contrastive.py::info_nce_loss` |
| Eq. 16 to 18, encoder and pooling | `models/encoder.py::GraphEncoder.encode` |
| Eq. 19, EMA momentum update | `models/framework.py::GMACoreFramework.momentum_update` |
| Eq. 20, query and key embeddings | `models/framework.py::contrastive_step` |
| Eq. 21, InfoNCE with momentum keys | `losses/contrastive.py::info_nce_loss` |
| Eq. 22, sample age | `models/memory_queue.py::MemoryQueue.age` |
| Eq. 23, decay weight gamma_i | `models/memory_queue.py::MemoryQueue.decay_weights` |
| Eq. 24, age-weighted contrastive loss | `losses/contrastive.py::age_weighted_info_nce_loss` |
| Sec. 3.5, representation drift | `evaluation/representation.py::representation_drift` |
| Sec. 3.4, encoder-key misalignment | `evaluation/representation.py::encoder_key_alignment` |

---

## Why the age weight multiplies the exponential

Equation 24 places `gamma_i` outside the exponential:

```
L = -log [ exp(q.k / tau) / ( exp(q.k / tau) + sum_i gamma_i exp(q.z_i / tau) ) ]
```

`gamma_i` is a weight on the negative's contribution to the partition function.
Because it appears inside a sum of exponentials, the loss must be evaluated in
log-space to avoid overflow, and the weight becomes an additive term on the
logit:

```
gamma_i * exp(s_i / tau) = exp(s_i / tau + log gamma_i)
```

The tempting shortcut, multiplying the similarity `s_i` by `gamma_i` before the
softmax, implements something different. It contracts each aged negative's
similarity towards zero, which for a negative with high similarity reduces its
influence as intended, but for a negative with strongly negative similarity
*increases* it. The weighting is meant to be monotone in influence, so the
log-space form is the correct one. This is tested explicitly.

---

## Why age is a defensible proxy for mismatch

A reviewer asked why sample age should reflect representation mismatch. The
argument implemented here is deliberately modest and matches the revised
manuscript text.

Age counts the number of encoder updates that have elapsed since a key was
produced. Each update moves the encoder parameters by a bounded step, so the
displacement of the representation space after `k` updates is bounded by a
quantity increasing in `k`. Age therefore upper-bounds, rather than measures,
the mismatch. It is used because it is exact, free to compute and requires no
re-encoding.

The alternative, re-encoding every queued negative with the current encoder to
measure mismatch directly, would cost a forward pass over the entire queue at
every step and would defeat the purpose of a queue. A useful ablation, which
`evaluation/representation.py` supports, is to measure the realized mismatch
periodically and check that it is in fact increasing in age. If it is not, the
proxy is not doing what the paper claims.

---

## Why the discriminator is a separate network

Equation 11 defines the discriminator `D_phi` independently of the contrastive
encoder `f_theta`. Keeping them separate matters: if the adversarial loss were
computed on the contrastive encoder, gradients from the generator's objective
would flow directly into the representation used for downstream transfer, and
the augmentation module would be able to shape the embedding space to make its
own job easier. With a separate discriminator, the generator's only route to
the representation is through the quality of the views it produces.

---

## Why the pooled representation transfers, not the projection

The projection head is trained to make the contrastive objective easy to
satisfy, which discards information that the objective does not need but a
downstream task might. Standard practice in contrastive pretraining is to
discard the projection head after pretraining and transfer the layer beneath
it. `GMACoreFramework.embed` and `DownstreamModel` both default to the pooled
representation, with the projection available via `use_projection=True` for
ablation.

---

## Why the adversarial objective is alignment rather than minimax

Equations 13 and 14 have the generator and the discriminator descending the
same quantity `||D(G) - D(G')||^2`. This is unusual: it is an alignment
objective, not a zero-sum game. The consequence is that the generator is
rewarded for producing augmentations the discriminator cannot distinguish from
the original, which is what makes the augmented view usable as a *positive*
pair. A true minimax formulation would push the generator towards views that
are maximally distinguishable, which would be wrong for contrastive positives.

The nomenclature "adversarial" is therefore somewhat loose in the manuscript.
The implementation follows the equations, and `minimax_discriminator: true`
exposes the zero-sum variant for ablation.

---

## Open items requiring author resolution

These are inconsistencies between the manuscript and the implementation that
cannot be resolved from the code alone.

1. **Attention heads.** Table 3 lists four attention heads and a feed-forward
   dimension of 256 for the GMACORE encoder, but the implemented encoder uses
   `GCNConv`, which has neither. Either the table should describe a GCN
   encoder, or the encoder should be replaced by a graph attention network and
   the experiments rerun. A reviewer checking the table against the released
   code will find this immediately.

2. **Base decay coefficient.** Table 3 gives `gamma = 0.99`; the notebook used
   `0.99999`. These produce very different weighting profiles. The
   configurations follow the table; confirm which value produced the reported
   results.

3. **Queue size and batch size.** Table 3 gives 50,000 and 64; the notebook
   used 65,536 and 32.

4. **Warm-up phase.** The notebook ran ten epochs of contrastive-only training
   before the main loop. This is not described in the manuscript.

5. **Contrastive margin.** Table 3 lists a "contrastive margin" of 1.0 and a
   "matching score function" of dot-product similarity. The InfoNCE objective
   in Equations 15, 21 and 24 has no margin term. Either the margin is unused
   and should be removed from the table, or an additional margin-based matching
   loss exists that is not described in the methods.

6. **Three-dimensional features.** The notebook generated conformers and used a
   bond-length feature; the Limitations section states that three-dimensional
   information is not incorporated. The implementation follows the manuscript.

7. **QM9 target set.** QM9 is a 12-task benchmark in DeepChem
   (`mu, alpha, homo, lumo, gap, r2, zpve, cv, u0, u298, h298, g298`). Table 4
   reports a single RMSE of 2.057 for QM9 without stating which targets it
   covers. Those targets differ by orders of magnitude in scale, so a single
   averaged RMSE is not interpretable or reproducible unless the target set is
   fixed. The repository defaults to a documented six-target subset and exposes
   `target_columns`, but the manuscript must state which targets were used.

8. **QM9 source file.** DeepChem's `load_qm9` reads `qm9.sdf` (from
   `qm9.tar.gz`), not `qm9.csv`. This repository uses `qm9.csv` because the
   pipeline is SMILES-driven throughout. The two sources differ in their energy
   columns: the SDF exposes atomization energies (`u0_atom`, `u298_atom`, and
   so on, in kcal/mol) whereas the CSV carries total energies in Hartree.
   Confirm which was used before quoting a QM9 RMSE.

Resolving items 1, 2, 3, 6 and 7 is a prerequisite for the repository to
withstand the reproducibility check that Reviewer 1, Comment 7 is asking for.
