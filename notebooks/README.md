# Notebooks

`legacy_gmacore_tm.ipynb` is the original exploratory development notebook. It
is retained for provenance only and is **not** the entry point for reproducing
the results.

The notebook differs from the packaged implementation in several ways that
affect numerical behaviour, including the form of the age weighting, the
absence of latent noise conditioning in the generator, and an undocumented
warm-up phase. These differences are enumerated in
[`../docs/CHANGES.md`](../docs/CHANGES.md).

Use `scripts/pretrain.py` and `scripts/finetune.py` instead.
