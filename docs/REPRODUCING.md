# Reproducing the reported results

## Hardware

The results in the manuscript were produced on a workstation with an NVIDIA
GeForce RTX 3050 GPU, an 11th generation Intel Core i7-11370H CPU at 3.30 GHz
and 16 GB of RAM. The framework runs on any CUDA-capable GPU and on CPU,
although CPU pretraining on the full corpus is impractical.

## Hyperparameters

The defaults in `configs/pretrain_gmacore_tm.yaml` correspond to Table 3 of the
manuscript:

| Group | Parameter | Value |
| --- | --- | --- |
| Encoder | embedding dimension | 128 |
| Encoder | message-passing layers | 3 |
| Encoder | dropout | 0.10 |
| Contrastive | temperature | 0.07 |
| Contrastive | memory queue size | 50,000 |
| Stability | momentum coefficient | 0.999 |
| Stability | base decay coefficient | 0.99 |
| Training | batch size | 64 |
| Training | learning rate | 5e-4 |
| Training | weight decay | 1e-4 |
| Training | epochs | 150 |

## Command sequence

```bash
# 1. Data
python scripts/download_moleculenet.py --all
# place the PubChem SMILES file at data/pubchem/pubchem_10m_clean.txt

# 2. Pretraining, one run per variant
python scripts/pretrain.py --config configs/pretrain_manualaug_cl.yaml
python scripts/pretrain.py --config configs/pretrain_gmacore.yaml
python scripts/pretrain.py --config configs/pretrain_gmacore_tm.yaml

# 3. Downstream evaluation
for variant in manualaug_cl gmacore gmacore_tm; do
  python scripts/finetune.py --config configs/finetune_classification.yaml \
      --checkpoint runs/${variant}/best_model.pt \
      --output-dir runs/finetune/${variant}_classification
  python scripts/finetune.py --config configs/finetune_regression.yaml \
      --checkpoint runs/${variant}/best_model.pt \
      --output-dir runs/finetune/${variant}_regression
done

# 4. Representation analysis
for variant in manualaug_cl gmacore gmacore_tm; do
  python scripts/analyze_representations.py \
      --embedding-dir runs/${variant}/embeddings \
      --output runs/${variant}/representation_analysis.json
done
```

`scripts/run_all.sh` wraps this sequence.

## Expected runtime

On the reference hardware, with a 100,000 molecule pretraining corpus:

| Stage | Approximate duration |
| --- | --- |
| Featurization and caching | 20 to 40 minutes, once |
| Pretraining, ManualAug-CL | 4 to 6 hours |
| Pretraining, GMACORE | 7 to 10 hours |
| Pretraining, GMACORE-TM | 8 to 11 hours |
| Fine-tuning, one benchmark, three seeds | 10 to 40 minutes |

GMACORE and GMACORE-TM are slower per epoch than the baseline because each step
additionally performs a discriminator update and a generator update. GMACORE-TM
adds an EMA parameter update and an age-weighted denominator, both of which are
inexpensive relative to the generator.

## Determinism

`pretrain.seed` seeds Python, NumPy and PyTorch. Setting
`pretrain.deterministic: true` additionally places cuDNN in deterministic mode.

Residual nondeterminism remains from three sources:

1. Atomic operations in scatter-based GPU kernels are not order-stable, so
   message passing can differ in the last bits between runs.
2. The generator samples fresh latent noise at every forward pass by design.
3. Multi-worker data loading interleaves batches nondeterministically. Set
   `pretrain.num_workers: 0` when exact reproduction is required.

Reported downstream results are averaged over three seeds and are given as
`mean +/- standard deviation`; individual seeds vary by a few tenths of a
ROC-AUC point on the smaller benchmarks.

## Verifying against the paper

The clearest single check is the ordering of the three variants on the
contrastive loss trajectory: `manualaug_cl` should remain highest throughout,
`gmacore` should sit below it, and `gmacore_tm` should converge fastest and
lowest. Compare `training_history.json` across the three runs before
interpreting downstream numbers.
