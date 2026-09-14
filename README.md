# GMACORE-TM

**Self-supervised Molecular Graph Representation Learning via Stability-Aware Generative Adversarial Contrastive Learning**

GMACORE-TM is a self-supervised molecular graph representation learning framework that combines generative graph augmentation, momentum-based representation learning, and age-aware weighting of negative samples.

The framework is designed to improve the diversity of molecular graph augmentations while reducing representation instability caused by stale negative samples in memory-based contrastive learning.

---

## Overview

GMACORE-TM consists of three main components:

1. **Generative molecular graph augmentation**

   A graph-based generative adversarial module learns structured perturbations of molecular graphs to provide diverse contrastive views beyond manually defined graph augmentations.

2. **Momentum encoder**

   A slowly updated target encoder is used to generate more stable key representations during contrastive pretraining.

3. **Age-aware negative weighting**

   Negative representations stored in the memory queue are weighted according to their age, reducing the influence of increasingly outdated representations during contrastive optimization.

The learned molecular representations are evaluated on both classification and regression tasks using standard MoleculeNet benchmarks.

---

## Repository Structure

```text
GMACORETM/
├── configs/             # Training and model configurations
├── src/
│   ├── data/            # Dataset loading and molecular graph preprocessing
│   ├── models/          # GMACORE-TM model components
│   ├── losses/          # Contrastive and age-weighted losses
│   ├── training/        # Pretraining and downstream training
│   └── evaluation/      # Evaluation and embedding analysis
├── scripts/             # Scripts for running experiments
├── results/             # Reported experimental results
├── data/                # Dataset instructions only; raw data are not stored
├── requirements.txt
└── README.md
