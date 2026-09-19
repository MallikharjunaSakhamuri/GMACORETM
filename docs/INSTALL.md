# Installation

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- PyTorch Geometric 2.4 or later
- RDKit 2023.03 or later
- A CUDA-capable GPU is recommended for pretraining but is not required

## Conda (recommended)

```bash
conda env create -f environment.yml
conda activate gmacore-tm
pip install -e .
```

The environment file installs the CPU build of PyTorch, which is sufficient for
the tests and for small-scale runs.

## GPU installation

PyTorch Geometric distributes its compiled extensions per torch and CUDA
version. Install PyTorch first, then the matching extension wheels:

```bash
# Example for PyTorch 2.1 with CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install rdkit
pip install -e .
```

Substitute the torch and CUDA versions in the `-f` URL to match your
installation. A mismatch between the wheel URL and the installed torch version
is the most common cause of import errors in `torch_geometric`.

## Verifying the installation

```bash
pytest
```

The data tests are skipped automatically if RDKit is unavailable, and the model
tests are skipped if PyTorch Geometric is unavailable, so a partial pass
indicates which component is missing.

A short end-to-end check on a small corpus:

```bash
python scripts/pretrain.py --config configs/pretrain_gmacore_tm.yaml \
    --set pretrain.epochs=1 data.limit=200 pretrain.device=cpu \
          pretrain.output_dir=runs/smoke_test
```

## Troubleshooting

**`ImportError` from `torch_geometric`.** The compiled extensions do not match
the installed torch version. Reinstall them using the `-f` URL for your exact
torch and CUDA combination.

**`RDKit` not found.** Install it from conda-forge
(`conda install -c conda-forge rdkit`); the PyPI wheel is not available for
every platform.

**Out-of-memory during pretraining.** Reduce `pretrain.batch_size`, or reduce
`framework.queue_size`. The queue holds `queue_size x output_dim` floats, so
the default of 50,000 by 128 occupies approximately 25 MB and is rarely the
constraint; the batch size is the usual cause.

**Featurization is slow.** The first run featurizes the whole corpus with
RDKit. Set `data.cache_path` so that subsequent runs load the cached graphs
instead. Keep `data.generate_conformer` set to `false` unless three-dimensional
features are genuinely required, since conformer embedding dominates
preprocessing time.
