# Data

This directory is not tracked by git. Populate it as follows.

## Pretraining corpus

Place a newline-delimited SMILES file at:

    data/pubchem/pubchem_10m_clean.txt

The reported results use the ChemBERTa-compiled PubChem collection, available
from https://pubchem.ncbi.nlm.nih.gov. Any newline-delimited SMILES file works;
point `data.smiles_file` in the pretraining config at your copy.

## Downstream benchmarks

    python scripts/download_moleculenet.py --all

writes the benchmark CSV files to `data/moleculenet/`. Files that cannot be
retrieved automatically are available from https://moleculenet.org/datasets.

## Cache

`data/cache/` holds featurized graphs so that repeated runs skip RDKit
preprocessing. It is safe to delete; it will be regenerated.
