# Contributing

## Development setup

```bash
conda env create -f environment.yml
conda activate gmacore-tm
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check src tests scripts
black --check src tests scripts
pytest
```

## Conventions

- Line length is 100 characters, enforced by `black` and `ruff`.
- Public functions and classes carry docstrings. Comments explain why a choice
  was made, not what a line does.
- New model behaviour is accompanied by a test. Changes to a loss function in
  particular should be tested against a closed-form evaluation where one
  exists, as in `tests/test_losses.py`.
- Hyperparameters belong in `configs/`, not in code. Anything hard-coded in a
  module should be a genuine architectural constant.
- Changes that alter numerical behaviour are recorded in `docs/CHANGES.md`.

## Reporting reproduction problems

Open an issue including the command, the resolved `config.json` from the run
directory, the output of `python -c "import torch; print(torch.__version__)"`
and the relevant excerpt from the run log.
