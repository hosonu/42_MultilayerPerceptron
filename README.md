# 42 Multilayer Perceptron

From-scratch multilayer perceptron for binary breast-cancer diagnosis (M/B).
The network is implemented with NumPy only — no ML frameworks for layers, loss, or training.

## Architecture

| Layer   | Units | Activation | Init |
|---------|-------|------------|------|
| Input   | 30    | —          | —    |
| Hidden1 | 24    | ReLU       | He   |
| Hidden2 | 24    | ReLU       | He   |
| Output  | 2     | Softmax    | He   |

- Loss: categorical cross-entropy
- Optimizer: mini-batch SGD

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (installed locally by the Makefile)

## Quick start

```bash
make install   # sync dependencies via uv
make split     # write data/train.csv and data/validation.csv
make train     # train MLP, save model.pkl, plot learning curves
make predict   # evaluate the saved model on the validation set
```

Optional exploration / architecture search:

```bash
make explore
make sweep     # grid-search lr / epochs / batch size / hidden layers / units
```

Edit the `CONFIG` block at the top of `scripts/sweep_architectures.py`, or pass lists via `ARGS`.

Pass extra CLI flags through `ARGS`:

```bash
make train ARGS="--epochs 500 --lr 0.01 --batch-size 32 --no-plot"
make predict ARGS="--model model.pkl --data data/validation.csv"
make split ARGS="--validation-ratio 0.2 --seed 42"
make sweep ARGS="--lrs 0.01 0.05 --epochs-list 500 --batch-sizes 16 32 --hidden-layers 1 2 --units 16 24"
```

## Project structure

```text
.
├── split.py                 # Create train / validation CSV splits
├── train.py                 # Train the MLP and save a checkpoint
├── predict.py               # Run inference with a saved model
├── main.py                  # Placeholder entry point
├── src/
│   ├── network.py           # MLP (forward, backward, fit)
│   ├── layers.py            # Dense layer + He init
│   ├── activations.py       # ReLU, Softmax
│   ├── losses.py            # Cross-entropy
│   ├── metrics.py           # Accuracy + learning-curve plot
│   ├── preprocessing.py     # Label encoding + feature scaling
│   ├── data_loader.py       # CSV loading helpers
│   └── dataset_split.py     # Stratified / seeded split utilities
├── scripts/
│   ├── explore_dataset.py       # Dataset EDA plots
│   └── sweep_architectures.py   # Architecture / hyperparameter grid search
├── data/
│   └── data.csv             # Raw breast-cancer dataset
├── results/                 # Sweep CSV output (gitignored)
├── tests/
├── Makefile
└── README.md
```

## Development

```bash
make lint      # flake8
make format    # autopep8
make test      # pytest
make check     # lint + test
make clean     # remove caches, model.pkl
make fclean    # also remove .venv and local uv installs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming and PR guidelines.
