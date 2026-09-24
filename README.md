# 42 Multilayer Perceptron

From-scratch multilayer perceptron for binary breast-cancer diagnosis (M/B).
The network is implemented with NumPy only — no ML frameworks for layers, loss, or training.

## Architecture

| Layer   | Units | Activation | Init |
|---------|-------|------------|------|
| Input   | 30    | —          | —    |
| Hidden1 | 16    | ReLU       | He   |
| Hidden2 | 16    | ReLU       | He   |
| Output  | 2     | Softmax    | He   |

Hidden-layer sizes are the default. Override them with `--layer` (one integer per hidden layer, ReLU). The output layer stays 2 units with softmax.

```bash
python train.py --layer 24 24 24
```

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

Optional exploration / hyperparameter search:

```bash
make explore
make sweep          # grid-search lr / epochs / batch size / hidden layers / units
make sweep-seeds    # fixed hyperparams, vary seeds only
make sweep-kfold    # stratified K-Fold CV for robust ranking
```

Edit the `CONFIG` block at the top of each sweep script, or pass lists via `ARGS`.
Architecture sweep also writes per-epoch curves to `results/*_curves.csv`.

`make train` loads `data/train.csv` and `data/validation.csv` from `make split`
(no re-split). `--seed` only controls weight init and mini-batch shuffling.
Defaults: `--layer 16 16 --epochs 500 --batch-size 16 --seed 16 --weight-decay 0.001`.

Pass extra CLI flags through `ARGS`:

```bash
make train ARGS="--layer 24 24 24 --epochs 250 --lr 0.05 --batch-size 16 --patience 10 --no-plot"
make predict ARGS="--model model.pkl --data data/validation.csv"
make split ARGS="--validation-ratio 0.2 --seed 42"
make sweep ARGS="--lrs 0.01 0.05 --epochs-list 200 --batch-sizes 16 32 --hidden-layers 1 2 --units 16 24"
make sweep-seeds ARGS="--seeds 42 0 1 7 123"
make sweep-kfold
```

## Project structure

```text
.
├── split.py                 # Create train / validation CSV splits
├── train.py                 # Train the MLP and save a checkpoint
├── predict.py               # Run inference with a saved model
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
│   ├── sweep_architectures.py   # Architecture / hyperparameter grid search
│   ├── sweep_seeds.py           # Fixed-config multi-seed sweep
│   └── sweep_kfold.py           # Stratified K-Fold CV sweep
├── data/
│   └── data.csv             # Raw breast-cancer dataset
├── results/                 # Sweep CSV output (gitignored)
├── Makefile
└── README.md
```

## Development

```bash
make lint      # flake8
make format    # autopep8
make check     # flake8
make clean     # remove caches, model.pkl
make fclean    # also remove .venv and local uv installs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming and PR guidelines.
