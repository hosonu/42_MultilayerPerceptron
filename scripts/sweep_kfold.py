#!/usr/bin/env python3
"""K-Fold cross-validation sweep to find robust MLP hyperparameters.

Evaluates every config in the grid across K data splits so that the ranking
does not depend on a lucky/unlucky single train/val split.

The final train.py pipeline stays hold-out based; this script only identifies
hyperparameter candidates that generalise consistently.

Run:
    make sweep-kfold
    # or
    uv run python scripts/sweep_kfold.py
"""

from __future__ import annotations

import csv
import itertools
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_TRAIN_PATH, LABEL_COLUMN, load_dataset
from src.network import MLP
from src.preprocessing import Preprocessor

# =============================================================================
# CONFIG — edit the lists below to change the search space
# =============================================================================

K_FOLDS: int = 5          # number of cross-validation folds
SHUFFLE_SEED: int = 42    # seed for fold assignment (keep fixed for reproducibility)
MODEL_SEED_BASE: int = 0  # model init seed for fold k = MODEL_SEED_BASE + k
ROBUST_LAMBDA: float = 0.5  # penalty on std: robust_score = mean_acc - λ * std_acc
TOP_K: int = 15             # how many top configs to print
PATIENCE: int = 20        # early stopping patience (matches train.py default)

# Search space — targeted ranges based on dataset size (569 samples, 30 features)
LRS: list[float] = [0.005, 0.01, 0.03, 0.05]
EPOCHS_LIST: list[int] = [50, 100, 200, 500]
BATCH_SIZES: list[int] = [8, 16, 32]
HIDDEN_LAYERS: list[int] = [2, 3]
UNITS: list[int] = [16, 24, 32]
WEIGHT_DECAYS: list[float] = [0.0, 1e-4, 5e-4, 1e-3]

N_FEATURES: int = 30
N_CLASSES: int = 2

DATA_PATH: Path = DEFAULT_TRAIN_PATH
RESULTS_PATH: Path = PROJECT_ROOT / "results" / "sweep_kfold.csv"

# =============================================================================


@dataclass(frozen=True)
class TrialConfig:
    lr: float
    epochs: int
    batch_size: int
    n_hidden: int
    units: int
    weight_decay: float

    @property
    def arch_label(self) -> str:
        hidden = " -> ".join([str(self.units)] * self.n_hidden)
        return f"{N_FEATURES} -> {hidden} -> {N_CLASSES}"

    @property
    def label(self) -> str:
        return (
            f"lr{self.lr:g}_bs{self.batch_size}"
            f"_ep{self.epochs}_h{self.n_hidden}_u{self.units}_wd{self.weight_decay:g}"
        )


@dataclass
class FoldResult:
    fold: int
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    final_val_acc: float
    final_val_loss: float
    overfit_gap: float


@dataclass
class KFoldResult:
    config: TrialConfig
    fold_results: list[FoldResult] = field(default_factory=list)

    mean_best_val_acc: float = 0.0
    std_best_val_acc: float = 0.0
    mean_best_val_loss: float = 0.0
    std_best_val_loss: float = 0.0
    mean_overfit_gap: float = 0.0
    robust_score: float = 0.0

    def aggregate(self) -> None:
        accs = [r.best_val_acc for r in self.fold_results]
        losses = [r.best_val_loss for r in self.fold_results]
        gaps = [r.overfit_gap for r in self.fold_results]

        n = len(accs)
        self.mean_best_val_acc = statistics.fmean(accs)
        self.std_best_val_acc = statistics.stdev(accs) if n >= 2 else 0.0
        self.mean_best_val_loss = statistics.fmean(losses)
        self.std_best_val_loss = statistics.stdev(losses) if n >= 2 else 0.0
        self.mean_overfit_gap = statistics.fmean(gaps)
        self.robust_score = self.mean_best_val_acc - ROBUST_LAMBDA * self.std_best_val_acc


def build_arch(n_hidden: int, units: int) -> list[tuple[int, int, str]]:
    arch: list[tuple[int, int, str]] = []
    n_in = N_FEATURES
    for _ in range(n_hidden):
        arch.append((n_in, units, "relu"))
        n_in = units
    arch.append((n_in, N_CLASSES, "softmax"))
    return arch


def stratified_kfold(
    df: pd.DataFrame,
    k: int,
    shuffle_seed: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return k (train_df, val_df) pairs using stratified splitting.

    Each fold's val set preserves the original class ratio.
    """
    label_parts: dict[str, list[pd.DataFrame]] = {}
    for label, group in df.groupby(LABEL_COLUMN, sort=True):
        shuffled = (
            group.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        )
        n = len(shuffled)
        indices = [
            list(range(i, n, k)) for i in range(k)
        ]
        label_parts[str(label)] = [shuffled.iloc[idx].reset_index(drop=True) for idx in indices]

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(k):
        val_parts = [label_parts[lbl][i] for lbl in label_parts]
        train_parts = [
            label_parts[lbl][j]
            for lbl in label_parts
            for j in range(k)
            if j != i
        ]
        val_df = pd.concat(val_parts, ignore_index=True)
        train_df = pd.concat(train_parts, ignore_index=True)

        rng = np.random.default_rng(shuffle_seed + i + 1)
        train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)
        val_df = val_df.iloc[rng.permutation(len(val_df))].reset_index(drop=True)

        folds.append((train_df, val_df))

    return folds


def run_fold(
    cfg: TrialConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_idx: int,
) -> FoldResult:
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_val, y_val = prep.transform(val_df)

    model = MLP(
        seed=MODEL_SEED_BASE + fold_idx,
        arch=build_arch(cfg.n_hidden, cfg.units),
    )
    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        patience=PATIENCE,
        verbose=False,
    )

    val_accs = history["val_acc"]
    val_losses = history["val_loss"]
    best_idx = max(range(len(val_accs)), key=lambda i: (val_accs[i], -val_losses[i]))

    return FoldResult(
        fold=fold_idx,
        best_val_acc=val_accs[best_idx],
        best_val_loss=val_losses[best_idx],
        best_epoch=best_idx + 1,
        final_val_acc=val_accs[-1],
        final_val_loss=val_losses[-1],
        overfit_gap=val_losses[-1] - val_losses[best_idx],
    )


def run_config(
    cfg: TrialConfig,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> KFoldResult:
    result = KFoldResult(config=cfg)
    for fold_idx, (train_df, val_df) in enumerate(folds):
        result.fold_results.append(run_fold(cfg, train_df, val_df, fold_idx))
    result.aggregate()
    return result


def iter_configs() -> list[TrialConfig]:
    return [
        TrialConfig(lr=lr, epochs=epochs, batch_size=bs, n_hidden=nh, units=u, weight_decay=wd)
        for lr, epochs, bs, nh, u, wd in itertools.product(
            LRS, EPOCHS_LIST, BATCH_SIZES, HIDDEN_LAYERS, UNITS, WEIGHT_DECAYS
        )
    ]


def write_csv(path: Path, results: list[KFoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "robust_score",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_best_val_loss",
        "std_best_val_loss",
        "mean_overfit_gap",
        "lr",
        "epochs",
        "batch_size",
        "weight_decay",
        "n_hidden",
        "units",
        "arch",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(results, start=1):
            cfg = r.config
            writer.writerow(
                {
                    "rank": rank,
                    "robust_score": f"{r.robust_score:.6f}",
                    "mean_best_val_acc": f"{r.mean_best_val_acc:.6f}",
                    "std_best_val_acc": f"{r.std_best_val_acc:.6f}",
                    "mean_best_val_loss": f"{r.mean_best_val_loss:.6f}",
                    "std_best_val_loss": f"{r.std_best_val_loss:.6f}",
                    "mean_overfit_gap": f"{r.mean_overfit_gap:.6f}",
                    "lr": cfg.lr,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "weight_decay": cfg.weight_decay,
                    "n_hidden": cfg.n_hidden,
                    "units": cfg.units,
                    "arch": cfg.arch_label,
                }
            )


def print_table(results: list[KFoldResult], top_k: int) -> None:
    rows = results[:top_k]
    print(
        f"\nTop {len(rows)} configs  "
        f"(robust_score = mean_acc - {ROBUST_LAMBDA} * std_acc, "
        f"{K_FOLDS}-fold CV):"
    )
    header = (
        f"{'#':>3}  {'robust':>8}  {'mean_acc':>9}  {'std_acc':>8}  "
        f"{'mean_loss':>10}  {'gap':>8}  "
        f"{'lr':>7}  {'ep':>5}  {'bs':>4}  {'wd':>7}  {'h':>2}  {'u':>4}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, start=1):
        cfg = r.config
        print(
            f"{i:>3}  {r.robust_score:>8.5f}  {r.mean_best_val_acc:>9.5f}  "
            f"{r.std_best_val_acc:>8.5f}  {r.mean_best_val_loss:>10.5f}  "
            f"{r.mean_overfit_gap:>8.5f}  "
            f"{cfg.lr:>7g}  {cfg.epochs:>5}  {cfg.batch_size:>4}  "
            f"{cfg.weight_decay:>7g}  {cfg.n_hidden:>2}  {cfg.units:>4}"
        )


def main() -> None:
    configs = iter_configs()
    total = len(configs)
    total_runs = total * K_FOLDS

    print(
        f"K-Fold sweep  k={K_FOLDS}  configs={total}  total_runs={total_runs}\n"
        f"LRS={LRS}\n"
        f"EPOCHS={EPOCHS_LIST}  BATCH_SIZES={BATCH_SIZES}\n"
        f"HIDDEN_LAYERS={HIDDEN_LAYERS}  UNITS={UNITS}\n"
        f"WEIGHT_DECAYS={WEIGHT_DECAYS}\n"
        f"robust_score = mean_acc - {ROBUST_LAMBDA} * std_acc\n"
        f"Data: {DATA_PATH}"
    )

    df = load_dataset(DATA_PATH)
    folds = stratified_kfold(df, K_FOLDS, SHUFFLE_SEED)
    print(
        f"Fold sizes (val): "
        + "  ".join(f"fold{i}={len(v)}" for i, (_, v) in enumerate(folds))
    )
    print()

    kfold_results: list[KFoldResult] = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i:>3}/{total}] {cfg.label}",
            end="  ",
            flush=True,
        )
        result = run_config(cfg, folds)
        kfold_results.append(result)
        print(
            f"mean_acc={result.mean_best_val_acc:.4f} ±{result.std_best_val_acc:.4f}"
            f"  loss={result.mean_best_val_loss:.4f} ±{result.std_best_val_loss:.4f}"
            f"  robust={result.robust_score:.4f}"
        )

    kfold_results.sort(
        key=lambda r: (-r.robust_score, r.mean_best_val_loss)
    )

    write_csv(RESULTS_PATH, kfold_results)
    print_table(kfold_results, TOP_K)

    best = kfold_results[0]
    print(
        f"\nResults saved to {RESULTS_PATH}"
        f"\n\nBest config (robust_score={best.robust_score:.5f}):"
        f"\n  lr={best.config.lr:g}  epochs={best.config.epochs}"
        f"  batch_size={best.config.batch_size}"
        f"  n_hidden={best.config.n_hidden}  units={best.config.units}"
        f"\n  arch: {best.config.arch_label}"
        f"\n\nRetrain tip:"
        f"\n  make train ARGS=\"--lr {best.config.lr:g}"
        f" --epochs {best.config.epochs}"
        f" --batch-size {best.config.batch_size}"
        f" --seed 42\""
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSweep interrupted.")
        raise SystemExit(130)
