#!/usr/bin/env python3
"""Grid-search MLP architectures / hyperparameters on the breast-cancer dataset.

Edit the CONFIG block below, then run:

    make sweep
    # or
    uv run python scripts/sweep_architectures.py

CLI flags override CONFIG when provided.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_TRAIN_PATH, load_dataset
from src.dataset_split import split_dataset
from src.network import MLP
from src.preprocessing import Preprocessor

# =============================================================================
# CONFIG — edit these lists to change the search space
# =============================================================================

LRS: list[float] = [0.001, 0.01, 0.05]
EPOCHS_LIST: list[int] = [500, 1000]
BATCH_SIZES: list[int] = [16, 32, 64]
HIDDEN_LAYERS: list[int] = [1, 2]  # number of ReLU hidden layers
UNITS: list[int] = [8, 16, 24]  # width of each hidden layer

VAL_RATIO: float = 0.2
SEED: int = 42
N_FEATURES: int = 30
N_CLASSES: int = 2

DATA_PATH: Path = DEFAULT_TRAIN_PATH
RESULTS_PATH: Path = PROJECT_ROOT / "results" / "sweep.csv"
TOP_K: int = 10  # how many best rows to print

# =============================================================================


@dataclass(frozen=True)
class TrialConfig:
    lr: float
    epochs: int
    batch_size: int
    n_hidden: int
    units: int

    @property
    def arch_label(self) -> str:
        hidden = " → ".join([str(self.units)] * self.n_hidden)
        return f"{N_FEATURES} → {hidden} → {N_CLASSES}"


@dataclass
class TrialResult:
    config: TrialConfig
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    overfit_gap: float  # final_val_loss - best_val_loss
    val_loss_curve: list[float]
    train_loss_curve: list[float]
    val_acc_curve: list[float]
    train_acc_curve: list[float]


def build_arch(n_hidden: int, units: int) -> list[tuple[int, int, str]]:
    """Build (n_in, n_out, activation) tuples for MLP."""
    if n_hidden < 1:
        raise ValueError("n_hidden must be >= 1")
    if units < 1:
        raise ValueError("units must be >= 1")

    arch: list[tuple[int, int, str]] = []
    n_in = N_FEATURES
    for _ in range(n_hidden):
        arch.append((n_in, units, "relu"))
        n_in = units
    arch.append((n_in, N_CLASSES, "softmax"))
    return arch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep MLP architectures and hyperparameters."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help=f"Dataset CSV (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_PATH,
        help=f"CSV results path (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=None,
        help="Learning rates (overrides CONFIG)",
    )
    parser.add_argument(
        "--epochs-list",
        type=int,
        nargs="+",
        default=None,
        help="Epoch counts (overrides CONFIG)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Mini-batch sizes (overrides CONFIG)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=None,
        help="Hidden-layer counts (overrides CONFIG)",
    )
    parser.add_argument(
        "--units",
        type=int,
        nargs="+",
        default=None,
        help="Hidden units per layer (overrides CONFIG)",
    )
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on number of combinations to run",
    )
    return parser.parse_args()


def iter_configs(
    lrs: list[float],
    epochs_list: list[int],
    batch_sizes: list[int],
    hidden_layers: list[int],
    units: list[int],
    max_trials: int | None,
) -> list[TrialConfig]:
    configs = [
        TrialConfig(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            n_hidden=n_hidden,
            units=u,
        )
        for lr, epochs, batch_size, n_hidden, u in itertools.product(
            lrs, epochs_list, batch_sizes, hidden_layers, units
        )
    ]
    if max_trials is not None:
        configs = configs[:max_trials]
    return configs


def run_trial(
    cfg: TrialConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int,
) -> TrialResult:
    model = MLP(seed=seed, arch=build_arch(cfg.n_hidden, cfg.units))
    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=False,
    )

    val_accs = history["val_acc"]
    val_losses = history["val_loss"]
    train_losses = history["train_loss"]
    train_accs = history["train_acc"]
    best_idx = max(range(len(val_accs)), key=lambda i: (val_accs[i], -val_losses[i]))

    final_val_loss = val_losses[-1]
    best_val_loss = val_losses[best_idx]

    return TrialResult(
        config=cfg,
        final_train_loss=train_losses[-1],
        final_train_acc=train_accs[-1],
        final_val_loss=final_val_loss,
        final_val_acc=val_accs[-1],
        best_val_acc=val_accs[best_idx],
        best_val_loss=best_val_loss,
        best_epoch=best_idx + 1,
        overfit_gap=final_val_loss - best_val_loss,
        val_loss_curve=val_losses,
        train_loss_curve=train_losses,
        val_acc_curve=val_accs,
        train_acc_curve=train_accs,
    )


def write_csv(path: Path, results: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "lr",
        "epochs",
        "batch_size",
        "n_hidden",
        "units",
        "arch",
        "best_epoch",
        "overfit_gap",
        "best_val_acc",
        "best_val_loss",
        "final_val_acc",
        "final_val_loss",
        "final_train_acc",
        "final_train_loss",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            cfg = result.config
            writer.writerow(
                {
                    "rank": rank,
                    "lr": cfg.lr,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "n_hidden": cfg.n_hidden,
                    "units": cfg.units,
                    "arch": cfg.arch_label,
                    "best_epoch": result.best_epoch,
                    "overfit_gap": f"{result.overfit_gap:.6f}",
                    "best_val_acc": f"{result.best_val_acc:.6f}",
                    "best_val_loss": f"{result.best_val_loss:.6f}",
                    "final_val_acc": f"{result.final_val_acc:.6f}",
                    "final_val_loss": f"{result.final_val_loss:.6f}",
                    "final_train_acc": f"{result.final_train_acc:.6f}",
                    "final_train_loss": f"{result.final_train_loss:.6f}",
                }
            )


def write_curves_csv(path: Path, results: list[TrialResult]) -> None:
    """Save per-epoch val/train loss and accuracy for every trial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "rank",
        "lr",
        "epochs",
        "batch_size",
        "n_hidden",
        "units",
        "epoch",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            cfg = result.config
            trial_id = (
                f"lr{cfg.lr:g}_bs{cfg.batch_size}"
                f"_ep{cfg.epochs}_h{cfg.n_hidden}_u{cfg.units}"
            )
            for ep, (tl, vl, ta, va) in enumerate(
                zip(
                    result.train_loss_curve,
                    result.val_loss_curve,
                    result.train_acc_curve,
                    result.val_acc_curve,
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        "trial_id": trial_id,
                        "rank": rank,
                        "lr": cfg.lr,
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "n_hidden": cfg.n_hidden,
                        "units": cfg.units,
                        "epoch": ep,
                        "train_loss": f"{tl:.6f}",
                        "val_loss": f"{vl:.6f}",
                        "train_acc": f"{ta:.6f}",
                        "val_acc": f"{va:.6f}",
                    }
                )


def print_table(results: list[TrialResult], top_k: int) -> None:
    rows = results[:top_k]
    print("\nTop results (sorted by best_val_acc, then best_val_loss):")
    header = (
        f"{'#':>3}  {'lr':>7}  {'epochs':>6}  {'bs':>4}  {'hid':>3}  {'units':>5}  "
        f"{'best_val_acc':>12}  {'best_val_loss':>13}  arch"
    )
    print(header)
    print("-" * len(header))
    for i, result in enumerate(rows, start=1):
        cfg = result.config
        print(
            f"{i:>3}  {cfg.lr:>7g}  {cfg.epochs:>6}  {cfg.batch_size:>4}  "
            f"{cfg.n_hidden:>3}  {cfg.units:>5}  {result.best_val_acc:>12.4f}  "
            f"{result.best_val_loss:>13.4f}  {cfg.arch_label}"
        )


def main() -> None:
    args = parse_args()

    lrs = args.lrs if args.lrs is not None else LRS
    epochs_list = args.epochs_list if args.epochs_list is not None else EPOCHS_LIST
    batch_sizes = (
        args.batch_sizes if args.batch_sizes is not None else BATCH_SIZES
    )
    hidden_layers = (
        args.hidden_layers if args.hidden_layers is not None else HIDDEN_LAYERS
    )
    units = args.units if args.units is not None else UNITS

    configs = iter_configs(
        lrs, epochs_list, batch_sizes, hidden_layers, units, args.max_trials
    )
    total = len(configs)
    print(
        f"Sweep: {total} trials | "
        f"lrs={lrs} epochs={epochs_list} batch_sizes={batch_sizes} "
        f"hidden_layers={hidden_layers} units={units}"
    )
    print(f"Data: {args.data}  val_ratio={args.val_ratio}  seed={args.seed}")

    df = load_dataset(args.data)
    train_df, val_df = split_dataset(df, args.val_ratio, seed=args.seed)
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_val, y_val = prep.transform(val_df)
    print(f"Split → train: {len(train_df)} | validation: {len(val_df)}\n")

    results: list[TrialResult] = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i}/{total}] lr={cfg.lr:g} epochs={cfg.epochs} "
            f"batch_size={cfg.batch_size} hidden={cfg.n_hidden} "
            f"units={cfg.units}  ({cfg.arch_label})",
            flush=True,
        )
        result = run_trial(cfg, X_train, y_train, X_val, y_val, args.seed)
        results.append(result)
        print(
            f"         → best_val_acc={result.best_val_acc:.4f}  "
            f"best_val_loss={result.best_val_loss:.4f}  "
            f"final_val_acc={result.final_val_acc:.4f}"
        )

    results.sort(
        key=lambda r: (-r.best_val_acc, r.best_val_loss, -r.final_val_acc)
    )
    write_csv(args.out, results)
    curves_path = args.out.with_name(args.out.stem + "_curves.csv")
    write_curves_csv(curves_path, results)
    print_table(results, args.top_k)

    best = results[0]
    print(f"\nResults saved to {args.out}")
    print(f"Epoch curves  saved to {curves_path}")
    print(
        "Best config → "
        f"lr={best.config.lr:g}  epochs={best.config.epochs}  "
        f"batch_size={best.config.batch_size}  "
        f"hidden={best.config.n_hidden}  units={best.config.units}  "
        f"arch={best.config.arch_label}"
    )
    print(
        "Retrain tip: update _DEFAULT_ARCH in src/network.py to match the "
        "winning arch, then:\n"
        f"  make train ARGS=\"--lr {best.config.lr:g} "
        f"--epochs {best.config.epochs} "
        f"--batch-size {best.config.batch_size} --seed {args.seed}\""
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSweep interrupted.")
        raise SystemExit(130)
