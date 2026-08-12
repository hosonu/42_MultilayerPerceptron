#!/usr/bin/env python3
"""Run a fixed hyperparameter config across multiple seeds.

Edit the CONFIG block below to set lr / epochs / batch_size / architecture.
Pass seeds on the CLI only:

    make sweep-seeds
    make sweep-seeds ARGS="--seeds 42 0 1 7 123"
    # or
    uv run python scripts/sweep_seeds.py --seeds 42 0 1
"""

from __future__ import annotations

import argparse
import csv
import statistics
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
# CONFIG — edit these to fix the hyperparameters under test
# =============================================================================

LR: float = 0.03
EPOCHS: int = 200
BATCH_SIZE: int = 16
N_HIDDEN: int = 2  # number of ReLU hidden layers
UNITS: int = 24  # width of each hidden layer

VAL_RATIO: float = 0.2
DEFAULT_SEEDS: list[int] = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
N_FEATURES: int = 30
N_CLASSES: int = 2

DATA_PATH: Path = DEFAULT_TRAIN_PATH
RESULTS_PATH: Path = PROJECT_ROOT / "results" / "seed_sweep.csv"

# =============================================================================


@dataclass(frozen=True)
class FixedConfig:
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
class SeedResult:
    seed: int
    config: FixedConfig
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    overfit_gap: float
    val_loss_curve: list[float]
    train_loss_curve: list[float]
    val_acc_curve: list[float]
    train_acc_curve: list[float]


METRIC_FIELDS: tuple[str, ...] = (
    "best_epoch",
    "overfit_gap",
    "best_val_acc",
    "best_val_loss",
    "final_val_acc",
    "final_val_loss",
    "final_train_acc",
    "final_train_loss",
)


def build_arch(n_hidden: int, units: int) -> list[tuple[int, int, str]]:
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
        description="Sweep seeds for a fixed MLP hyperparameter config."
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
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run (overrides CONFIG DEFAULT_SEEDS)",
    )
    return parser.parse_args()


def run_seed(
    cfg: FixedConfig,
    df,
    seed: int,
) -> SeedResult:
    train_df, val_df = split_dataset(df, VAL_RATIO, seed=seed)
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_val, y_val = prep.transform(val_df)

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

    return SeedResult(
        seed=seed,
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


def _metric_value(result: SeedResult, field: str) -> float:
    return float(getattr(result, field))


def summarize(results: list[SeedResult]) -> dict[str, dict[str, float]]:
    """Return {field: {"mean": ..., "std": ...}} for each metric."""
    summary: dict[str, dict[str, float]] = {}
    for field in METRIC_FIELDS:
        values = [_metric_value(r, field) for r in results]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        summary[field] = {"mean": mean, "std": std}
    return summary


def write_csv(
    path: Path,
    results: list[SeedResult],
    summary: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "seed",
        "lr",
        "epochs",
        "batch_size",
        "n_hidden",
        "units",
        "arch",
        *METRIC_FIELDS,
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            cfg = result.config
            writer.writerow(
                {
                    "rank": rank,
                    "seed": result.seed,
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

        cfg = results[0].config
        mean_row = {
            "rank": "mean",
            "seed": "mean",
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "n_hidden": cfg.n_hidden,
            "units": cfg.units,
            "arch": cfg.arch_label,
        }
        std_row = {
            "rank": "std",
            "seed": "std",
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "n_hidden": cfg.n_hidden,
            "units": cfg.units,
            "arch": cfg.arch_label,
        }
        for field in METRIC_FIELDS:
            mean_row[field] = f"{summary[field]['mean']:.6f}"
            std_row[field] = f"{summary[field]['std']:.6f}"
        writer.writerow(mean_row)
        writer.writerow(std_row)


def write_curves_csv(path: Path, results: list[SeedResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "rank",
        "seed",
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
                f"seed{result.seed}_lr{cfg.lr:g}_bs{cfg.batch_size}"
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
                        "seed": result.seed,
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


def print_table(results: list[SeedResult]) -> None:
    print("\nResults by seed (sorted by best_val_acc, then best_val_loss):")
    header = (
        f"{'#':>3}  {'seed':>6}  {'best_ep':>7}  {'overfit':>8}  "
        f"{'best_acc':>9}  {'best_loss':>10}  {'final_acc':>9}  {'final_loss':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, result in enumerate(results, start=1):
        print(
            f"{i:>3}  {result.seed:>6}  {result.best_epoch:>7}  "
            f"{result.overfit_gap:>8.4f}  "
            f"{result.best_val_acc:>9.4f}  {result.best_val_loss:>10.4f}  "
            f"{result.final_val_acc:>9.4f}  {result.final_val_loss:>10.4f}"
        )


def print_summary(summary: dict[str, dict[str, float]], n_seeds: int) -> None:
    print(f"\nAverage over {n_seeds} seeds (mean ± std):")
    for field in METRIC_FIELDS:
        mean = summary[field]["mean"]
        std = summary[field]["std"]
        print(f"  {field:>16} = {mean:.6f} ± {std:.6f}")


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else list(DEFAULT_SEEDS)
    if not seeds:
        raise SystemExit("At least one seed is required.")

    cfg = FixedConfig(
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        n_hidden=N_HIDDEN,
        units=UNITS,
    )

    print(
        f"Seed sweep: {len(seeds)} seeds | "
        f"lr={cfg.lr:g} epochs={cfg.epochs} batch_size={cfg.batch_size} "
        f"hidden={cfg.n_hidden} units={cfg.units} ({cfg.arch_label})"
    )
    print(f"Data: {args.data}  val_ratio={VAL_RATIO}  seeds={seeds}")

    df = load_dataset(args.data)
    print()

    results: list[SeedResult] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"[{i}/{len(seeds)}] seed={seed}", flush=True)
        result = run_seed(cfg, df, seed)
        results.append(result)
        print(
            f"         → best_val_acc={result.best_val_acc:.4f}  "
            f"best_val_loss={result.best_val_loss:.4f}  "
            f"final_val_acc={result.final_val_acc:.4f}  "
            f"overfit_gap={result.overfit_gap:.4f}"
        )

    results.sort(
        key=lambda r: (-r.best_val_acc, r.best_val_loss, -r.final_val_acc)
    )
    summary = summarize(results)

    write_csv(args.out, results, summary)
    curves_path = args.out.with_name(args.out.stem + "_curves.csv")
    write_curves_csv(curves_path, results)

    print_table(results)
    print_summary(summary, len(seeds))

    print(f"\nResults saved to {args.out}")
    print(f"Epoch curves  saved to {curves_path}")
    print(
        "Retrain tip:\n"
        f"  make train ARGS=\"--lr {cfg.lr:g} --epochs {cfg.epochs} "
        f"--batch-size {cfg.batch_size} --seed <chosen_seed>\""
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSeed sweep interrupted.")
        raise SystemExit(130)
