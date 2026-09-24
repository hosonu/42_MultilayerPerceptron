"""Train the MLP on the breast-cancer dataset and save the model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.data_loader import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    load_dataset,
)
from src.metrics import plot_learning_curves
from src.network import MLP
from src.preprocessing import Preprocessor

DEFAULT_MODEL_PATH = Path(__file__).parent / "model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLP classifier.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Training CSV from split.py (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help=(
            "Validation CSV from split.py "
            f"(default: {DEFAULT_VALIDATION_PATH})"
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model pickle (default: model.pkl)",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Maximum number of training epochs (early stopping may stop sooner)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="L2 weight decay coefficient λ (default: 0.001, 0 to disable)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience: stop after this many epochs without val_loss improvement",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=16,
        help="RNG seed for weight init and mini-batch shuffling",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip learning curve plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ data
    for path in (args.train_data, args.val_data):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run `make split` first to create train/validation CSVs."
            )

    print(f"Loading train from {args.train_data} …")
    train_df = load_dataset(args.train_data)
    print(f"Loading validation from {args.val_data} …")
    val_df = load_dataset(args.val_data)
    print(
        f"Loaded → train: {len(train_df)} samples | "
        f"validation: {len(val_df)} samples"
    )

    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_val, y_val = prep.transform(val_df)

    # ----------------------------------------------------------------- train
    print(
        f"\nTraining MLP  lr={args.lr}  epochs={args.epochs}"
        f"  batch_size={args.batch_size}  weight_decay={args.weight_decay}"
        f"  seed={args.seed}"
    )
    model = MLP(seed=args.seed)
    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        verbose=True,
        log_every=10,
        patience=args.patience,
    )

    # ------------------------------------------------------------------ save
    checkpoint = {"model": model, "preprocessor": prep}
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nModel saved to {args.model_out}")

    # ------------------------------------------------------------------ plot
    if not args.no_plot:
        plot_learning_curves(history)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
