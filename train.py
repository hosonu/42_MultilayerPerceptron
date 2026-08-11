"""Train the MLP on the breast-cancer dataset and save the model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.data_loader import DEFAULT_TRAIN_PATH, load_dataset
from src.dataset_split import split_dataset
from src.metrics import plot_learning_curves
from src.network import MLP
from src.preprocessing import Preprocessor

DEFAULT_MODEL_PATH = Path(__file__).parent / "model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLP classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the dataset CSV (default: data/data.csv)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model pickle (default: model.pkl)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true", help="Skip learning curve plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ data
    print(f"Loading dataset from {args.data} …")
    df = load_dataset(args.data)

    train_df, val_df = split_dataset(df, args.val_ratio, seed=args.seed)
    print(
        f"Split → train: {len(train_df)} samples | "
        f"validation: {len(val_df)} samples"
    )

    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_val, y_val = prep.transform(val_df)

    # ----------------------------------------------------------------- train
    print(
        f"\nTraining MLP  lr={args.lr}  epochs={args.epochs}"
        f"  batch_size={args.batch_size}  seed={args.seed}"
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
        verbose=True,
        log_every=100,
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
        print("\nInterrupted before training started.")
        raise SystemExit(130)
