"""Run inference with a trained MLP model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from src.data_loader import DEFAULT_VALIDATION_PATH, load_dataset
from src.metrics import accuracy
from src.preprocessing import INT_TO_LABEL

DEFAULT_MODEL_PATH = Path(__file__).parent / "model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a trained MLP.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model pickle produced by train.py (default: model.pkl)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Path to the dataset CSV (default: data/data.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ load
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Run train.py first to create a checkpoint."
        )

    with open(args.model, "rb") as f:
        checkpoint: dict = pickle.load(f)

    model = checkpoint["model"]
    prep = checkpoint["preprocessor"]

    # ------------------------------------------------------------------ data
    print(f"Loading dataset from {args.data} …")
    df = load_dataset(args.data)
    X, y_true = prep.transform(df)

    # --------------------------------------------------------------- predict
    y_pred = model.predict(X)
    acc = accuracy(y_true, y_pred)

    labels_pred = np.vectorize(INT_TO_LABEL.get)(y_pred)
    labels_true = np.vectorize(INT_TO_LABEL.get)(y_true)

    # ---------------------------------------------------------------- report
    n_correct = int((y_pred == y_true).sum())
    n_total = len(y_true)

    print(f"\nPredictions on {n_total} samples:")
    print(f"  Correct   : {n_correct}")
    print(f"  Incorrect : {n_total - n_correct}")
    print(f"  Accuracy  : {acc:.4f} ({acc * 100:.2f} %)")

    # Per-class breakdown
    for label_int, label_str in sorted(INT_TO_LABEL.items()):
        mask = y_true == label_int
        class_acc = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.0
        print(f"  {label_str} ({mask.sum()} samples) : {class_acc:.4f}")


if __name__ == "__main__":
    main()
