#!/usr/bin/env python3
"""Create train and validation CSV splits from the raw dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import DEFAULT_CSV_PATH, load_dataset
from src.dataset_split import save_split, split_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the dataset into train and validation CSV files.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Input CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Train CSV output path (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help=(
            "Validation CSV output path "
            f"(default: {DEFAULT_VALIDATION_PATH})"
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.csv_path)
    train_df, validation_df = split_dataset(
        df,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    save_split(train_df, args.train_output)
    save_split(validation_df, args.validation_output)

    print(f"Input rows: {len(df)}")
    print(f"Train rows: {len(train_df)} -> {args.train_output}")
    print(f"Validation rows: {len(validation_df)} -> {args.validation_output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
