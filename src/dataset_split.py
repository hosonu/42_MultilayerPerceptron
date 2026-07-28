"""Split the dataset into train and validation CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import COLUMN_NAMES, LABEL_COLUMN


def split_dataset(
    df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe with stratified sampling by diagnosis label."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(LABEL_COLUMN, sort=True):
        shuffled = group.sample(frac=1, random_state=seed)
        shuffled = shuffled.reset_index(drop=True)
        validation_size = int(round(len(shuffled) * validation_ratio))
        validation_size = min(max(validation_size, 1), len(shuffled) - 1)

        validation_parts.append(shuffled.iloc[:validation_size])
        train_parts.append(shuffled.iloc[validation_size:])

    train_df = pd.concat(train_parts, ignore_index=True)
    validation_df = pd.concat(validation_parts, ignore_index=True)

    rng = np.random.default_rng(seed)
    train_df = train_df.iloc[rng.permutation(len(train_df))]
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.iloc[rng.permutation(len(validation_df))]
    validation_df = validation_df.reset_index(drop=True)

    return train_df, validation_df


def save_split(df: pd.DataFrame, output_path: Path | str) -> None:
    """Write a split CSV without header, preserving the original format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, header=False, index=False, columns=COLUMN_NAMES)
