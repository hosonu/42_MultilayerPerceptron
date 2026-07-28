"""Load the Wisconsin Diagnostic Breast Cancer CSV dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "data.csv"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"
EXPECTED_COLUMNS = 32

COLUMN_NAMES: list[str] = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

FEATURE_COLUMNS = COLUMN_NAMES[2:]
LABEL_COLUMN = "diagnosis"
VALID_LABELS = {"B", "M"}


def load_dataset(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read the CSV file and return a validated dataframe."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    _validate_dataset(df)
    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, got {df.shape[1]}",
        )

    invalid_labels = set(df[LABEL_COLUMN].unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(
            f"Unexpected diagnosis labels: {sorted(invalid_labels)}",
        )
