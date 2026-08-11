#!/usr/bin/env python3
"""Explore the breast cancer dataset structure and statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "data.csv"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore dataset structure and visualize key statistics.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the CSV file (default: {DEFAULT_CSV_PATH})",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, got {df.shape[1]}",
        )

    invalid_labels = set(df["diagnosis"].unique()) - {"M", "B"}
    if invalid_labels:
        raise ValueError(
            f"Unexpected diagnosis labels: {sorted(invalid_labels)}",
        )

    return df


def print_structure_summary(df: pd.DataFrame) -> None:
    print("=== Dataset structure ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]} (expected {EXPECTED_COLUMNS})")
    print(f"Column names: {', '.join(df.columns)}")
    print()


def print_diagnosis_distribution(df: pd.DataFrame) -> None:
    counts = df["diagnosis"].value_counts().sort_index()
    ratios = df["diagnosis"].value_counts(normalize=True).sort_index()

    print("=== Diagnosis distribution ===")
    for label in counts.index:
        print(f"{label}: {counts[label]} ({ratios[label]:.1%})")
    print()


def print_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    total_missing = int(missing.sum())

    print("=== Missing values ===")
    if total_missing == 0:
        print("No missing values detected.")
    else:
        print(missing[missing > 0].to_string())
    print()


def print_feature_scales(df: pd.DataFrame) -> None:
    print("=== Feature scale summary ===")
    print(df[FEATURE_COLUMNS].describe().round(4).to_string())
    print()


def print_high_correlations(df: pd.DataFrame, threshold: float = 0.9) -> None:
    corr = df[FEATURE_COLUMNS].corr()
    pairs: list[tuple[str, str, float]] = []

    for i, col_a in enumerate(FEATURE_COLUMNS):
        for col_b in FEATURE_COLUMNS[i + 1:]:
            value = corr.loc[col_a, col_b]
            if abs(value) >= threshold:
                pairs.append((col_a, col_b, value))

    print(f"=== Feature pairs with |correlation| >= {threshold} ===")
    if not pairs:
        print("No pairs above the threshold.")
    else:
        for col_a, col_b, value in sorted(
            pairs,
            key=lambda item: abs(item[2]),
            reverse=True,
        ):
            print(f"{col_a} <-> {col_b}: {value:.3f}")
    print()


def plot_diagnosis_distribution(df: pd.DataFrame) -> None:
    counts = df["diagnosis"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        ax=axes[0],
        palette="Set2",
        legend=False,
    )
    axes[0].set_title("Diagnosis count")
    axes[0].set_xlabel("Diagnosis")
    axes[0].set_ylabel("Count")

    axes[1].pie(
        counts.values,
        labels=[f"{label} ({count})" for label, count in counts.items()],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", n_colors=len(counts)),
    )
    axes[1].set_title("Diagnosis ratio")

    fig.suptitle("Diagnosis distribution (M / B)", fontsize=14)
    fig.tight_layout()


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#4C72B0")
    ax.set_title("Missing values per column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()


def plot_feature_scales(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    sns.boxplot(data=df[FEATURE_COLUMNS], ax=axes[0], color="#4C72B0")
    axes[0].set_title("Feature scales (raw values)")
    axes[0].set_ylabel("Value")
    axes[0].tick_params(axis="x", rotation=90)

    sns.boxplot(data=df[FEATURE_COLUMNS], ax=axes[1], color="#55A868")
    axes[1].set_yscale("log")
    axes[1].set_title("Feature scales (log y-axis)")
    axes[1].set_ylabel("Value (log scale)")
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = load_dataset(args.csv_path)

    print_structure_summary(df)
    print_diagnosis_distribution(df)
    print_missing_values(df)
    print_feature_scales(df)
    print_high_correlations(df)

    plot_diagnosis_distribution(df)
    plot_missing_values(df)
    plot_feature_scales(df)
    plot_correlation_heatmap(df)

    plt.show()


if __name__ == "__main__":
    main()
