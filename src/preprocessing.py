"""Preprocess dataset features and labels for model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, LABEL_COLUMN

LABEL_TO_INT = {"B": 0, "M": 1}
INT_TO_LABEL = {value: label for label, value in LABEL_TO_INT.items()}


@dataclass
class Preprocessor:
    """Fit feature standardization on training data and transform splits."""

    feature_means_: pd.Series | None = None
    feature_stds_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        features = _extract_features(df)
        self.feature_means_ = features.mean()
        self.feature_stds_ = features.std(ddof=0)
        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_means_ is None or self.feature_stds_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        features = _standardize(
            _extract_features(df),
            self.feature_means_,
            self.feature_stds_,
        )
        labels = encode_labels(df[LABEL_COLUMN])
        return features.to_numpy(dtype=np.float64), labels

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = set(labels.unique()) - set(LABEL_TO_INT)
    if unknown:
        raise ValueError(f"Unexpected diagnosis labels: {sorted(unknown)}")

    return labels.map(LABEL_TO_INT).to_numpy(dtype=np.int64)


def _standardize(
    features: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    safe_stds = stds.replace(0, 1)
    return (features - means) / safe_stds
