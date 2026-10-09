from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-motivated, target-independent morphology ratios."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        if not self.enabled:
            return frame
        engineered = frame.copy()
        pairs = {
            "area_per_radius": ("area_mean", "radius_mean"),
            "perimeter_per_radius": ("perimeter_mean", "radius_mean"),
            "concavity_perimeter": ("concavity_mean", "perimeter_mean"),
            "compactness_perimeter": ("compactness_mean", "perimeter_mean"),
        }
        for name, (numerator, denominator) in pairs.items():
            if numerator in frame and denominator in frame:
                engineered[name] = frame[numerator] / frame[denominator].replace(0, np.nan)
        return engineered

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if not self.enabled:
            return np.asarray(names, dtype=object)
        additions = [
            "area_per_radius",
            "perimeter_per_radius",
            "concavity_perimeter",
            "compactness_perimeter",
        ]
        return np.asarray(names + [name for name in additions if name not in names], dtype=object)
