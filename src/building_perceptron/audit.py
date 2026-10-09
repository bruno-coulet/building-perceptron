from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def audit_features(X: pd.DataFrame, y: pd.Series | None = None, correlation_threshold: float = 0.9) -> dict[str, Any]:
    """Produce an auditable feature-quality and redundancy report."""
    numeric = X.select_dtypes(include=[np.number])
    missing = numeric.isna().mean().sort_values(ascending=False)
    constant = numeric.nunique(dropna=False)
    outlier_counts: dict[str, int] = {}
    for column in numeric:
        values = numeric[column].dropna()
        if values.empty:
            outlier_counts[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr else 0

    correlation = numeric.corr().abs()
    redundant: list[dict[str, Any]] = []
    for index, first in enumerate(correlation.columns):
        for second in correlation.columns[index + 1 :]:
            value = correlation.loc[first, second]
            if pd.notna(value) and value >= correlation_threshold:
                redundant.append({"feature_1": first, "feature_2": second, "absolute_correlation": float(value)})
    redundant.sort(key=lambda item: item["absolute_correlation"], reverse=True)

    useful: list[dict[str, Any]] = []
    if y is not None and len(numeric) > 0:
        filled = numeric.fillna(numeric.median())
        scores = mutual_info_classif(filled, y, random_state=42)
        useful = sorted(
            [{"feature": name, "mutual_information": float(score)} for name, score in zip(numeric.columns, scores)],
            key=lambda item: item["mutual_information"],
            reverse=True,
        )
    return {
        "shape": {"rows": len(X), "features": len(numeric.columns)},
        "missing_rate": {column: float(value) for column, value in missing.items() if value},
        "constant_features": constant[constant <= 1].index.tolist(),
        "outlier_counts_iqr": outlier_counts,
        "redundant_pairs": redundant,
        "useful_features_mutual_information": useful,
    }
