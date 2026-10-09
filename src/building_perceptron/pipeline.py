from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureEngineer


@dataclass
class PreparedData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    pipeline: Pipeline
    pca: PCA | None
    feature_engineering: bool


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    use_pca: bool = True,
    n_components: int = 10,
    random_state: int = 42,
    feature_engineering: bool = True,
) -> PreparedData:
    """Split first, then fit all preprocessing steps on train data only."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    base_steps = [
        ("features", FeatureEngineer(enabled=feature_engineering)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    pca = None
    if use_pca:
        valid_components = min(n_components, X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=valid_components, random_state=random_state)
        base_steps.append(("pca", pca))

    transformer = Pipeline(base_steps)
    X_train_ready = transformer.fit_transform(X_train)
    X_test_ready = transformer.transform(X_test)
    engineered_names = list(transformer.named_steps["features"].get_feature_names_out(X.columns))
    feature_names = (
        [f"PC{i + 1}" for i in range(X_train_ready.shape[1])]
        if use_pca
        else engineered_names
    )
    return PreparedData(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train=X_train_ready,
        X_test=X_test_ready,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        pipeline=transformer,
        pca=pca,
        feature_engineering=feature_engineering,
    )
