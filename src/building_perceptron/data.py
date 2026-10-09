from pathlib import Path

import pandas as pd

from .config import ID_COLUMN, TARGET


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the Wisconsin Diagnostic CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {ID_COLUMN, TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes : {sorted(missing)}")

    frame[TARGET] = frame[TARGET].astype("string").str.strip().str.upper()
    invalid_labels = sorted(set(frame[TARGET].dropna()) - {"B", "M"})
    if invalid_labels:
        raise ValueError(f"Labels de diagnostic inconnus : {invalid_labels}")
    return frame


def feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate numeric features from the diagnosis target."""
    y = frame[TARGET].map({"B": 0, "M": 1}).astype("int64")
    X = frame.drop(columns=[ID_COLUMN, TARGET]).apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    return X, y
