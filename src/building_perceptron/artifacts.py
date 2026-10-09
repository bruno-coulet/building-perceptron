from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .pipeline import PreparedData


class ExperimentStore:
    """Persist every data-science step in a timestamped, inspectable run folder."""

    def __init__(self, root: str | Path, run_id: str | None = None) -> None:
        self.root = Path(root)
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.run_dir = self.root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"

    def log(self, action: str, **details: Any) -> None:
        event = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "action": action,
            **details,
        }
        with self.trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def save_cleaned(self, frame: pd.DataFrame, report: dict[str, Any]) -> Path:
        path = self.run_dir / "cleaned_data.csv"
        frame.to_csv(path, index=False)
        self.save_json("cleaning_report.json", report)
        self.log("cleaning_saved", path=str(path), rows=len(frame), columns=len(frame.columns))
        return path

    def save_prepared(self, prepared: PreparedData, config: dict[str, Any]) -> Path:
        raw_train = prepared.X_train_raw.copy()
        raw_train["target"] = prepared.y_train.to_numpy()
        raw_test = prepared.X_test_raw.copy()
        raw_test["target"] = prepared.y_test.to_numpy()
        raw_train.to_csv(self.run_dir / "train_raw.csv", index=False)
        raw_test.to_csv(self.run_dir / "test_raw.csv", index=False)
        train = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
        train["target"] = prepared.y_train.to_numpy()
        test = pd.DataFrame(prepared.X_test, columns=prepared.feature_names)
        test["target"] = prepared.y_test.to_numpy()
        train_path = self.run_dir / "train_prepared.csv"
        test_path = self.run_dir / "test_prepared.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        joblib.dump(prepared.pipeline, self.run_dir / "preprocessing.joblib")
        metadata = {
            "config": config,
            "feature_names": prepared.feature_names,
            "train_shape": list(prepared.X_train.shape),
            "test_shape": list(prepared.X_test.shape),
            "train_target_distribution": prepared.y_train.value_counts().sort_index().to_dict(),
            "test_target_distribution": prepared.y_test.value_counts().sort_index().to_dict(),
            "explained_variance_ratio": prepared.pca.explained_variance_ratio_.tolist() if prepared.pca else None,
        }
        self.save_json("preparation_metadata.json", metadata)
        self.log("preparation_saved", train=str(train_path), test=str(test_path), **metadata)
        return self.run_dir

    def save_metrics(self, results: dict[str, dict[str, Any]], config: dict[str, Any]) -> Path:
        serializable: dict[str, Any] = {"schema_version": 2, "config": json_value(config), "models": {}}
        for name, result in results.items():
            serializable["models"][name] = {
                key: json_value(value)
                for key, value in result.items()
                if key not in {"model", "cv"}
            }
            serializable["models"][name]["cv"] = json_value(result["cv"])
            joblib.dump(result["model"], self.run_dir / f"model_{name.replace(' ', '_').lower()}.joblib")
        path = self.save_json("evaluation.json", serializable)
        self.log("evaluation_saved", path=str(path), models=list(results), config=config)
        return path

    def save_selection(self, model_name: str, metric: str) -> Path:
        path = self.save_json("selected_model.json", {"model": model_name, "metric": metric})
        self.log("model_selected", model=model_name, metric=metric, path=str(path))
        return path

    def save_perceptron_search(self, table: pd.DataFrame, best: dict[str, Any]) -> Path:
        """Persist the complete Perceptron hyperparameter experiment."""
        table_path = self.run_dir / "perceptron_hyperparameter_search.csv"
        table.to_csv(table_path, index=False)
        summary = {key: json_value(value) for key, value in best.items() if key != "model"}
        summary_path = self.save_json("perceptron_hyperparameter_best.json", summary)
        if "model" in best:
            joblib.dump(best["model"], self.run_dir / "model_perceptron_optimized.joblib")
        self.log(
            "perceptron_hyperparameter_search_saved",
            table=str(table_path),
            summary=str(summary_path),
            combinations=len(table),
            metric=best["metric"],
        )
        return table_path

    def trace(self) -> list[dict[str, Any]]:
        if not self.trace_path.exists():
            return []
        return [json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines()]

    def manifest(self) -> dict[str, Any]:
        files = sorted(path for path in self.run_dir.iterdir() if path.is_file())
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "files": [path.name for path in files],
            "sha256": {path.name: file_hash(path) for path in files},
        }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    """Convert NumPy values into stable JSON primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def load_prepared(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str], Any]:
    """Reload saved train/test matrices and their fitted preprocessing pipeline."""
    directory = Path(run_dir)
    metadata = json.loads((directory / "preparation_metadata.json").read_text(encoding="utf-8"))
    train = pd.read_csv(directory / "train_prepared.csv")
    test = pd.read_csv(directory / "test_prepared.csv")
    pipeline = joblib.load(directory / "preprocessing.joblib")
    feature_names = metadata["feature_names"]
    return (
        train[feature_names].to_numpy(dtype=float),
        test[feature_names].to_numpy(dtype=float),
        train["target"].astype("int64"),
        test["target"].astype("int64"),
        feature_names,
        pipeline,
    )


def diagnose_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable quality diagnostics before cleaning."""
    missing = frame.isna().sum()
    return {
        "shape": {"rows": len(frame), "columns": len(frame.columns)},
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(value) for column, value in missing.items() if value},
        "duplicate_rows": int(frame.duplicated().sum()),
        "null_rows": int(frame.isna().any(axis=1).sum()),
        "target_distribution": frame["diagnosis"].value_counts(dropna=False).to_dict() if "diagnosis" in frame else {},
    }


def clean_dataset(
    frame: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_empty_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit, deterministic cleaning actions and report their effects."""
    cleaned = frame.copy()
    before = diagnose_dataset(cleaned)
    dropped_duplicates = 0
    dropped_empty: list[str] = []
    if drop_empty_columns:
        dropped_empty = cleaned.columns[cleaned.isna().all()].tolist()
        cleaned = cleaned.drop(columns=dropped_empty)
    if drop_duplicates:
        dropped_duplicates = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report = {
        "before": before,
        "after": diagnose_dataset(cleaned),
        "actions": {
            "drop_duplicates": drop_duplicates,
            "dropped_duplicate_rows": dropped_duplicates,
            "drop_empty_columns": drop_empty_columns,
            "dropped_empty_columns": dropped_empty,
        },
    }
    return cleaned, report
