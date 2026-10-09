import numpy as np
import pandas as pd

from building_perceptron.audit import audit_features
from building_perceptron.evaluation import (
    build_results_report,
    evaluate_model,
    generalization_report,
    search_perceptron_hyperparameters,
    select_best_model,
)
from building_perceptron.features import FeatureEngineer
from building_perceptron.perceptron import PerceptronClassifier
from building_perceptron.pipeline import prepare_data


def test_perceptron_learns_or_gate() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    model = PerceptronClassifier(learning_rate=0.1, max_iter=100, random_state=1)
    model.fit(X, y)
    assert np.array_equal(model.predict(X), y)
    assert len(model.losses_) == model.n_iter_
    assert model.losses_[-1] == 0.0


def test_perceptron_is_sklearn_compatible() -> None:
    model = PerceptronClassifier()
    params = model.get_params()
    assert params["learning_rate"] == 0.01
    assert "max_iter" in params


def test_perceptron_activation_modes_produce_binary_outputs() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    for activation in ("step", "sigmoid", "tanh"):
        model = PerceptronClassifier(activation=activation, max_iter=20, random_state=1)
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        assert np.all((model.predict(X) == 0) | (model.predict(X) == 1))
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_feature_engineering_adds_domain_ratios() -> None:
    frame = pd.DataFrame({"area_mean": [10.0], "radius_mean": [2.0], "perimeter_mean": [8.0]})
    transformed = FeatureEngineer().fit_transform(frame)
    assert transformed.loc[0, "area_per_radius"] == 5.0
    assert "perimeter_per_radius" in transformed


def test_audit_finds_redundant_features() -> None:
    frame = pd.DataFrame({"first": [1, 2, 3, 4], "second": [2, 4, 6, 8], "other": [4, 1, 3, 2]})
    report = audit_features(frame, pd.Series([0, 0, 1, 1]))
    assert report["redundant_pairs"][0]["feature_1"] == "first"


def test_model_selection_uses_cross_validation_metric() -> None:
    results = {
        "test_best": {"recall": 0.99, "cv": {"recall": 0.70}},
        "cv_best": {"recall": 0.80, "cv": {"recall": 0.90}},
    }
    assert select_best_model(results, "recall") == "cv_best"


def test_generalization_report_exposes_train_validation_test_gaps() -> None:
    results = {
        "model": {
            "train_f1": 1.0,
            "f1": 0.7,
            "cv": {"f1": 0.65},
        }
    }
    report = generalization_report(results, "f1")
    assert report.loc["model", "Écart Train-CV"] == 0.35
    assert report.loc["model", "Diagnostic"] == "Surapprentissage probable"


def test_perceptron_search_evaluates_cartesian_grid() -> None:
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (3, 1))
    y = pd.Series([0, 1, 1, 1] * 3)
    table, best = search_perceptron_hyperparameters(
        X,
        y,
        X,
        y,
        learning_rates=[0.1, 0.2],
        max_iters=[10, 20],
        thresholds=[0.0],
        cv=2,
    )
    assert len(table) == 4
    assert set(best["parameters"]) == {"learning_rate", "max_iter", "threshold", "activation"}


def test_evaluation_records_train_and_test_learning_history() -> None:
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (3, 1))
    y = pd.Series([0, 1, 1, 1] * 3)
    result = evaluate_model(PerceptronClassifier(max_iter=5), X, y, X, y, cv=2)
    history = result["training_history"]
    assert len(history["errors"]) == len(history["test_errors"])
    assert len(history["losses"]) == len(history["test_losses"])


def test_results_report_contains_model_comparison() -> None:
    result = {
        "train_accuracy": 0.9,
        "train_precision": 0.9,
        "train_recall": 0.9,
        "train_f1": 0.9,
        "accuracy": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "roc_auc": 0.85,
        "confusion_matrix": np.array([[8, 2], [2, 8]]),
        "cv": {"accuracy": 0.82, "precision": 0.82, "recall": 0.81, "f1": 0.815},
    }
    report = build_results_report({"Modèle A": result}, "recall")
    assert "Modèle A" in report
    assert "Comparaison relative" in report


def test_prepared_data_keeps_raw_held_out_rows() -> None:
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0, 1] * 10)
    prepared = prepare_data(X, y, test_size=0.2, use_pca=False, feature_engineering=False, random_state=42)
    assert len(prepared.X_test_raw) == len(prepared.y_test)
    assert set(prepared.X_test_raw.index).isdisjoint(set(prepared.X_train_raw.index))
