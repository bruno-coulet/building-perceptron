from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC

from .perceptron import PerceptronClassifier


def candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return the documented model set compared in each experiment."""
    return {
        "Perceptron personnalisé": PerceptronClassifier(random_state=random_state),
        "Régression logistique": LogisticRegression(max_iter=2000, random_state=random_state),
        "SVM linéaire": LinearSVC(random_state=random_state, dual="auto", max_iter=5000),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
    }


def select_best_model(results: dict[str, dict[str, Any]], metric: str = "recall") -> str:
    """Select the best model using cross-validation on train, never the test set."""
    if metric not in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
        raise ValueError(f"Métrique de sélection inconnue : {metric}")
    return max(results, key=lambda name: results[name]["cv"][metric])


def search_perceptron_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    learning_rates: list[float],
    max_iters: list[int],
    thresholds: list[float],
    activations: list[str] | None = None,
    metric: str = "recall",
    cv: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a Perceptron grid and select parameters using train CV only."""
    if metric not in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
        raise ValueError(f"Métrique de recherche inconnue : {metric}")
    parameter_grid = ParameterGrid(
        {
            "learning_rate": learning_rates,
            "max_iter": max_iters,
            "threshold": thresholds,
            "activation": activations or ["step"],
        }
    )
    rows: list[dict[str, Any]] = []
    fitted_results: list[dict[str, Any]] = []
    for parameters in parameter_grid:
        result = evaluate_model(
            PerceptronClassifier(**parameters, random_state=random_state),
            X_train,
            y_train,
            X_test,
            y_test,
            cv=cv,
        )
        row = {
            **parameters,
            "train_accuracy": result["train_accuracy"],
            "train_precision": result["train_precision"],
            "train_recall": result["train_recall"],
            "train_f1": result["train_f1"],
            "test_accuracy": result["accuracy"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_f1": result["f1"],
            "test_roc_auc": result["roc_auc"],
            **{f"cv_{name}": value for name, value in result["cv"].items()},
            **{f"cv_std_{name}": value for name, value in result["cv_std"].items()},
            "train_cv_gap": result[f"train_{metric}"] - result["cv"][metric],
            "n_iter": result["model"].n_iter_,
        }
        rows.append(row)
        fitted_results.append(result)

    table = pd.DataFrame(rows)
    best_index = int(
        table.sort_values(
            by=[f"cv_{metric}", f"cv_std_{metric}", "train_cv_gap"],
            ascending=[False, True, True],
        ).index[0]
    )
    best = {
        "parameters": {key: table.loc[best_index, key] for key in ("learning_rate", "max_iter", "threshold", "activation")},
        "metric": metric,
        "cv_score": float(table.loc[best_index, f"cv_{metric}"]),
        "cv_std": float(table.loc[best_index, f"cv_std_{metric}"]),
        "test_score": float(table.loc[best_index, f"test_{metric}"] if metric != "roc_auc" else table.loc[best_index, "test_roc_auc"]),
        "train_cv_gap": float(table.loc[best_index, "train_cv_gap"]),
        "n_iter": int(table.loc[best_index, "n_iter"]),
        "model": fitted_results[best_index]["model"],
    }
    return table, best


def best_perceptron_configuration(table: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Recompute the selected row from a persisted search table."""
    if table.empty or f"cv_{metric}" not in table.columns:
        raise ValueError(f"La table ne contient pas la métrique CV '{metric}'.")
    ranked = table.sort_values(
        by=[f"cv_{metric}", f"cv_std_{metric}", "train_cv_gap"],
        ascending=[False, True, True],
    )
    row = ranked.iloc[0]
    return {
        "parameters": {
            "learning_rate": row["learning_rate"],
            "max_iter": row["max_iter"],
            "threshold": row["threshold"],
            "activation": row.get("activation", "step"),
        },
        "metric": metric,
        "cv_score": float(row[f"cv_{metric}"]),
        "cv_std": float(row.get(f"cv_std_{metric}", np.nan)),
        "test_score": float(row.get(f"test_{metric}", np.nan)),
        "train_cv_gap": float(row.get("train_cv_gap", np.nan)),
        "n_iter": int(row.get("n_iter", row["max_iter"])),
        "selection_rule": "CV score desc, CV standard deviation asc, train-CV gap asc",
    }


def generalization_report(
    results: dict[str, dict[str, Any]], metric: str = "f1"
) -> pd.DataFrame:
    """Compare train, cross-validation, and test scores for generalization checks."""
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Métrique de généralisation inconnue : {metric}")
    rows = []
    for name, result in results.items():
        train = float(result[f"train_{metric}"])
        validation = float(result["cv"][metric])
        test = float(result[metric])
        train_cv_gap = train - validation
        cv_test_gap = validation - test
        if train_cv_gap >= 0.10:
            diagnosis = "Surapprentissage probable"
        elif train < 0.75 and validation < 0.75:
            diagnosis = "Sous-apprentissage possible"
        elif abs(cv_test_gap) >= 0.10:
            diagnosis = "Écart validation/test à examiner"
        else:
            diagnosis = "Généralisation cohérente"
        rows.append(
            {
                "Modèle": name,
                "Train": train,
                "Validation CV": validation,
                "Test": test,
                "Écart Train-CV": train_cv_gap,
                "Écart CV-Test": cv_test_gap,
                "Diagnostic": diagnosis,
            }
        )
    return pd.DataFrame(rows).set_index("Modèle")


def build_results_report(
    results: dict[str, dict[str, Any]], metric: str = "recall"
) -> str:
    """Build a concise scientific interpretation of a model comparison."""
    if not results:
        return "Aucun résultat disponible pour construire le rapport."
    generalization = generalization_report(results, "f1")
    metric_labels = {
        "accuracy": "accuracy",
        "precision": "précision",
        "recall": "rappel des cas malins",
        "f1": "F1-score",
        "roc_auc": "AUC ROC",
    }
    lines = [
        "### Rapport d'interprétation des résultats",
        (
            "Ce rapport compare les modèles sur le train, la validation croisée et le test. "
            "La sélection doit s'appuyer sur la validation croisée ; le test reste une évaluation finale indépendante."
        ),
    ]

    for name, result in results.items():
        f1_row = generalization.loc[name]
        test_metric = float(result[metric])
        cv_metric = float(result["cv"][metric])
        train_metric = float(result.get(f"train_{metric}", np.nan))
        auc = float(result.get("roc_auc", np.nan))
        matrix = np.asarray(result.get("confusion_matrix", [[0, 0], [0, 0]]))
        _, fp, fn, _ = matrix.ravel() if matrix.size == 4 else (0, 0, 0, 0)
        observations: list[str] = []
        if f1_row["Diagnostic"] == "Surapprentissage probable":
            observations.append("écart train-CV compatible avec un surapprentissage")
        elif f1_row["Diagnostic"] == "Sous-apprentissage possible":
            observations.append("scores train et CV faibles, compatibles avec un sous-apprentissage")
        else:
            observations.append("écarts train-CV et CV-test globalement maîtrisés")
        if fn:
            observations.append(f"{int(fn)} faux négatif(s) sur le test")
        if fp:
            observations.append(f"{int(fp)} faux positif(s) sur le test")
        lines.append(
            f"**{name}.** {metric_labels[metric]} : train **{train_metric:.3f}**, "
            f"CV **{cv_metric:.3f}**, test **{test_metric:.3f}** ; "
            f"F1 test **{result['f1']:.3f}**, AUC ROC **{auc:.3f}**. "
            f"Diagnostic : {f1_row['Diagnostic'].lower()} ; "
            + "; ".join(observations) + "."
        )

    lines.append("### Comparaison relative")
    for current_metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        available = {name: float(result[current_metric]) for name, result in results.items() if current_metric in result}
        if available:
            best_name = max(available, key=available.get)
            worst_name = min(available, key=available.get)
            lines.append(
                f"- **{metric_labels[current_metric]} test** : meilleur résultat pour **{best_name}** "
                f"({available[best_name]:.3f}), plus faible pour **{worst_name}** ({available[worst_name]:.3f})."
            )
    best_cv = max(results, key=lambda name: results[name]["cv"][metric])
    lines.append(
        f"### Conclusion opérationnelle\nSelon le critère choisi — **{metric_labels[metric]} moyen en validation croisée** — "
        f"le meilleur modèle est **{best_cv}** ({results[best_cv]['cv'][metric]:.3f}). "
        "Cela ne signifie pas qu'il est le meilleur sur toutes les métriques : "
        "les résultats précédents indiquent le meilleur modèle pour chaque objectif. "
        "Le choix final doit tenir compte de la priorité métier, notamment du rappel et du nombre de faux négatifs, "
        "ainsi que de la stabilité entre folds et des écarts train/test. "
        "Il s'agit d'une comparaison expérimentale et non d'une conclusion clinique."
    )
    return "\n\n".join(lines)


def evaluate_model(model: Any, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series, cv: int = 5) -> dict[str, Any]:
    """Fit a classifier and return test metrics plus stratified CV metrics."""
    fitted = clone(model).fit(X_train, y_train)
    train_prediction = fitted.predict(X_train)
    prediction = fitted.predict(X_test)
    test_scores = _model_scores(fitted, X_test)
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    test_learning_history = _perceptron_test_history(fitted, X_test, y_test)
    scores = {
        "train_accuracy": accuracy_score(y_train, train_prediction),
        "train_precision": precision_score(y_train, train_prediction, zero_division=0),
        "train_recall": recall_score(y_train, train_prediction, zero_division=0),
        "train_f1": f1_score(y_train, train_prediction, zero_division=0),
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, prediction),
        "y_test": np.asarray(y_test),
        "test_scores": test_scores,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
        "training_history": {
            "errors": getattr(fitted, "errors_", []),
            "error_rates": getattr(fitted, "error_rates_", []),
            "losses": getattr(fitted, "losses_", []),
            **test_learning_history,
        },
        "model": fitted,
    }
    scores["roc_auc"] = roc_auc_score(y_test, test_scores)

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        clone(model), X_train, y_train, cv=splitter,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
    )
    scores["cv"] = {metric: float(np.mean(cv_scores[f"test_{metric}"])) for metric in ("accuracy", "precision", "recall", "f1")}
    scores["cv_std"] = {metric: float(np.std(cv_scores[f"test_{metric}"])) for metric in ("accuracy", "precision", "recall", "f1")}
    scores["cv_folds"] = {metric: cv_scores[f"test_{metric}"] for metric in ("accuracy", "precision", "recall", "f1")}
    return scores


def _model_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _perceptron_test_history(model: Any, X_test: np.ndarray, y_test: pd.Series) -> dict[str, list[float]]:
    """Evaluate saved Perceptron states on test data for post-hoc visualization."""
    if not hasattr(model, "weights_history_"):
        return {"test_errors": [], "test_error_rates": [], "test_losses": []}
    signed_labels = 2 * np.asarray(y_test) - 1
    errors: list[float] = []
    error_rates: list[float] = []
    losses: list[float] = []
    for weights, bias in zip(model.weights_history_, model.bias_history_):
        scores = X_test @ weights + bias - model.threshold
        error_count = int(np.count_nonzero((model._activate(scores) >= 0.5).astype(int) != np.asarray(y_test)))
        errors.append(float(error_count))
        error_rates.append(float(error_count / len(y_test)))
        margins = signed_labels * scores
        losses.append(float(np.maximum(0.0, -margins).mean()))
    return {"test_errors": errors, "test_error_rates": error_rates, "test_losses": losses}


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append({"Modèle": name, **{metric.capitalize(): result[metric] for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    return pd.DataFrame(rows).set_index("Modèle")
