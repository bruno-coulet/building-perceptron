import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from building_perceptron.artifacts import (
    ExperimentStore,
    clean_dataset,
    diagnose_dataset,
    load_prepared,
)
from building_perceptron.audit import audit_features
from building_perceptron.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DESCRIPTION_PATH,
    RANDOM_STATE,
    TARGET_LABELS,
)
from building_perceptron.data import feature_frame, load_dataset
from building_perceptron.evaluation import (
    best_perceptron_configuration,
    build_results_report,
    candidate_models,
    evaluate_model,
    generalization_report,
    search_perceptron_hyperparameters,
    select_best_model,
)
from building_perceptron.perceptron import PerceptronClassifier
from building_perceptron.pipeline import prepare_data

st.set_page_config(page_title="Breast Cancer Lab", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 1.4rem 0 1rem; border-bottom: 1px solid #d9e1e8; margin-bottom: 1.4rem; }
    .eyebrow { color: #087f8c; text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 700; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.8rem); line-height: 1.05; margin: .25rem 0 .7rem; color: #102a43; }
    .hero p { max-width: 820px; color: #486581; font-size: 1.05rem; }
    [data-testid="stMetric"] { background: #f4f8fa; border-left: 4px solid #f08a5d; padding: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data
def get_description() -> str:
    return Path(DESCRIPTION_PATH).read_text(encoding="utf-8")


def parse_grid_values(raw: str, value_type: type) -> list[float] | list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("La grille ne peut pas être vide.")
    return [value_type(item) for item in values]


def metric_cell_style(value: object) -> str:
    """Color metric cells from red (low) to green (high), keeping N/D readable."""
    if not isinstance(value, (int, float, np.number)) or pd.isna(value):
        return "color: #829ab1; font-style: italic;"
    score = max(0.0, min(1.0, float(value)))
    red = int(220 * (1 - score) + 46 * score)
    green = int(70 * (1 - score) + 160 * score)
    blue = int(65 * (1 - score) + 90 * score)
    text_color = "#102a43" if 0.2 < score < 0.8 else "#ffffff"
    return f"background-color: rgb({red}, {green}, {blue}); color: {text_color};"


def generalization_gap_style(value: object) -> str:
    """Color small train/CV/test gaps green and large gaps red."""
    if not isinstance(value, (int, float, np.number)) or pd.isna(value):
        return "color: #829ab1; font-style: italic;"
    gap = min(1.0, abs(float(value)) / 0.20)
    return metric_cell_style(1.0 - gap)


def generalization_diagnostic_style(value: object) -> str:
    """Color the textual generalization diagnosis by risk level."""
    text = str(value)
    if "Surapprentissage" in text or "examiner" in text:
        return "background-color: #f8c4b4; color: #7b241c; font-weight: 600;"
    if "Sous-apprentissage" in text:
        return "background-color: #fce8a6; color: #7d5a00; font-weight: 600;"
    return "background-color: #b7e4c7; color: #1b4332; font-weight: 600;"


def relative_score_column_style(column: pd.Series) -> pd.Series:
    """Color the lowest model score red and the highest green within a column."""
    numeric = pd.to_numeric(column, errors="coerce")
    valid = numeric.dropna()
    styles = pd.Series("", index=column.index, dtype="object")
    if valid.empty:
        return styles
    minimum, maximum = valid.min(), valid.max()
    scale = maximum - minimum
    for index, value in numeric.items():
        if pd.notna(value):
            normalized = 0.5 if scale == 0 else (value - minimum) / scale
            styles.loc[index] = metric_cell_style(normalized)
        else:
            styles.loc[index] = "color: #829ab1; font-style: italic;"
    return styles


def relative_score_row_style(row: pd.Series) -> pd.Series:
    """Color train/CV/test relatively within one model and one metric."""
    numeric = pd.to_numeric(row, errors="coerce")
    valid = numeric.dropna()
    styles = pd.Series("", index=row.index, dtype="object")
    if valid.empty:
        return styles
    minimum, maximum = valid.min(), valid.max()
    scale = maximum - minimum
    for index, value in numeric.items():
        if pd.notna(value):
            normalized = 0.5 if scale == 0 else (value - minimum) / scale
            styles.loc[index] = metric_cell_style(normalized)
    return styles


def relative_gap_column_style(column: pd.Series) -> pd.Series:
    """Color the smallest generalization gap green and the largest red."""
    numeric = pd.to_numeric(column, errors="coerce").abs()
    valid = numeric.dropna()
    styles = pd.Series("", index=column.index, dtype="object")
    if valid.empty:
        return styles
    minimum, maximum = valid.min(), valid.max()
    scale = maximum - minimum
    for index, value in numeric.items():
        if pd.notna(value):
            normalized = 0.5 if scale == 0 else 1 - (value - minimum) / scale
            styles.loc[index] = metric_cell_style(normalized)
    return styles


if "store" not in st.session_state:
    st.session_state.store = ExperimentStore(ARTIFACTS_DIR)
store: ExperimentStore = st.session_state.store
frame = get_data()

with st.sidebar:
    st.markdown("## Breast Cancer Lab")
    st.caption("Workflow scientifique persistant")
    section = st.radio(
        "Étape du workflow",
        [
            "1 · Diagnostiquer et nettoyer",
            "2 · Auditer les variables",
            "3 · Préparer, feature engineering et PCA",
            "4 · Optimiser le Perceptron",
            "5 · Entraîner, comparer et valider",
            "6 · Choisir et prédire",
            "7 · Journal de traçabilité",
        ],
    )
    st.divider()
    existing_runs = sorted((ARTIFACTS_DIR / "runs").glob("*")) if (ARTIFACTS_DIR / "runs").exists() else []
    run_options = ["Nouvelle expérience"] + [path.name for path in reversed(existing_runs) if path.is_dir()]
    selected_run = st.selectbox("Expérience à charger", run_options, index=min(run_options.index(store.run_id), len(run_options) - 1) if store.run_id in run_options else 0)
    if selected_run != "Nouvelle expérience" and selected_run != store.run_id:
        st.session_state.store = ExperimentStore(ARTIFACTS_DIR, selected_run)
        st.rerun()
    st.caption("Expérience active")
    st.code(store.run_id, language="text")
    st.caption(f"Artefacts : {store.run_dir}")
    if st.button("Nouvelle expérience"):
        st.session_state.store = ExperimentStore(ARTIFACTS_DIR)
        st.rerun()

st.markdown(
    '<div class="hero"><div class="eyebrow">Laboratoire de modélisation médicale</div><h1>Breast Cancer Lab</h1><p>Chaque action est explicite, enregistrée et rejouable : diagnostic, nettoyage, préparation PCA, entraînement et validation.</p></div>',
    unsafe_allow_html=True,
)

if section == "1 · Diagnostiquer et nettoyer":
    st.subheader("Diagnostic initial et nettoyage contrôlé")
    st.info("Le diagnostic est calculé avant toute modification. Les options de nettoyage sont appliquées uniquement lorsque vous cliquez sur le bouton d'enregistrement.")
    diagnosis_path = store.run_dir / "diagnosis_raw.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8")) if diagnosis_path.exists() else diagnose_dataset(frame)
    if not diagnosis_path.exists():
        store.save_json("diagnosis_raw.json", diagnosis)
        store.log("diagnosis_computed", source=str(DATA_PATH), **diagnosis["shape"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", diagnosis["shape"]["rows"])
    c2.metric("Colonnes", diagnosis["shape"]["columns"])
    c3.metric("Doublons", diagnosis["duplicate_rows"])
    c4.metric("Cellules manquantes", sum(diagnosis["missing"].values()))
    left, right = st.columns(2)
    with left:
        st.markdown("### Valeurs manquantes")
        missing = pd.Series(diagnosis["missing"], name="Effectif").sort_values(ascending=False)
        st.dataframe(missing if not missing.empty else pd.DataFrame({"Statut": ["Aucune valeur manquante"]}), use_container_width=True)
    with right:
        st.markdown("### Répartition de la cible")
        target_counts = frame["diagnosis"].map(TARGET_LABELS).value_counts().rename("Effectif")
        st.plotly_chart(px.bar(target_counts, orientation="h", color_discrete_sequence=["#087f8c"]), use_container_width=True)
    st.markdown("### Actions de nettoyage")
    drop_duplicates = st.checkbox("Supprimer les doublons exacts", value=True)
    drop_empty = st.checkbox("Supprimer les colonnes entièrement vides", value=True)
    if st.button("Appliquer et enregistrer le nettoyage", type="primary"):
        cleaned, report = clean_dataset(frame, drop_duplicates, drop_empty)
        store.save_cleaned(cleaned, report)
        store.save_json("manifest_after_cleaning.json", store.manifest())
        st.session_state.cleaned_frame = cleaned
        st.success(f"Nettoyage enregistré : {len(cleaned)} lignes et {len(cleaned.columns)} colonnes.")
        st.json(report["actions"])
    cleaned_path = store.run_dir / "cleaned_data.csv"
    if cleaned_path.exists():
        cleaned = pd.read_csv(cleaned_path)
        st.caption(f"Dernier nettoyage enregistré : {cleaned_path}")
        st.dataframe(cleaned.head(10), use_container_width=True)
    with st.expander("Descriptif des variables"):
        st.markdown(get_description())

elif section == "2 · Auditer les variables":
    st.subheader("Audit des variables et redondances")
    cleaned_path = store.run_dir / "cleaned_data.csv"
    if not cleaned_path.exists():
        st.warning("Commencez par enregistrer le nettoyage de cette expérience.")
    else:
        cleaned = pd.read_csv(cleaned_path)
        X, y = feature_frame(cleaned)
        threshold = st.slider("Seuil de corrélation absolue pour redondance", .70, .99, .90, .01)
        audit = audit_features(X, y, threshold)
        audit_path = store.run_dir / "feature_audit.json"
        audit["correlation_threshold"] = threshold
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        store.save_json("feature_audit.json", audit)
        if previous_audit.get("correlation_threshold") != threshold:
            store.log("feature_audit_computed", correlation_threshold=threshold, **audit["shape"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables analysées", audit["shape"]["features"])
        c2.metric("Variables constantes", len(audit["constant_features"]))
        c3.metric("Paires redondantes", len(audit["redundant_pairs"]))
        c4.metric("Variables atypiques", sum(count > 0 for count in audit["outlier_counts_iqr"].values()))
        left, right = st.columns(2)
        with left:
            st.markdown("### Variables potentiellement utiles")
            useful = pd.DataFrame(audit["useful_features_mutual_information"])
            st.dataframe(useful.head(15), use_container_width=True)
            if not useful.empty:
                st.plotly_chart(px.bar(useful.head(15).sort_values("mutual_information"), x="mutual_information", y="feature", orientation="h", color_discrete_sequence=["#087f8c"]), use_container_width=True)
        with right:
            st.markdown("### Paires redondantes")
            redundant = pd.DataFrame(audit["redundant_pairs"])
            st.dataframe(redundant.head(30) if not redundant.empty else pd.DataFrame({"Statut": ["Aucune paire au seuil choisi"]}), use_container_width=True)
            st.markdown("### Valeurs atypiques par règle IQR")
            outliers = pd.Series(audit["outlier_counts_iqr"], name="Nombre").sort_values(ascending=False)
            st.dataframe(outliers.head(15), use_container_width=True)
        st.caption("La mutual information et les corrélations sont des outils d'exploration. La sélection finale et la PCA doivent rester ajustées sur le train uniquement.")

elif section == "3 · Préparer, feature engineering et PCA":
    st.subheader("Préparation reproductible et réduction PCA")
    cleaned_path = store.run_dir / "cleaned_data.csv"
    if not cleaned_path.exists():
        st.warning("Commencez par diagnostiquer et enregistrer le nettoyage de cette expérience.")
    else:
        cleaned = pd.read_csv(cleaned_path)
        X, y = feature_frame(cleaned)
        st.write(f"Source utilisée : `{cleaned_path.name}` · {len(cleaned)} observations · {X.shape[1]} variables numériques")
        c1, c2, c3, c4 = st.columns(4)
        test_size = c1.slider("Part du test", .15, .35, .20, .05)
        feature_engineering = c2.toggle("Feature engineering", value=True)
        use_pca = c3.toggle("Activer PCA", value=True)
        n_components = c4.slider("Composantes", 2, X.shape[1] + 4, min(10, X.shape[1])) if use_pca else X.shape[1] + 4
        if st.button("Construire et enregistrer train/test", type="primary"):
            prepared = prepare_data(X, y, test_size, use_pca, n_components, RANDOM_STATE, feature_engineering)
            config = {"test_size": test_size, "use_pca": use_pca, "n_components": n_components, "feature_engineering": feature_engineering, "random_state": RANDOM_STATE, "source": str(cleaned_path)}
            store.save_prepared(prepared, config)
            store.save_json("manifest_after_preparation.json", store.manifest())
            st.success("Les matrices train/test, le pipeline de transformation et les métadonnées PCA sont enregistrés.")
        metadata_path = store.run_dir / "preparation_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Train enregistré", metadata["train_shape"][0])
            b.metric("Test enregistré", metadata["test_shape"][0])
            c.metric("Dimensions finales", metadata["train_shape"][1])
            distributions = pd.DataFrame({
                "Original": frame["diagnosis"].map({"B": 0, "M": 1}).value_counts(normalize=True).sort_index(),
                "Train": pd.Series(metadata["train_target_distribution"], dtype=float).rename(index=lambda value: int(value)).sort_index() / metadata["train_shape"][0],
                "Test": pd.Series(metadata["test_target_distribution"], dtype=float).rename(index=lambda value: int(value)).sort_index() / metadata["test_shape"][0],
            }, index=[0, 1]).rename(index={0: "Bénigne", 1: "Maligne"})
            st.markdown("### Contrôle de la stratification")
            st.dataframe((100 * distributions).round(2).astype(str) + " %", use_container_width=True)
            explained = metadata["explained_variance_ratio"]
            if explained:
                chart = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(explained))], y=explained, name="Variance par composante", marker_color="#087f8c"))
                chart.add_scatter(x=[f"PC{i + 1}" for i in range(len(explained))], y=np.cumsum(explained), mode="lines+markers", name="Cumul", line={"color": "#f08a5d"})
                chart.update_layout(yaxis_title="Variance expliquée", xaxis_title="Composante")
                st.plotly_chart(chart, use_container_width=True)
            st.download_button("Télécharger les métadonnées", metadata_path.read_bytes(), metadata_path.name, "application/json")

elif section == "4 · Optimiser le Perceptron":
    st.subheader("Laboratoire d'hyperparamètres du Perceptron")
    st.info("Chaque configuration est évaluée avec la même validation croisée stratifiée. Le meilleur réglage est choisi sur le train ; le test sert uniquement à mesurer cette configuration après sélection.")
    train_path = store.run_dir / "train_prepared.csv"
    test_path = store.run_dir / "test_prepared.csv"
    if not train_path.exists() or not test_path.exists():
        st.warning("Préparez et enregistrez d'abord les données train/test.")
    else:
        X_train, X_test, y_train, y_test, _, _ = load_prepared(store.run_dir)
        optimized_defaults = st.session_state.get(
            "perceptron_defaults",
            {"learning_rate": 0.01, "max_iter": 1000, "threshold": 0.0, "activation": "step"},
        )
        with st.form("perceptron_search_form"):
            c1, c2, c3, c4 = st.columns(4)
            learning_rates_text = c1.text_input("Taux d'apprentissage", "0.001, 0.01, 0.1")
            max_iters_text = c2.text_input("Itérations maximales", "100, 500, 1000")
            thresholds_text = c3.text_input("Seuils d'activation", "-0.5, 0.0, 0.5")
            activations_text = c4.text_input("Fonctions d'activation", "step, sigmoid, tanh")
            search_metric = st.selectbox("Métrique de sélection CV", ["recall", "f1", "accuracy", "precision"])
            run_search = st.form_submit_button("Lancer la recherche", type="primary")
        if run_search:
            try:
                learning_rates = parse_grid_values(learning_rates_text, float)
                max_iters = parse_grid_values(max_iters_text, int)
                thresholds = parse_grid_values(thresholds_text, float)
                activations = [value.strip() for value in activations_text.split(",") if value.strip()]
                invalid_activations = set(activations) - {"step", "sigmoid", "tanh"}
                if not activations or invalid_activations:
                    raise ValueError("Les activations autorisées sont : step, sigmoid, tanh.")
                search_table, best = search_perceptron_hyperparameters(
                    X_train, y_train, X_test, y_test,
                    learning_rates, max_iters, thresholds,
                    activations=activations,
                    metric=search_metric, random_state=RANDOM_STATE,
                )
                store.save_perceptron_search(search_table, best)
                store.save_json("manifest_after_perceptron_search.json", store.manifest())
                st.session_state.perceptron_search = search_table
                st.session_state.perceptron_best = best
                st.success(f"Recherche terminée : {len(search_table)} configurations évaluées.")
            except ValueError as error:
                st.error(f"Grille invalide : {error}")
        search_path = store.run_dir / "perceptron_hyperparameter_search.csv"
        search_table = st.session_state.get("perceptron_search")
        if search_table is None and search_path.exists():
            search_table = pd.read_csv(search_path)
        if search_table is not None:
            best = best_perceptron_configuration(search_table, search_metric)
            st.session_state.perceptron_best = best
            if best:
                st.markdown("### Meilleur compromis détecté")
                st.json({key: value for key, value in best.items() if key != "model"})
                if st.button("Injecter le meilleur réglage dans l'entraînement", type="primary"):
                    st.session_state.perceptron_defaults = best["parameters"]
                    st.success("Réglage injecté. Ouvrez l'étape d'entraînement pour le vérifier et l'utiliser.")
            st.markdown("### Toutes les configurations")
            display_columns = [column for column in ["learning_rate", "max_iter", "threshold", "activation", "train_f1", "cv_f1", "test_f1", "train_recall", "cv_recall", "test_recall", "cv_std_recall", "train_cv_gap", "n_iter"] if column in search_table]
            st.dataframe(
                search_table.sort_values(f"cv_{search_metric}", ascending=False)[display_columns].style.format(
                    lambda value: f"{value:.3f}"
                    if isinstance(value, (int, float, np.number)) and pd.notna(value)
                    else str(value)
                ),
                use_container_width=True,
            )
            st.plotly_chart(px.scatter(search_table, x=f"cv_{search_metric}", y="test_recall", size="n_iter", color="activation", symbol="activation", hover_data=["learning_rate", "max_iter", "threshold"], title=f"Recherche Perceptron : CV {search_metric} vs rappel test"), use_container_width=True)
            st.plotly_chart(px.scatter_3d(search_table, x="learning_rate", y="max_iter", z=f"cv_{search_metric}", color="activation", hover_data=["threshold", "test_recall", "train_cv_gap"], title="Espace des hyperparamètres"), use_container_width=True)

elif section == "5 · Entraîner, comparer et valider":
    st.subheader("Entraînement, comparaison et validation sur les artefacts enregistrés")
    train_path = store.run_dir / "train_prepared.csv"
    test_path = store.run_dir / "test_prepared.csv"
    if not train_path.exists() or not test_path.exists():
        st.warning("Préparez et enregistrez d'abord les données train/test dans l'étape 2.")
    else:
        X_train, X_test, y_train, y_test, feature_names, _ = load_prepared(store.run_dir)
        st.write(f"Les données utilisées viennent de `{train_path.name}` et `{test_path.name}`. Aucun recalcul depuis les données brutes n'est effectué.")
        optimized_defaults = st.session_state.get(
            "perceptron_defaults",
            {"learning_rate": 0.01, "max_iter": 1000, "threshold": 0.0, "activation": "step"},
        )
        with st.form("training"):
            c1, c2, c3, c4, c5 = st.columns(5)
            learning_rate = c1.number_input("Taux d'apprentissage", .0001, 1.0, float(optimized_defaults.get("learning_rate", 0.01)), format="%.4f")
            max_iter = c2.number_input("Itérations maximales", 10, 5000, int(optimized_defaults.get("max_iter", 1000)), step=50)
            threshold = c3.number_input("Seuil d'activation", -5.0, 5.0, float(optimized_defaults.get("threshold", 0.0)), step=.1)
            activation = c4.selectbox("Fonction d'activation", ["step", "sigmoid", "tanh"], index=["step", "sigmoid", "tanh"].index(optimized_defaults.get("activation", "step")))
            selection_metric = c5.selectbox("Métrique de sélection", ["recall", "f1", "roc_auc", "precision", "accuracy"])
            selection_mode = st.selectbox("Mode de sélection", ["Automatique (CV)", "Manuel"])
            manual_model = st.selectbox("Modèle à retenir si sélection manuelle", list(candidate_models(RANDOM_STATE)))
            submitted = st.form_submit_button("Entraîner et enregistrer l'évaluation", type="primary")
        if submitted:
            models = candidate_models(RANDOM_STATE)
            models["Perceptron personnalisé"] = PerceptronClassifier(learning_rate=learning_rate, max_iter=max_iter, threshold=threshold, activation=activation, random_state=RANDOM_STATE)
            results = {name: evaluate_model(model, X_train, y_train, X_test, y_test) for name, model in models.items()}
            automatic_model = select_best_model(results, selection_metric)
            selected_model = automatic_model if selection_mode.startswith("Automatique") else manual_model
            config = {"learning_rate": learning_rate, "max_iter": max_iter, "threshold": threshold, "activation": activation, "selection_metric": selection_metric, "selection_mode": selection_mode, "automatic_model": automatic_model, "random_state": RANDOM_STATE, "source_train": str(train_path), "source_test": str(test_path)}
            store.save_metrics(results, config)
            store.save_selection(selected_model, selection_metric)
            store.save_json("manifest_after_evaluation.json", store.manifest())
            st.session_state.results = results
            st.success(f"Évaluation enregistrée. Modèle retenu : {selected_model} ({selection_mode}).")
        evaluation_path = store.run_dir / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            current_schema = evaluation.get("schema_version", 1)
            if current_schema < 2:
                st.warning("Cette évaluation provient d'un ancien format : les métriques train et les courbes ROC ne sont pas enregistrées. Relancez l'entraînement pour actualiser les artefacts.")
            current_results = st.session_state.get("results")
            if current_results is None and current_schema >= 2:
                current_results = {
                    name: {**values, "cv": values.get("cv", {})}
                    for name, values in evaluation["models"].items()
                }
            if current_results:
                st.markdown(build_results_report(current_results, evaluation.get("config", {}).get("selection_metric", "recall")))
            metric_rows = []
            for name, values in evaluation["models"].items():
                row = {"Modèle": name}
                row.update({f"Train {metric}": values.get(f"train_{metric}", "N/D") for metric in ("accuracy", "precision", "recall", "f1")})
                row.update({f"Test {metric}": values[metric] for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")})
                row.update({f"CV {metric}": values["cv"][metric] for metric in ("accuracy", "precision", "recall", "f1")})
                metric_rows.append(row)
            metric_frame = pd.DataFrame(metric_rows).set_index("Modèle")
            st.markdown("### Synthèse test et validation croisée")
            st.caption("Dans ce tableau, chaque colonne est normalisée entre les modèles : le meilleur score de la colonne est vert, le moins bon rouge. Les couleurs montrent donc le classement relatif, pas un seuil absolu de qualité.")
            metric_columns = [column for column in metric_frame.columns if column.startswith(("Train ", "Test ", "CV "))]
            st.dataframe(
                metric_frame.style.format(
                    lambda value: f"{value:.3f}"
                    if isinstance(value, (int, float, np.number)) and pd.notna(value)
                    else str(value)
                ).apply(relative_score_column_style, subset=metric_columns, axis=0),
                use_container_width=True,
            )
            profile_rows = []
            for model_name, values in evaluation["models"].items():
                for metric in ("accuracy", "precision", "recall", "f1"):
                    profile_rows.append(
                        {
                            "Modèle": model_name,
                            "Métrique": metric,
                            "Train": values.get(f"train_{metric}", "N/D"),
                            "CV": values.get("cv", {}).get(metric, "N/D"),
                            "Test": values.get(metric, "N/D"),
                        }
                    )
            profile_frame = pd.DataFrame(profile_rows)
            st.markdown("### Profil relatif de chaque modèle")
            st.caption("Chaque ligne compare train, CV et test pour un modèle et une métrique donnés. Le vert indique la valeur la plus élevée des trois, le rouge la plus faible.")
            st.dataframe(
                profile_frame.style.format(
                    lambda value: f"{value:.3f}"
                    if isinstance(value, (int, float, np.number)) and pd.notna(value)
                    else str(value)
                ).apply(relative_score_row_style, subset=["Train", "CV", "Test"], axis=1),
                use_container_width=True,
            )
            st.markdown("### Analyse de généralisation")
            diagnostic_metric = st.selectbox(
                "Métrique à analyser pour les écarts",
                ["f1", "recall", "accuracy", "precision"],
            )
            if current_results:
                report = generalization_report(current_results, diagnostic_metric)
                st.dataframe(
                    report.style.format(
                        lambda value: f"{value:.3f}"
                        if isinstance(value, (int, float, np.number)) and pd.notna(value)
                        else str(value)
                    ).map(metric_cell_style, subset=["Train", "Validation CV", "Test"])
                    .apply(relative_gap_column_style, subset=["Écart Train-CV", "Écart CV-Test"], axis=0)
                    .map(generalization_diagnostic_style, subset=["Diagnostic"]),
                    use_container_width=True,
                )
                comparison = report.reset_index().melt(
                    id_vars="Modèle",
                    value_vars=["Train", "Validation CV", "Test"],
                    var_name="Jeu",
                    value_name="Score",
                )
                st.plotly_chart(
                    px.bar(
                        comparison,
                        x="Modèle",
                        y="Score",
                        color="Jeu",
                        barmode="group",
                        range_y=[0, 1],
                        title=f"{diagnostic_metric.upper()} : train / validation / test",
                    ),
                    use_container_width=True,
                )
                gaps = report.reset_index().melt(
                    id_vars="Modèle",
                    value_vars=["Écart Train-CV", "Écart CV-Test"],
                    var_name="Écart",
                    value_name="Valeur",
                )
                st.plotly_chart(
                    px.bar(
                        gaps,
                        x="Modèle",
                        y="Valeur",
                        color="Écart",
                        barmode="group",
                        title="Écarts de généralisation",
                    ),
                    use_container_width=True,
                )
                for model_name, row in report.iterrows():
                    if row["Diagnostic"] != "Généralisation cohérente":
                        st.warning(f"{model_name} : {row['Diagnostic']}.")
            else:
                st.info("Relancez l'entraînement pour calculer l'analyse train/CV/test sur les résultats actuels.")
            selection_path = store.run_dir / "selected_model.json"
            if selection_path.exists():
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                st.info(f"Modèle sélectionné comme meilleur **selon le critère choisi** sur la validation croisée du train : **{selection['model']}** selon **{selection['metric']}**. Ce choix ne signifie pas qu'il domine toutes les autres métriques.")
            model_names = list(evaluation["models"])
            selected = st.selectbox("Matrice de confusion", model_names)
            matrix = evaluation["models"][selected]["confusion_matrix"]
            st.plotly_chart(px.imshow(matrix, text_auto=True, x=["Bénigne prédite", "Maligne prédite"], y=["Bénigne réelle", "Maligne réelle"], color_continuous_scale="Tealgrn"), use_container_width=True)
            st.markdown("### Courbes ROC test")
            roc_chart = go.Figure()
            roc_count = 0
            for name, values in evaluation["models"].items():
                curve = values.get("roc_curve")
                if curve:
                    roc_chart.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{name} (AUC {values['roc_auc']:.3f})"))
                    roc_count += 1
            if roc_count:
                roc_chart.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line={"dash": "dash", "color": "#9aa5b1"}))
                roc_chart.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", yaxis_range=[0, 1], xaxis_range=[0, 1])
                st.plotly_chart(roc_chart, use_container_width=True)
            else:
                st.info("Aucune courbe ROC n'est disponible dans cet artefact. Relancez l'entraînement avec le code actuel.")
            st.markdown("### Comparaison des métriques de validation croisée")
            cv_rows = [{"Modèle": name, "Métrique": metric, "Score": values["cv"][metric], "Écart-type": values.get("cv_std", {}).get(metric, 0)} for name, values in evaluation["models"].items() for metric in ("accuracy", "precision", "recall", "f1")]
            st.plotly_chart(px.bar(pd.DataFrame(cv_rows), x="Métrique", y="Score", color="Modèle", barmode="group", error_y="Écart-type", range_y=[0, 1]), use_container_width=True)
            stored_results = st.session_state.get("results")
            if stored_results and selected in stored_results and "Perceptron" in selected:
                model = stored_results[selected]["model"]
                history = {
                    "errors": model.errors_,
                    "error_rates": model.error_rates_,
                    "losses": model.losses_,
                    "test_errors": evaluation["models"][selected].get("training_history", {}).get("test_errors", []),
                    "test_error_rates": evaluation["models"][selected].get("training_history", {}).get("test_error_rates", []),
                    "test_losses": evaluation["models"][selected].get("training_history", {}).get("test_losses", []),
                }
            else:
                history = evaluation["models"][selected].get("training_history", {})
            if "Perceptron" in selected and history.get("errors"):
                history_frame = pd.DataFrame(
                    {
                        "Itération": range(1, len(history["errors"]) + 1),
                        "Erreurs de classification": history["errors"],
                        "Taux d'erreur train": history.get("error_rates", []),
                        "Perte du Perceptron": history.get("losses", []),
                        "Erreurs test": history.get("test_errors", []),
                        "Taux d'erreur test": history.get("test_error_rates", []),
                        "Perte test": history.get("test_losses", []),
                    }
                )
                st.markdown("### Dynamique d'apprentissage du Perceptron")
                st.caption("La convergence est considérée atteinte si le nombre d'erreurs train devient nul. Les courbes test sont descriptives uniquement et ne doivent pas servir à choisir les hyperparamètres.")
                learning_chart = go.Figure()
                learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Erreurs de classification"], mode="lines+markers", name="Erreurs train", yaxis="y"))
                learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Taux d'erreur train"], mode="lines", name="Taux erreur train", yaxis="y3", line={"color": "#2f855a"}))
                learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Perte du Perceptron"], mode="lines+markers", name="Perte train", yaxis="y2", line={"color": "#087f8c"}))
                if history.get("test_errors"):
                    learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Erreurs test"], mode="lines", name="Erreurs test", yaxis="y", line={"dash": "dot", "color": "#f08a5d"}))
                    learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Taux d'erreur test"], mode="lines", name="Taux erreur test", yaxis="y3", line={"dash": "dot", "color": "#c53030"}))
                    learning_chart.add_trace(go.Scatter(x=history_frame["Itération"], y=history_frame["Perte test"], mode="lines", name="Perte test", yaxis="y2", line={"dash": "dot", "color": "#102a43"}))
                learning_chart.update_layout(
                    xaxis_title="Itération",
                    yaxis={"title": "Nombre d'erreurs", "rangemode": "tozero"},
                    yaxis2={"title": "Perte moyenne", "overlaying": "y", "side": "right", "rangemode": "tozero"},
                    yaxis3={"title": "Taux d'erreur", "overlaying": "y", "side": "right", "position": 1.0, "range": [0, 1]},
                )
                st.plotly_chart(learning_chart, use_container_width=True)
                if history["errors"][-1] == 0:
                    st.success(f"Convergence opérationnelle atteinte après {len(history['errors'])} itérations : aucune erreur sur le train à la dernière époque.")
                elif len(history["errors"]) >= max_iter:
                    st.warning("Le Perceptron a atteint la limite d'itérations sans erreur nulle : il n'est pas démontré comme convergent sur ce train.")
                    with st.expander("Interprétation scientifique"):
                        st.write(
                            "Ce résultat ne signifie pas que le Perceptron est inutilisable. "
                            "Le théorème de convergence garantit l'arrêt avec zéro erreur uniquement "
                            "si les classes sont linéairement séparables. Ici, les données peuvent ne pas "
                            "être parfaitement séparables par une frontière linéaire. Le modèle peut donc "
                            "conserver de bonnes performances en test tout en continuant à faire quelques "
                            "erreurs sur le train. Augmenter max_iter peut être testé, mais ne garantit pas "
                            "la convergence si les données ne sont pas séparables."
                        )
                        st.markdown(
                            "**Conclusion à retenir :** le Perceptron peut être performant et pédagogique, "
                            "mais sa convergence parfaite n'est pas démontrée sur ces données."
                        )
                else:
                    st.info("Les erreurs ne sont pas nulles, mais l'entraînement s'est arrêté avant la limite : inspectez la courbe de perte et les paramètres.")
            st.warning("Dans ce contexte médical, le rappel des cas malins doit être examiné avant l'accuracy.")

elif section == "6 · Choisir et prédire":
    st.subheader("Sélection du modèle et prédiction")
    evaluation_path = store.run_dir / "evaluation.json"
    preparation_path = store.run_dir / "preparation_metadata.json"
    cleaned_path = store.run_dir / "cleaned_data.csv"
    test_raw_path = store.run_dir / "test_raw.csv"
    if not evaluation_path.exists() or not preparation_path.exists() or not cleaned_path.exists() or not test_raw_path.exists():
        st.warning("Terminez le nettoyage, la préparation et l'entraînement avec la version actuelle avant de prédire. Les scénarios doivent provenir du test tenu à l'écart de l'entraînement.")
    else:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        selected_path = store.run_dir / "selected_model.json"
        saved_selection = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
        model_names = list(evaluation["models"])
        default_model = saved_selection.get("model", model_names[0])
        model_name = st.selectbox("Modèle à utiliser", model_names, index=model_names.index(default_model) if default_model in model_names else 0)
        if st.button("Enregistrer ce modèle pour les prédictions", type="primary"):
            metric = saved_selection.get("metric", "recall")
            store.save_selection(model_name, metric)
            st.success(f"Modèle actif enregistré : {model_name}.")
        model_path = store.run_dir / f"model_{model_name.replace(' ', '_').lower()}.joblib"
        pipeline = joblib.load(store.run_dir / "preprocessing.joblib")
        model = joblib.load(model_path)
        cleaned = pd.read_csv(cleaned_path)
        test_raw = pd.read_csv(test_raw_path)
        X = test_raw.drop(columns=["target"])
        st.caption(f"Modèle chargé depuis `{model_path.name}` et pipeline depuis `preprocessing.joblib`.")
        st.markdown("### Scénarios de test")
        st.caption("Les cas bénin, malin et aléatoire viennent du fichier test tenu à l'écart de l'entraînement. Le cas flou est synthétique, construit entre un cas bénin et un cas malin : son label attendu est donc inconnu.")
        malignant_rows = test_raw[test_raw["target"] == 1]
        benign_rows = test_raw[test_raw["target"] == 0]
        scenario_buttons = st.columns(4)
        if scenario_buttons[0].button("Charger un cas malin", key="scenario_malignant"):
            st.session_state.prediction_scenario = "malignant"
        if scenario_buttons[1].button("Charger un cas bénin", key="scenario_benign"):
            st.session_state.prediction_scenario = "benign"
        if scenario_buttons[2].button("Charger un cas flou", key="scenario_ambiguous"):
            st.session_state.prediction_scenario = "ambiguous"
        if scenario_buttons[3].button(
            "Charger un cas aléatoire",
            key="scenario_random"
        ):
            st.session_state.prediction_scenario = "random"
            st.session_state.random_index = np.random.randint(
                0,
                len(test_raw)
            )

        scenario = st.session_state.get("prediction_scenario")

        expected_label: str | None = None
        scenario_note = "Saisie manuelle : aucun label attendu n'est fourni."

        if scenario == "malignant" and not malignant_rows.empty:
            scenario_frame = malignant_rows.iloc[[0]]
            expected_label = "Maligne"
            scenario_note = (
                "Cas réel du jeu test tenu à l'écart de l'entraînement : "
                "le label attendu est maligne."
            )

        elif scenario == "benign" and not benign_rows.empty:
            scenario_frame = benign_rows.iloc[[0]]
            expected_label = "Bénigne"
            scenario_note = (
                "Cas réel du jeu test tenu à l'écart de l'entraînement : "
                "le label attendu est bénigne."
            )

        elif scenario == "random":

            if "random_index" not in st.session_state:
                st.session_state.random_index = np.random.randint(
                    0,
                    len(test_raw)
                )

            scenario_frame = test_raw.iloc[
                [st.session_state.random_index]
            ]

            expected_label = (
                "Maligne"
                if scenario_frame.iloc[0]["target"] == 1
                else "Bénigne"
            )

            scenario_note = (
                "Échantillon aléatoire du jeu test : jamais utilisé pour "
                "ajuster les poids mais déjà utilisé pour l'évaluation."
            )

        elif (
            scenario == "ambiguous"
            and not malignant_rows.empty
            and not benign_rows.empty
        ):
            ambiguous_values = (
                malignant_rows[X.columns].iloc[0]
                + benign_rows[X.columns].iloc[0]
            ) / 2

            scenario_frame = pd.DataFrame(
                [ambiguous_values],
                columns=X.columns
            )

            scenario_note = (
                "Observation synthétique intermédiaire : "
                "aucun diagnostic réel n'est disponible."
            )

        else:
            scenario_frame = None

        # ====================================================
        # Injection des valeurs dans les widgets
        # ====================================================

        if scenario_frame is not None:

            st.info(scenario_note)

            loaded_values = (
                scenario_frame[X.columns]
                .iloc[0]
                .astype(float)
                .to_dict()
            )

            for feature, value in loaded_values.items():
                st.session_state[f"pred_{feature}"] = float(value)

        else:
            st.info(
                "Choisissez un scénario ou saisissez une observation manuellement."
            )

        # Initialisation des champs au premier chargement
        for feature in X.columns:

            key = f"pred_{feature}"

            if key not in st.session_state:
                st.session_state[key] = float(
                    X[feature].median()
                )

        # ====================================================
        # Formulaire
        # ====================================================

        with st.form("prediction"):

            for start in range(0, len(X.columns), 5):

                cols = st.columns(5)

                for feature, col in zip(
                    X.columns[start:start + 5],
                    cols
                ):

                    col.number_input(
                        feature,
                        key=f"pred_{feature}",
                        format="%.5f",
                    )

            predict = st.form_submit_button(
                "Évaluer l'observation",
                type="primary",
            )

        # ====================================================
        # Prédiction
        # ====================================================

        if predict:

            observation = pd.DataFrame(
                [
                    {
                        feature: st.session_state[f"pred_{feature}"]
                        for feature in X.columns
                    }
                ]
            )

            transformed = pipeline.transform(observation)

            label = int(model.predict(transformed)[0])

            probability = (
                float(model.predict_proba(transformed)[0, 1])
                if hasattr(model, "predict_proba")
                else None
            )

            st.metric(
                "Résultat",
                "Maligne" if label else "Bénigne"
            )

            if expected_label is not None:

                predicted_label = (
                    "Maligne"
                    if label
                    else "Bénigne"
                )

                if predicted_label == expected_label:
                    st.success(
                        f"Résultat correct par rapport au label connu : "
                        f"{expected_label}."
                    )
                else:
                    st.error(
                        f"Désaccord avec le label connu : "
                        f"attendu {expected_label}, "
                        f"prédit {predicted_label}."
                    )

            if probability is not None:
                st.progress(
                    probability,
                    text=f"Score indicatif de malignité : {probability:.1%}"
                )

            st.caption(
                "Cette sortie est pédagogique et ne constitue pas "
                "un diagnostic médical."
            )

else:
    st.subheader("Journal de traçabilité et artefacts")
    events = store.trace()
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("Aucune action n'est encore enregistrée dans cette expérience.")
    manifest = store.manifest()
    st.markdown("### Fichiers de l'expérience")
    st.dataframe(pd.DataFrame({"Fichier": manifest["files"], "SHA-256": [manifest["sha256"][name] for name in manifest["files"]]}), use_container_width=True)
    trace_path = store.trace_path
    if trace_path.exists():
        st.download_button("Télécharger le journal JSONL", trace_path.read_bytes(), trace_path.name, "application/jsonl")
    st.caption("Les timestamps UTC, paramètres, sources et empreintes SHA-256 permettent de reconstituer l'expérience.")
