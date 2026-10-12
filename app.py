from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Import de vos propres modules (selon votre structure src/)
# from building_perceptron.data.data_utils import (
#     load_data_function_if_needed,
# ) 
from building_perceptron.model.logistic_regression import LogisticRegression
from building_perceptron.model.perceptron import Perceptron

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Perceptron vs Régression Logistique - Wisconsin Cancer",
    page_icon="🧬",
    layout="wide",
)

st.title(
    "🧬 Analyse du Cancer du Sein (Wisconsin) : Perceptron vs Régression Logistique"
)


# 1. Chargement des données processées
@st.cache_data
def load_datasets():
    train_path = Path("data/processed_data/bcw_train.csv")
    test_path = Path("data/processed_data/bcw_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


train_data, test_data = load_datasets()

# 2. Navigation par onglets
tab_accueil, tab_exploration, tab_perceptron, tab_comparison = st.tabs(
    [
        "🏠 Accueil & Contexte",
        "📊 Exploration & Données",
        "🧠 Le Perceptron",
        "⚖️ Comparaison : Perceptron vs Régression Logistique",
    ]
)

with tab_accueil:
    st.header("Contexte du Projet")
    st.write(
        """
    Cette application interactive permet d'explorer le dataset du cancer du sein du Wisconsin 
    et d'étudier la séparation linéaire des classes à l'aide d'un **Perceptron** fait maison, 
    comparé aux performances d'une **Régression Logistique**.
    """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Échantillons d'entraînement", value=len(train_data)
        )
    with col2:
        st.metric(label="Échantillons de test", value=len(test_data))

with tab_exploration:
    st.header("Aperçu des données")
    st.write(
        "Voici un aperçu des données d'entraînement prêtes pour la modélisation :"
    )
    st.dataframe(train_data.head())

with tab_perceptron:
    st.header("Entraînement et Évaluation du Perceptron")

    # Paramètres du Perceptron dans la barre latérale
    st.sidebar.header("Paramètres du Perceptron")
    learning_rate = st.sidebar.slider(
        "Taux d'apprentissage (learning rate)", 0.0001, 0.1, 0.01, format="%.4f"
    )
    n_epochs = st.sidebar.slider("Nombre d'époques", 10, 500, 100, step=10)

    if st.button("Lancer l'entraînement du Perceptron"):
        # Exemple d'instanciation de votre classe Perceptron
        # (Adaptez les arguments selon la signature exacte de votre classe dans perceptron.py)
        perceptron_model = Perceptron(
            lr=learning_rate, epochs=n_epochs
        )

        # Simulation visuelle ou affichage des résultats
        st.success(
            f"Perceptron entraîné avec succès (lr={learning_rate}, époques={n_epochs}) !"
        )
        # TODO: Ajouter l'évaluation sur le jeu de test et l'affichage de la frontière de décision

with tab_comparison:
    st.header("Comparaison : Perceptron vs Régression Logistique")
    st.write(
        """
    Le **Perceptron** cherche un hyperplan séparateur sans se soucier de la probabilité, 
    ce qui le rend instable si les données ne sont pas linéairement séparables. 
    À l'inverse, la **Régression Logistique** optimise une fonction de coût logistique (sigmoïde) 
    permettant d'obtenir une estimation probabiliste et une meilleure robustesse.
    """
    )

    if st.button("Lancer la comparaison des deux modèles"):
        # Instanciation et comparaison des deux approches issues de src/building_perceptron/model/
        st.info(
            "Entraînement en cours de la Régression Logistique et du Perceptron..."
        )

        # Affichage d'un tableau comparatif fictif ou calculé
        comparison_results = pd.DataFrame(
            {
                "Modèle": ["Perceptron", "Régression Logistique"],
                "Précision (Test)": ["À calculer", "À calculer"],
                "Interprétabilité": ["Élevée (poids bruts)", "Élevée (Odds Ratios)"],
            }
        )
        st.table(comparison_results)