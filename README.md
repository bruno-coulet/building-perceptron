# building-perceptron

Projet pédagogique d'implémentation d'un perceptron pour la classification de tumeurs du sein.

## Contexte du projet

Ce projet s'inscrit dans le cadre de l'apprentissage de l'intelligence artificielle et du machine learning. L'objectif est de :

1. **Comprendre le fonctionnement fondamental d'un perceptron** à travers une implémentation simple et pédagogique
2. **Développer un perceptron compatible avec Scikit-learn** pour résoudre un problème de classification médicale
3. **Réaliser une analyse exploratoire complète** (EDA) des données 
4. **Évaluer et comparer les performances** du modèle personnalisé


## Données

### Source

Le projet utilise le **Breast Cancer Wisconsin (Diagnostic) Dataset** pour classifier des tumeurs comme bénignes ou malignes.

**Dataset** (`raw_data/bcw_data.csv`)
https://drive.google.com/file/d/1itXdRo4WJuhqCjtVX4WGvT327WWp4LB7/view

### Description
- **Nombre de caractéristiques** : 30 features numériques calculées à partir d'images de masses cellulaires
- **Variable cible** : `diagnosis` (M = Maligne, B = Bénigne)
- **Types de features** :
  - Mesures moyennes (_mean) : rayon, texture, périmètre, aire, etc.
  - Erreurs standard (_se) : variabilité des mesures
  - Valeurs maximales (_worst) : cas les plus extrêmes

### Caractéristiques principales
```
- radius_mean : rayon moyen des cellules
- texture_mean : écart-type des valeurs de niveau de gris
- perimeter_mean : périmètre moyen
- area_mean : aire moyenne
- smoothness_mean : variation locale des longueurs de rayon
- compactness_mean : (périmètre² / aire) - 1.0
- concavity_mean : sévérité des portions concaves du contour
- concave points_mean : nombre de portions concaves du contour
- symmetry_mean : symétrie
- fractal_dimension_mean : "approximation de côte" - 1
```

## Structure du projet

```
├── app.py
├── data
│   ├── processed_data
│   │   ├── bcw_test.csv
│   │   └── bcw_train.csv
│   └── raw_data
│       └── bcw_data.csv
├── documentation
│   └── Questions.pptx
├── notebooks
│   ├── archives
│   │   ├── create-db-incendies.ipynb
│   │   ├── eda_archive.ipynb
│   │   └── eda_utils.py
│   ├── eda.ipynb
│   └── training.ipynb
├── pyproject.toml
├── README.md
├── scripts
│   └── eda
│       └── run_eda.py
├── src
│   └── building_perceptron
│       ├── data
│       │   ├── data_utils.py
│       │   └── __init__.py
│       ├── __init__.py
│       ├── model
│       │   ├── __init__.py
│       │   ├── logistic_regression.py
│       │   └── perceptron.py
│       ├── services
│       │   └── __init__.py
│       └── views
│           └── __init__.py
└── uv.lock
```

## Analyse

### Processus d'analyse exploratoire (EDA)

Le notebook `eda.ipynb` réalise une analyse complète des données :
1. Chargement et inspection initiale du dataset
    - Détection et traitement de l'encodage, des doublons, des colonnes vides
    - Contrôle et harmonisation des types de données
2. Statistiques descriptives
    - Analyse de la distribution des variables
    - Etude de la bimodalité de certaines distributions
    - Analyse de la corrélation entre features, et des features avec la cible
    - Etude du chevauchement entre features
    - Tri des variables à l'information redondante
3. Standardisation / Normalisation
    - Choix du Robust Scaler
0. Analyse en Composantes Principales (PCA)


## Training

### Training et Evaluation

Le notebook `training.ipynb` réalise l'entraînement sur le dataset nettoyé en utilisant le Perceptron et la régression logistique.
1. Test du Perceptron
    - Test de la porte logique AND
    - Test de valeurs aléatoires généré avec make_classification
2. Préparation
    - Séparation des caractéristiques (X) et de la cible (y)
    - Fonctions : Metrics, Confusion Matrix, Learning Curve
    - GridSearchCV : 
        - Perceptron : get_evaluation
        - Régression Logistique : get_evaluation_lr, get_evaluation_lr_seuils
    - ROC/AUC Curve
3. Training / Evaluation
    - Training du Perceptron 
        - Matrice de confusion / Courbe d'apprentissage
        - Utilisation de GridSearchCV
        - Matrice de score
    - Training de la Régression Logistique
        - Matrice de confusion / Courbe d'apprentissage / Score ROC/AUC
        - Utilisation de GridSearchCV : Seuil de la Régression Logistique (fixe) / Seuil de la Régression Logistique (Liste)
        - Score ROC/AUC

### Métriques de performance

**⚠️ Importance du contexte médical :**

En santé, un **Faux Négatif** (dire qu'une tumeur est bénigne alors qu'elle est maligne) est **bien plus grave** qu'un **Faux Positif** (faire des examens complémentaires inutiles).

**Focus prioritaire sur le Recall (Sensibilité)** pour minimiser les faux négatifs et ne pas manquer de cas malins.

**Métriques utilisées** :
- **Recall (Sensibilité)** : Proportion de vrais malins correctement détectés
- **Precision** : Proportion de diagnostics malins corrects
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **Accuracy** : Taux de bonnes classifications global
- **Matrice de confusion** : Visualisation détaillée des erreurs


## Installation et utilisation

### 1. Cloner le projet
```bash
git clone <url-du-repo>
cd building-perceptron
```

### 2. Créer l'environnement virtuel avec uv
```bash
uv venv
source .venv/Scripts/activate  # Git Bash
```

### 3. Installer les dépendances
```bash
uv pip install -e .
```

### 4. Tester le perceptron
```bash
uv run scripts/training/run_perceptron.py --num x
1 : Porte logique AND
2 : Test de valeurs aléatoires 
```


## Exemple d'utilisation du Perceptron personnalisé

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import plotly.express as px
import plotly.io as pio

from building_perceptron.model.perceptron import Perceptron


def get_evaluation(algorithme, param_grid, X_train, y_train, X_test, y_test):

    # Recherche par grille avec 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=algorithme,
        param_grid=param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )

    # Entraînement sur les données
    grid_search.fit(X_train, y_train)

    # Extraction du meilleur modèle et prédictions sur le test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calcul des métriques d'évaluation
    test_accuracy = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)

    # Affichage synthétique dans la console
    print("=== Résultats du Tuning ===")
    print("Meilleurs paramètres :", grid_search.best_params_)
    print(f"Meilleur Rappel CV (Recall) : {grid_search.best_score_:.4f}")
    print(f"Précision sur Jeu Test  : {test_accuracy:.4f}")

    # Retour d'un dictionnaire complet d'artefacts
    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "cv_results": pd.DataFrame(grid_search.cv_results_)
    }


# Séparation des caractéristiques (X) et de la cible (y)
target_col = "diagnosis"

X_train = df_train.drop(columns=[target_col]).values
y_train = df_train[target_col].values

X_test = df_test.drop(columns=[target_col]).values
y_test = df_test[target_col].values


# Algorithme
algorithme = Perceptron()

# Grille d'hyperparamètres à tester
param_grid = {
    'lr': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'epochs': [50, 100, 200, 500, 1000],
    'method': ['classique', 'strict']
}

results = get_evaluation(algorithme, param_grid, X_train, y_train, X_test, y_test)
```

#### Retour d'un dictionnaire complet d'artefacts
```python
    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "cv_results": pd.DataFrame(grid_search.cv_results_)
    }
```

## Etape suivante

Développer l'interface graphique (streamlit)

**Auteur** : Philippe Laurens   
**Date** : Octobre 2026  
**Version** : 1.0.0  
**Contexte** : Projet pédagogique - Formation IA en alternance (Marseille)  
**Remerciements** : Bruno et Pierre