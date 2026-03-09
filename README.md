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

## Analyse

### Processus d'analyse exploratoire (EDA)

Le notebook `eda.ipynb` réalise une analyse complète des données :
- Chargement et inspection initiale du dataset
- Détection et traitement des valeurs manquantes
- Identification et gestion des valeurs aberrantes
- Analyse de la distribution des variables
- Visualisation des corrélations entre features
- Équilibrage des classes (Maligne vs Bénigne)

### Module utilitaire `eda_utils.py`

Boîte à outils complète pour l'analyse exploratoire contenant :

**Détection de problèmes** :
- `empty_columns()` : colonnes complètement vides
- `unique_value_columns()` : colonnes avec une seule modalité
- `high_na_columns()` : colonnes avec trop de valeurs manquantes
- `high_cardinality_columns()` : colonnes catégorielles avec trop de modalités
- `duplicate_rows()` : détection des doublons

**Nettoyage** :
- `drop_columns()` : suppression de colonnes
- Gestion des valeurs manquantes
- Encodage des variables catégorielles

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

## Structure du projet

```
building-perceptron/
├── raw_data/
│   └── bcw_data.csv              # Dataset Breast Cancer Wisconsin
├── perceptron.py                 # Implémentation pédagogique simple
├── model_utils.py                # Perceptron sklearn + fonctions d'évaluation
├── eda_utils.py                  # Utilitaires pour l'EDA
├── eda.ipynb                     # Notebook d'analyse exploratoire
├── 2_analysis.ipynb              # Notebook d'analyse complémentaire
├── pyproject.toml                # Configuration du projet (uv)
└── README.md                     # Documentation
```

## Outils utilisés

### Environnement
- **Python** : 3.12+
- **Gestionnaire de packages** : `uv` (environnement virtuel)
- **OS** : Windows 11 (PowerShell / Git Bash)

### Bibliothèques principales
- **pandas** (≥3.0.0) : manipulation de données
- **numpy** : calculs numériques
- **scikit-learn** (≥1.8.0) : algorithmes ML et métriques
- **matplotlib** (≥3.10.8) : visualisation de base
- **seaborn** (≥0.13.2) : visualisation statistique
- **plotly** (≥6.6.0) : visualisations interactives

### Modules personnalisés

#### `perceptron.py` - Implémentation pédagogique
- Classe `DemoPerceptron` avec génération de données aléatoires
- Démonstration du calcul de la somme pondérée
- Fonction d'activation seuil
- Visualisation pas à pas du fonctionnement

#### `model_utils.py` - Outils de modélisation
- **Classe `Perceptron`** : 
  - Compatible avec l'API Scikit-learn (hérite de `BaseEstimator` et `ClassifierMixin`)
  - Paramètres ajustables : `threshold`, `learning_rate`, `n_iterations`
  - Méthodes `fit()`, `predict()` standard
  - Utilisable avec `GridSearchCV` et `RandomizedSearchCV`

- **Fonction `eval_classification()`** :
  - Optimisation d'hyperparamètres (Grid ou Random Search)
  - Validation croisée configurable
  - Calcul automatique des métriques (accuracy, precision, recall, F1)
  - Affichage de la matrice de confusion
  - Rapport de classification détaillé

#### `eda_utils.py` - Utilitaires d'analyse
- Fonctions de détection de problèmes dans les données
- Outils de nettoyage et préparation
- Compatible avec les pipelines Scikit-learn

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
# ou
.venv\Scripts\activate         # PowerShell
```

### 3. Installer les dépendances
```bash
uv pip install -e .
```

### 4. Lancer l'analyse
```bash
# Perceptron pédagogique
uv run perceptron.py

# Notebook d'analyse exploratoire
jupyter notebook eda.ipynb
```

## Exemple d'utilisation du Perceptron personnalisé

```python
from model_utils import Perceptron, eval_classification
from sklearn.model_selection import train_test_split
import pandas as pd

# Chargement des données
df = pd.read_csv('raw_data/bcw_data.csv')
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Définition du modèle et de la grille de paramètres
perceptron = Perceptron()
param_grid = {
    'threshold': [0.3, 0.5, 0.7],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_iterations': [50, 100, 200]
}

# Évaluation avec optimisation
results = eval_classification(
    algo=perceptron,
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    search_type='grid',
    scoring='recall',  # Focus sur le Recall pour le contexte médical
    cv=5
)
```

## Conclusion

Ce projet permet de :
1. ✅ **Comprendre** les fondamentaux du perceptron et de l'apprentissage supervisé
2. ✅ **Implémenter** un algorithme ML from scratch compatible avec Scikit-learn
3. ✅ **Maîtriser** l'analyse exploratoire des données (EDA)
4. ✅ **Évaluer** correctement un modèle avec les métriques adaptées au contexte
5. ✅ **Développer** des outils réutilisables pour de futurs projets

Le perceptron, bien que simpliste, illustre parfaitement les concepts clés du machine learning : fonction de coût, descente de gradient, optimisation d'hyperparamètres.

## Bibliographie

- **Dataset** : [UCI Machine Learning Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Scikit-learn Documentation** : [https://scikit-learn.org/](https://scikit-learn.org/)
- **Rosenblatt, F. (1958)** : "The perceptron: A probabilistic model for information storage and organization in the brain"
- **Demangel, Eric** : "Maîtrisez la data science avec Python" (référence pour eda_utils.py)

---

**Auteur** : Bruno Coulet  
**Date** : Février 2026  
**Version** : 1.0.0  
**Contexte** : Projet pédagogique - Formation IA en alternance (Marseille)
