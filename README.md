# Breast Cancer Lab

Application Streamlit pédagogique et reproductible consacrée à l'analyse du **Breast Cancer Wisconsin Diagnostic Dataset** et à l'implémentation d'un Perceptron binaire en programmation orientée objet.

## Fonctionnalités

- vue d'ensemble du dataset et de la cible;
- exploration des distributions, doublons, valeurs manquantes et corrélations;
- audit des variables constantes, atypiques, redondantes et informatives;
- séparation stratifiée train/test;
- feature engineering métier optionnel, ajusté dans le pipeline;
- comparaison des fonctions d'activation `step`, `sigmoid` et `tanh` pour le Perceptron;
- imputation médiane, standardisation et réduction PCA sans fuite de données;
- comparaison du Perceptron, d'une régression logistique, d'un SVM et d'une forêt aléatoire;
- validation croisée stratifiée et comparaison avec une régression logistique;
- sélection du meilleur modèle sur la validation croisée du train, jamais sur le test;
- accuracy, précision, rappel, F1-score, AUC ROC et matrices de confusion;
- visualisation de l'historique des runs et des artefacts;
- prédiction individuelle avec le modèle et le pipeline enregistrés.

## Structure

```text
building-perceptron/
├── app.py
├── data/
│   ├── raw/bcw_data.csv
│   └── bcw_description.md
├── src/building_perceptron/
│   ├── config.py
│   ├── data.py
│   ├── artifacts.py
│   ├── audit.py
│   ├── features.py
│   ├── pipeline.py
│   ├── perceptron.py
│   └── evaluation.py
├── tests/test_perceptron.py
└── pyproject.toml
```

## Installation et lancement

Python 3.12+ et `uv` sont recommandés :

```bash
uv sync --dev
uv run streamlit run app.py
```

L'application est accessible à l'adresse affichée par Streamlit, généralement `http://localhost:8501`.

## Données et méthode

Le fichier local contient 569 observations, 30 variables numériques, un identifiant et la cible `diagnosis` (`B` bénigne, `M` maligne). L'identifiant et la colonne vide héritée du fichier UCI sont exclus des variables explicatives.

Le workflow applique les étapes suivantes :

1. diagnostic de la source brute et rapport de qualité;
2. nettoyage explicite et sauvegarde d'une version nettoyée;
3. audit exploratoire des redondances, valeurs atypiques et variables informatives;
4. séparation stratifiée train/test, avec contrôle des proportions de classes;
5. feature engineering target-independent, imputation et standardisation ajustées sur le train;
6. PCA ajustée sur le train, lorsque l'option est activée;
7. validation croisée sur le train pour sélectionner le modèle;
8. évaluation finale sur le test conservé à part;
9. sélection persistée du modèle et prédiction avec le pipeline sauvegardé.

Chaque expérience est enregistrée dans `data/processed/runs/<run_id>/`, avec les CSV nettoyés et préparés, les modèles `joblib`, les paramètres, les métriques, les métadonnées PCA, un manifeste SHA-256 et un journal `trace.jsonl`.

Le rappel (`recall`) des cas malins est prioritaire dans l'interprétation, car un faux négatif peut retarder une prise en charge. Cette application ne constitue pas un outil de diagnostic médical.

### Interprétation de la convergence du Perceptron

Le théorème de convergence du Perceptron garantit une convergence avec zéro erreur
uniquement lorsque les classes sont linéairement séparables. Si `max_iter` est
atteint alors qu'il reste des erreurs, cela ne signifie pas que le modèle est
inutilisable : cela indique que la convergence parfaite n'est pas démontrée sur
ce train. Le Perceptron peut conserver de bonnes performances de généralisation
tout en faisant quelques erreurs d'entraînement. Augmenter le nombre d'itérations
peut être testé, mais ne garantit pas la convergence lorsque les données ne sont
pas séparables par une frontière linéaire.

## Tests et qualité

```bash
uv run pytest -q
uv run ruff check app.py src tests
```

## Limites et pistes d'amélioration

Le Perceptron est un modèle linéaire et sa sortie n'est pas une probabilité calibrée. Pour une étude plus avancée, on pourrait comparer des modèles non linéaires, optimiser le seuil selon le coût des erreurs, calibrer les probabilités, réaliser une validation externe, étudier l'explicabilité et vérifier la robustesse sur des données issues d'autres centres médicaux.
