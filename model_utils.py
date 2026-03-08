import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns, matplotlib.pyplot as plt


def eval_classification(algo, param_grid, X_train, y_train, X_test, y_test, 
                        search_type='grid', 
                        scoring='accuracy',
                        cv=5):
    """
    Entraîne un modèle de classification avec GridSearchCV ou RandomizedSearchCV
    prédit les valeurs de test et  calcule les métriques de performance
    : parametre algo : l'instance de l'algo à entraîner
    : parametre param_grid : dictionnaire des paramètres à tester dans GridSearchCV ou RandomizedSearchCV
    ( mettre 'None' si vide )
    : parametre search_type : type recherche ('grid' pour GridSearchCV, 'random' pour RandomizedSearchCV)
    : parametre scoring : métrique d'évaluation
    : parametre cv : nombre de folds pour la validation croisée
    : return : un dict contenant les meilleures paramètres, 
    accuracy, precision, recall, f1, confusion_matrix et classification report
    """
    # Si y_train est déjà numérique, on définit les noms de classes manuellement
    if np.issubdtype(y_train.dtype, np.number):
        y_train_encoded = y_train
        y_test_encoded = y_test
        # On crée des noms de catégories sous forme de texte pour le rapport
        unique_categories = [str(c) for c in np.unique(y_train)]
    else:
        # Encodage des étiquettes
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Récupération des catégories originales
        unique_categories = label_encoder.classes_ 

    # Entraine sans optimisation d'hyperparamètres si param_grid est vide
    if param_grid is None:
        algo.fit(X_train, y_train_encoded)
        best_model = algo
        best_params = algo.get_params()
        cv_resuls = cross_val_score(
            best_model, 
            X_train, 
            y_train_encoded, 
            cv=cv, 
            scoring=scoring
            )
    else:
        # Choisir le type de recherche d'hyperparamètres
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=algo,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                estimator=algo,
                param_distributions=param_grid,
                # n_iter=10,  # Nombre d'itérations pour RandomizedSearchCV
                cv=cv,
                scoring=scoring
            )
        else:
            raise ValueError("search_type doit être 'grid' ou 'random'")
        
        # Entraînement et optimisation
        search.fit(X_train, y_train_encoded)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_

    # Prédiction avec le meilleur modèle
    y_pred = best_model.predict(X_test)

    # Calcul des indicateurs (métriques de performance)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    # classification_report fournit la précision, le rappel et le F1-score pour chaque classe
    # zero_division=0 affiche 0.0 au lieu d'un avertissement quand une classe n'a pas de prédictions
    class_report = classification_report(y_test_encoded, y_pred, target_names=unique_categories, zero_division=0)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    # Affichage des résultats
    print()
    print(f"Modèle : {algo.__class__.__name__}")
    print(f"Meilleurs paramètres : {best_params}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"\n Rapport de classification :\n")
    print(class_report)

    # Affichage de la matrice de confusion avec Seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=unique_categories, yticklabels=unique_categories)
    plt.xlabel('Etiquettes prédites')
    plt.ylabel('Etiquettes réelles')
    plt.title(f'Matrice de confusion pour {algo.__class__.__name__}')
    plt.show()

    return{
        "best_params": best_params,
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "cv_results": cv_results,
        "best_model": best_model
    }


# ======== Perceptron avec learning rate et itérations - Début ================
# La fonction np.where agit exactement comme ta threshold_function


from sklearn.base import BaseEstimator, ClassifierMixin
# On hérite de BaseEstimator (pour GridSearchCV) et ClassifierMixin (pour le score)
class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, learning_rate=0.01, n_iterations=100):
        # IMPORTANT : Les noms des arguments doivent être identiques 
        # aux noms des attributs self.xxx
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        
        for _ in range(self.n_iterations):
            for i in range(len(y)):
                prediction = self.predict(X[i].reshape(1, -1))[0]
                error = y[i] - prediction
                
                if error != 0:
                    # Utilisation de self.learning_rate (nom exact du init)
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
        return self # Toujours retourner self dans fit()

    def predict(self, X):
        X = np.array(X)
        weighted_sum = np.dot(X, self.weights) + self.bias
        return np.where(weighted_sum >= self.threshold, 1, 0)