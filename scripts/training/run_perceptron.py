import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.io as pio

from sklearn.base import BaseEstimator, ClassifierMixin
from building_perceptron.model.perceptron import Perceptron

from perceptron_utils import get_separation_curve, get_learning_curve


def get_decision_curve(model : Perceptron):

    # Données
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Poids issus de ton modèle
    w1, w2 = model.w
    b = model.b

    # Points de la droite de décision
    # La frontière de décision est l'ensemble des points où z = 0 :
    # w1·x1 + w2·x2 + b = 0
    # avec : x2 = -(w1·x1 + b) / w2

    # abscisses des deux points
    x1_vals = np.array([-0.5, 1.5])
    # ordonnées des deux points
    x2_vals = -(w1 / w2) * x1_vals - (b / w2)

    plt.figure(figsize=(6, 5))

    # Affichage des points (Classe 0 en bleu, Classe 1 en rouge)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Classe 0', s=100)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Classe 1', s=100)

    # Affichage de la droite de décision
    plt.plot(x1_vals, x2_vals, color='black', linestyle='--', label='Frontière de décision')

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Frontière de décision du Perceptron (Porte AND)')
    plt.legend()
    plt.grid(True)
    plt.show()


### Porte logique AND ###
def porte_and():

    X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    # Instanciation du Perceptron
    model = Perceptron(lr=0.1, epochs=10, method='strict')

    # Entrainement
    model.fit(X, y)

    # Prédictions
    predictions = model.predict(X)

    print("\n--- Résultats ---")
    print("Prédictions :", predictions)
    print("Vrais labels :", y)
    print("État du modèle :", model)

    get_decision_curve(model)


### Test de valeurs aléatoires ###
def get_classifications():

    X, y = make_classification(
        # n_samples=300 : Définit le nombre total d'exemples (ou de lignes) à générer dans le jeu de données synthétique. Ici, la fonction crée 300 observations au total.
        n_samples=300, 
        # n_features=2 : Représente le nombre total de caractéristiques (colonnes ou dimensions) pour chaque observation. 
        #   Fixer cette valeur à 2 permet de représenter directement les données sur un graphique en deux dimensions.
        n_features=2, 
        # n_redundant=0 : Indique le nombre de caractéristiques redondantes. Ce sont des colonnes construites par combinaison linéaire des caractéristiques informatives. 
        #   La valeur 0 signifie qu'aucune colonne n'est un doublon ou une redondance mathématique des autres.
        n_redundant=0, 
        # n_informative=2 : Déclare le nombre de caractéristiques utiles pour prédire la classe cible. 
        #   Ici, les 2 caractéristiques créées contiennent de l'information pertinente pour séparer les classes (aucune d'entre elles n'est du bruit pur).
        n_informative=2,
        # n_clusters_per_class=1 : Détermine le nombre de groupes (clusters) distincts par classe. A
        #   Avec la valeur 1, chaque classe forme un unique nuage de points compact, évitant d'avoir des sous-groupes éparpillés un peu partout.
        n_clusters_per_class=1, 
        # Contrôle l'écart (la distance) entre les classes. 
        #   Plus cette valeur est grande, plus les nuages de points sont éloignés et faciles à séparer par un modèle. Plus elle est petite, plus les classes se chevauchent.
        class_sep=2.0, 
        # random_state=42 : Fixe la graine aléatoire (seed). Elle garantit la reproductibilité : 
        #   à chaque fois que vous relancerez votre script, le générateur aléatoire produira exactement le même jeu de données.
        random_state=42
    )

    # Séparation en train / test ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du Perceptron
    perceptron = Perceptron(lr=0.001, epochs=500, method='strict')
    perceptron.fit(X_train, y_train)

    # Évaluation : grâce à ClassifierMixin, la méthode .score() est disponible automatiquement
    accuracy = perceptron.score(X_test, y_test)
    print(f"Précision sur le jeu de test : {accuracy * 100:.2f}%")

    get_separation_curve(perceptron, X, y, X_test, y_test)

    get_learning_curve(perceptron)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, nargs='+', required=True)
    args = parser.parse_args()

    if args.num == [1]:
        porte_and()
    elif args.num == [2]:
        get_classifications()