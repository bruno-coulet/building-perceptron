import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.io as pio

from building_perceptron.model.perceptron import Perceptron


def get_separation_curve(perceptron : Perceptron, X, y, X_test, y_test):
    # Tracé des points
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Classe 0', marker='o')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Classe 1', marker='x')

    # Tracé de la droite de décision (w_0 * x_1 + w_1 * x_2 + b = 0  =>  x_2 = (-w_0 * x_1 - b) / w_1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100)

    if perceptron.w[1] != 0:
        yy = (-perceptron.w[0] * xx - perceptron.b) / perceptron.w[1]
        plt.plot(xx, yy, color='green', linewidth=2, label="Frontière de décision")

    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()
    plt.title("Perceptron - Frontière de décision sur données aléatoires")
    plt.grid(True)
    plt.show()


def get_learning_curve(model : Perceptron):

    # Sécurité pour l'affichage Console / Script
    pio.renderers.default = "browser"

    # Création du DataFrame pour Plotly
    epochs = list(range(1, len(model.loss_history) + 1))
    df_loss = pd.DataFrame({
        'Epoch': epochs,
        'Loss': model.loss_history
    })

    # Graphique
    fig = px.line(
        df_loss, 
        x='Epoch', 
        y='Loss',
        title="Courbe d'apprentissage",
        labels={'Epoch': "Époques (Epochs)", 'Loss': "Perte (Loss)"}
    )

    fig.update_traces(line_color='#2b5c8f', line_width=2.5)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()