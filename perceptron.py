import random
# import numpy as np


LEARNING_RATE = 0.1

# ======== Début - Données factices générées de manière aléatoire ================
try:
    INPUT_SIZE = int(input("\nChoisissez la taille de l'input (entrez un chiffre) : "))
except ValueError:
    print("\nVeuillez entrer un ENTIER, sinon cela ne fonctionnera pas.\n")
    exit()

try:
    THRESHOLD = float(input("Choisissez le seuil (entrez un chiffre) : "))
except ValueError:
    print("Veuillez entrer un NOMBRE, sinon cela ne fonctionnera pas.\n")
    exit()

inputs = []
weights = []

for _ in range(INPUT_SIZE):
    # génère un entier, bornes inclusives [0, 10]
    inputs.append(random.randint(0, 10))
    # génère un float,distribution uniforme sur l'intervalle [0.0, 1.0[
    weights.append(random.random())

print(f"\ninput générés :\n{inputs}\n")
print(f"weights générés :\n{weights}\n")

# ======== FIN - Données factices générées de manière aléatoire ================






# ========= Perceptron en POO ====================================

class Perceptron:

    def __init__(self, input_size=INPUT_SIZE, weights=weights, threshold=THRESHOLD, learning_rate=LEARNING_RATE):

        self.input_size = input_size
        self.weights = weight
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.bias = 0.0