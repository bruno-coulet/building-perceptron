import random
# import numpy as np



# ======== Données factices générées de manière aléatoire ================
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

LEARNING_RATE = 0.1


x_input = []
w_weights = []

for _ in range(INPUT_SIZE):
    # génère un entier, bornes inclusives [0, 10]
    x_input.append(random.randint(0, 10))
    # génère un float,distribution uniforme sur l'intervalle [0.0, 1.0[
    w_weights.append(random.random())

print(f"\ninput générés :\n{x_input}\n")
print(f"weights générés :\n{w_weights}\n")

# ========= Perceptron en POO ====================================

class Perceptron:

    def __init__(self, input_size, weight, threshold, learning_rate):

        self.input_size = input_size
        self.weights = weight
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.bias = 0.0