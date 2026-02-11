import random

# ======== Début - Données factices générées de manière aléatoire ================
try:
    input_size = int(input("\nChoisissez la taille de l'input (entrez un chiffre) : "))
except ValueError:
    print("\nVeuillez entrer un ENTIER, sinon cela ne fonctionnera pas.\n")
    exit()

try:
    threshold = float(input("Choisissez le seuil (entrez un décimal) : "))
except ValueError:
    print("Veuillez entrer un NOMBRE, sinon cela ne fonctionnera pas.\n")
    exit()

inputs = []
weights = []

for _ in range(input_size):
    # génère un entier, bornes inclusives [0, 10]
    inputs.append(random.randint(0, 10))
    # génère un float,distribution uniforme sur l'intervalle [0.0, 1.0[
    weights.append(random.random())

print(f"\ninput générés :\n{inputs}\n")
print(f"weights générés :\n{weights}\n")

# ======== FIN - Données factices générées de manière aléatoire ================






# ========= Perceptron en POO ====================================
class Perceptron:

    # def __init__(self, input_size=input_size, inputs=inputs, weights=weights, threshold=threshold, learning_rate=LEARNING_RATE):

    #     self.input_size = input_size
    #     self.inputs = inputs
    #     self.weights = weights
    #     self.threshold = threshold
    #     self.learning_rate = learning_rate
    #     self.bias = 0.0
    def __init__(self, threshold=threshold):
        self.inputs = inputs
        self.threshold = threshold



    # Fonction d'activation
    def threshold_function(self, weighted_sum):
        print(f"Seuil : {self.threshold}\n")
        print("Calcul de la fonction d'activation :")
        if weighted_sum >= self.threshold:
            print(f"Somme pondérée : {weighted_sum}\n{weighted_sum} >= threshold\n")
            return 1
        else:
            print(f"Somme pondérée : {weighted_sum}\n{weighted_sum} < threshold\n")
            return 0


    # Fonction de calcul du perceptron (appelle la fonction d'activation)
    def perceptron(self):

        weighted_sum=0
        print(f"\nLongueur du vecteur d'input : ({len(inputs)},)\n")

        print("Calcul de la somme pondérée :")
        for x,w in zip(inputs, weights):
            weighted_sum += x * w
        print(f"Somme pondérée = {weighted_sum}\n")

        return self.threshold_function(weighted_sum)





if __name__ == "__main__" :

    mon_premier_truc_bidule = Perceptron()
    sortie = mon_premier_truc_bidule.perceptron()
    print(f"Sortie du perceptron = {str(sortie)}\n")