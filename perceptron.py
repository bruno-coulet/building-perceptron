import random

# ======== Données factices générées de manière aléatoire - Début ================
try:
    input_size = int(input(f"{'='*40}\n\nChoisissez la taille de l'input (entrez un chiffre) : "))
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

print(f"\n{'='*40}\n\ninput générés :\n{inputs}\n")
print(f"weights générés :\n{weights}")

# ======== Données factices générées de manière aléatoire - FIN ================






# ========= Perceptron en POO ====================================

class Perceptron:
    def __init__(self, bias = 0.0, threshold=threshold):

        self.inputs = inputs
        self.bias = bias
        self.threshold = threshold
        self.bias = 0.0

    # Fonction d'activation
    def threshold_function(self, weighted_sum):
        print("\nCalcul de la fonction d'activation :\n")
        if weighted_sum >= self.threshold:
            print(f"Seuil : threshold = {self.threshold}")
            print(f"Somme pondérée : {weighted_sum}\n{weighted_sum} >= threshold\n")
            return 1
        else:
            print(f"Seuil : threshold = {self.threshold}")
            print(f"Somme pondérée : {weighted_sum}\n{weighted_sum} < threshold\n")
            return 0


    # Fonction de calcul du perceptron
    def calcul(self):

        # Initialise la somme pondérée à zéro
        weighted_sum=0

        print(f"\nLongueur du vecteur d'input : ({len(inputs)},)\n+ le biais :{self.bias}\n\n{'='*40}\n")

        # Calcul de la somme pondéré
        # --- multiplie chaque entré par le poids correspondant
        # --- additionne tous les produits obtenus
        for x,w in zip(inputs, weights):
            weighted_sum += x * w
        # --- ajoute le biais
        weighted_sum += self.bias

        print(f"Calcul de la somme pondérée (+ le biais) : {weighted_sum}\n\n{'='*40}")

        # appelle la fonction d'activation pour obtenir la sortie du perceptron
        return self.threshold_function(weighted_sum)





if __name__ == "__main__" :

    objet_perceptron = Perceptron()
    sortie = objet_perceptron.calcul()
    print(f"{'='*40}\n\nSortie du perceptron = {str(sortie)}\nEt BIM !\n\n{'='*40}")