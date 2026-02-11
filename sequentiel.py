# ========= Perceptron en programmation séquentielle ========================

x_input = [2, 5, 1, 8]
w_weights = [.4, .3, .2, .1]
threshold = .5

# Fonction d'activation : step function
def threshold_step(weighted_sum):
    print("Calcul de la fonction d'activation :")
    if weighted_sum >= threshold:
        print(f"Threshold : {threshold}\nSomme pondérée : {weighted_sum}\nSomme pondérée >= threshold\n")
        return 1
    else:
        print(f"Threshold : {threshold}\nSomme pondérée : {weighted_sum}\nSomme pondérée < threshold\n")
        return 0


# Fonction de calcul du perceptron (appelle la fonction d'activation)
def perceptron():

    weighted_sum=0
    print(f"\nVecteur d'input : ({len(x_input)},)\n")
    print("Calcul de la somme pondérée :")

    for x,w in zip(x_input, w_weights):
        weighted_sum += x * w
        print(f"Somme partielle index {x_input.index(x)} : {x} * {w} = {str(weighted_sum)}")
    print()

    return threshold_step(weighted_sum)


output = perceptron()
print(f"Output = {str(output)}\n")