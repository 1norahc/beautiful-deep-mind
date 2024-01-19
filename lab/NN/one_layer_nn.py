import numpy as np

# Funkcja aktywacji - przykładowa funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji dla wstecznej propagacji błędu
def sigmoid_derivative(x):
    return x * (1 - x)

# Dane treningowe
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Oczekiwane wyniki (bramka logiczna XOR)
targets = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Inicjalizacja wag
np.random.seed(42)
weights = np.random.uniform(size=(2, 1))

# Współczynnik uczenia
learning_rate = 0.1

# Liczba epok
epochs = 10000

# Uczenie sieci
for epoch in range(epochs):
    # Przekazywanie danych przez sieć
    layer_input = np.dot(inputs, weights)
    layer_output = sigmoid(layer_input)

    # Obliczanie błędu
    error = targets - layer_output

    # Aktualizacja wag na podstawie błędu i pochodnej funkcji aktywacji
    weights += learning_rate * np.dot(inputs.T, error * sigmoid_derivative(layer_output))

# Testowanie na nowych danych
new_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

predicted_output = sigmoid(np.dot(new_data, weights))
print("Wagi po uczeniu:")
print(weights)
print("\nPrzewidywane wyniki:")
print(predicted_output)
