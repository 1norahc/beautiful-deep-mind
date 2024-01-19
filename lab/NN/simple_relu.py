import numpy as np
#https://chat.openai.com/share/e4b8070c-8058-426d-ac70-6f31e60e4def
# Funkcja aktywacji - ReLU
def relu(x):
    return np.maximum(0, x)

# Inicjalizacja wag i biasów dla warstw
input_size = 3
hidden_size = 4

# Losowe inicjalizacje dla celów ilustracyjnych
input_layer = np.random.rand(input_size)
hidden_weights = np.random.rand(input_size, hidden_size)
hidden_bias = np.random.rand(hidden_size)

# Krok 1: Sumowanie ważonych sygnałów dla neuronu w warstwie ukrytej
hidden_sum = np.dot(input_layer, hidden_weights) + hidden_bias

# Krok 2: Funkcja aktywacji - ReLU
hidden_activation = relu(hidden_sum)

# Wypisanie wyników
print("Warstwa wejściowa:")
print(input_layer)

print("\nWagi warstwy ukrytej:")
print(hidden_weights)

print("\nSuma ważonych sygnałów dla neuronu w warstwie ukrytej przed aktywacją:")
print(hidden_sum)

print("\nAktywacja neuronu w warstwie ukrytej po funkcji aktywacji (ReLU):")
print(hidden_activation)

print("\n")
print(hidden_weights[0,0])