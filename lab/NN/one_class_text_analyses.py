import numpy as np

# Funkcja aktywacji - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji
def sigmoid_derivative(x):
    return x * (1 - x)

# Przygotowanie danych treningowych
# Zakładamy, że mamy pewien zestaw tekstów i odpowiadające im etykiety (klasy)
# Możemy zakodować każde słowo jako unikalną liczbę, a cały tekst jako sekwencję tych liczb

# Załóżmy, że mamy tylko dwa słowa: "hello" i "world"
words = ["hello", "world"]

# Zakodujmy każde słowo jako unikalną liczbę
word_to_index = {word: i for i, word in enumerate(words)}

print(word_to_index)

# Przykładowy tekst do analizy
input_text = "hello world"

# Zamieńmy tekst na reprezentację liczbową
input_sequence = [word_to_index[word] for word in input_text.split()]

# Przygotujmy wejście dla sieci neuronowej - zakodowana sekwencja tekstu
X = np.array(input_sequence).reshape(1, len(input_sequence))

# Oczekiwane wyjście - zakładamy jedną klasę wyjściową (np. 0 lub 1)
# Możemy dostosować to do rzeczywistych etykiet, jeśli mamy zestaw danych treningowych
y = np.array([[1]])

# Parametry sieci neuronowej
input_size = len(words)
hidden_size = 10
output_size = 1

# Inicjalizacja wag
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Uczenie sieci neuronowej (propagacja wsteczna)
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # Propagacja wsteczna
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)
    
    # Obliczanie błędu
    error = y - predicted_output
    
    # Propagacja wsteczna - aktualizacja wag
    output_delta = error * sigmoid_derivative(predicted_output)
    
    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate

# Po zakończeniu uczenia możemy użyć naszej sieci do analizy tekstu
# Przeprowadźmy te same kroki, co wcześniej, aby zamienić tekst na reprezentację liczbową
input_sequence = [word_to_index[word] for word in input_text.split()]
X = np.array(input_sequence).reshape(1, len(input_sequence))

# Propagacja w przód
hidden_layer_input = np.dot(X, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

# Wynik analizy tekstu
print("Wynik analizy tekstu:", predicted_output)
