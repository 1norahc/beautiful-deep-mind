import random
import pickle

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = None

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights.")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        self.output = self._sigmoid(weighted_sum + self.bias)
        return self.output

    def _sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden_neurons)]
        self.output_neuron = Neuron(num_hidden_neurons)

    def activate(self, inputs):
        hidden_activations = [neuron.activate(inputs) for neuron in self.hidden_layer]
        return self.output_neuron.activate(hidden_activations)

    def train(self, inputs, target_output, learning_rate=0.1):
        # Propagacja wsteczna
        output_error = target_output - self.output_neuron.output
        output_delta = output_error * self._sigmoid_derivative(self.output_neuron.output)

        for i, neuron in enumerate(self.hidden_layer):
            hidden_error = output_delta * self.output_neuron.weights[i]
            hidden_delta = hidden_error * self._sigmoid_derivative(neuron.output)

            # Aktualizacja wag
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * hidden_delta * inputs[j]

            # Aktualizacja biasu
            neuron.bias += learning_rate * hidden_delta

        for i in range(len(self.output_neuron.weights)):
            self.output_neuron.weights[i] += learning_rate * output_delta * self.hidden_layer[i].output

        # Aktualizacja biasu dla neuronu wyjściowego
        self.output_neuron.bias += learning_rate * output_delta

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

def save_network(network, filename):
    with open(filename, 'wb') as file:
        pickle.dump(network, file)

def load_network(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Przykład użycia do treningu
num_inputs = 4
num_hidden_neurons = 3
num_outputs = 1

network = NeuralNetwork(num_inputs, num_hidden_neurons, num_outputs)

# Przykładowe dane treningowe
training_data = [
    ([0, 0, 0, 0], 0),
    ([0, 0, 0, 1], 1),
    ([0, 0, 1, 0], 1),
    ([0, 1, 0, 0], 0),
    ([1, 0, 0, 0], 1),
]

# Trening sieci
epochs = 1000
for epoch in range(epochs):
    for inputs, target in training_data:
        network.activate(inputs)
        network.train(inputs, target)

# Zapisywanie wytrenowanej sieci
save_network(network, 'trained_network.pkl')

# Wczytywanie sieci z pliku
loaded_network = load_network('trained_network.pkl')

# Przykład użycia do aktywacji
sample_input = [1, 0, 1, 0]
output = loaded_network.activate(sample_input)
print(f"Network Output for {sample_input}: {output}")
