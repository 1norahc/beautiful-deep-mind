import random

class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = random.uniform(-1, 1) # bias - generowamy losowo dla pojedynczego neuronu

    def initialize_weights(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]

    def activate(self, inputs):
        if not self.weights:
            raise ValueError("Neuron weights not initialized.")
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights.")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return self._sigmoid(weighted_sum + self.bias)

    def _sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))


def create_neurons(num_neurons, num_inputs):
    neurons = [Neuron() for _ in range(num_neurons)]
    for neuron in neurons:
        neuron.initialize_weights(num_inputs)
    return neurons

# Przykład użycia
num_neurons = 1
num_inputs = 4

neurons = create_neurons(num_neurons, num_inputs)
for i, neuron in enumerate(neurons, 1):
    print(f"Neuron {i}: Weights: {neuron.weights}, Bias: {neuron.bias}")
