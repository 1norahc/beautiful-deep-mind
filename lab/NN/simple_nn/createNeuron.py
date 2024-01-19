import random
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = None

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights.")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return self._sigmoid(weighted_sum + self.bias)

    def _sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))

def create_neuron(num_inputs):
    return Neuron(num_inputs)

# Przykład użycia
num_inputs = 4

neuron = create_neuron(num_inputs)
print(f"Weights: {neuron.weights}\nBias: {neuron.bias}")
