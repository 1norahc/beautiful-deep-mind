import random
import matplotlib.pyplot as plt
import networkx as nx

class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = random.uniform(-1, 1)

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


def draw_neural_network(neurons):
    G = nx.DiGraph()

    for i, neuron in enumerate(neurons, 1):
        G.add_node(f"Neuron {i}")

    for i, neuron in enumerate(neurons, 1):
        for j, weight in enumerate(neuron.weights, 1):
            G.add_edge(f"Neuron {i}", f"Neuron {j}", weight=weight)

    pos = nx.spring_layout(G, seed=42)  # Ustawienie ziarna losowego (seed) na przykładzie 42
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


# Przykład użycia
num_neurons = 3
num_inputs = 4

neurons = create_neurons(num_neurons, num_inputs)
draw_neural_network(neurons)
