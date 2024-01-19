import matplotlib.pyplot as plt
import networkx as nx

from simulation_creating_neurons import create_neurons

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = []
        self.initialize_neurons()

    def initialize_neurons(self):
        for i in range(1, len(self.layers)):
            layer_neurons = create_neurons(self.layers[i], self.layers[i - 1])
            self.neurons.append(layer_neurons)

    def forward_pass(self, inputs):
        if len(inputs) != self.layers[0]:
            raise ValueError("Number of input values must match the input layer size.")

        layer_inputs = inputs
        for layer_neurons in self.neurons:
            layer_outputs = [neuron.activate(layer_inputs) for neuron in layer_neurons]
            layer_inputs = layer_outputs

        return layer_outputs


def draw_neural_network_graph(neural_network):
    G = nx.DiGraph()

    for layer_idx, layer_neurons in enumerate(neural_network.neurons):
        for i, neuron in enumerate(layer_neurons, 1):
            G.add_node(f"Layer {layer_idx + 1}, Neuron {i}")

    for layer_idx, layer_neurons in enumerate(neural_network.neurons[:-1]):
        for i, neuron in enumerate(layer_neurons, 1):
            for j, weight in enumerate(neural_network.neurons[layer_idx + 1], 1):
                G.add_edge(f"Layer {layer_idx + 1}, Neuron {i}", f"Layer {layer_idx + 2}, Neuron {j}", weight=weight.weights[i - 1])

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


# Przykład użycia
network_layers = [4, 3, 2]  # 4 input neurons, 3 neurons in hidden layer, 2 output neurons
neural_network = NeuralNetwork(network_layers)

input_values = [0.5, 0.3, 0.8, 0.2]
output_values = neural_network.forward_pass(input_values)

draw_neural_network_graph(neural_network)
