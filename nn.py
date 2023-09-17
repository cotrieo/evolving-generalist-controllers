import numpy as np


class NeuralNetwork:

    def __init__(self, x):
        self.x = x

    @staticmethod
    def calculate_total_connections(layer_sizes):
        total_connections = 0
        for i in range(len(layer_sizes) - 2):
            total_connections += (layer_sizes[i] + layer_sizes[-1]) * layer_sizes[i + 1]
        return total_connections

    @staticmethod
    def calculate_layer_shapes(num_layers, layer_sizes):
        layer_shapes = []
        for i in range(num_layers - 1):
            shape = [layer_sizes[i + 1], layer_sizes[i] + layer_sizes[-1]]
            layer_shapes.append(shape)
        return layer_shapes

    def reshape_layers(self, layer_sizes):

        num_layers = len(layer_sizes) - 1

        layer_shapes = NeuralNetwork.calculate_layer_shapes(num_layers, layer_sizes)
        input_data = self.x
        start_index = 0
        total_connections = 0
        layer_weights = []

        for i in range(num_layers - 2):
            total_connections = (layer_shapes[i][0] * layer_shapes[i][1]) + start_index
            weight_matrix = input_data[start_index:total_connections].reshape(layer_shapes[i][0], layer_shapes[i][1])
            start_index = total_connections
            layer_weights.append(weight_matrix)

        last_layer_weights = input_data[total_connections:].reshape(layer_shapes[-1][0], layer_shapes[-1][1])
        layer_weights.append(last_layer_weights)
        return layer_weights

    def feedforward(self, layer_weights, layer_sizes, observations):

        bias_units = layer_sizes[-1]
        layer_sizes = layer_sizes[:-1]

        input_layer = np.ones((layer_sizes[0] + bias_units, 1))

        input_layer[1:, 0] = observations[0:]
        current_layer = layer_weights[0] @ input_layer

        for i in range(len(layer_sizes) - 2):
            hidden_layer = np.tanh(np.ones((layer_sizes[1] + bias_units, 1)))
            hidden_layer[1:, 0] = current_layer[0:, 0]
            current_layer = np.tanh(layer_weights[i + 1] @ hidden_layer)

        return current_layer.flatten()
