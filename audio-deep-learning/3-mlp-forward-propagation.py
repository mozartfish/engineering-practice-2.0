import numpy as np


class MLP:
    def __init__(self, num_inputs=3, hidden_layers=None, num_outputs=2, rng=None):
        """
        Constructor for a multi-layer-perceptron(MLP).
        Args:
            num_inputs(int): number of neurons in the input layer
            hidden_layers = (list[int], optional): Number of neurons in each hidden layer.
            num_outputs(int): Number of neurons in the output layer
        """

        if hidden_layers is None:
            hidden_layers = [3, 5]
        if rng is None:
            rng = np.random.default_rng(seed=42)

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self._rng = rng

        # layer representation
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initialize network weights
        self.weights = [
            self._rng.random((layers[i], layers[i + 1])) for i in range(len(layers) - 1)
        ]

    def forward_propagate(self, inputs):
        """
        Computes forward propagation of the network based on input signals.

        Args:
            inputs(np.ndarray): input signals

        Returns:
            nd.array: Output layer activations
        """

        # input layer activations = input layer
        activations = inputs

        for w in self.weights:
            # compute previous activation with the current weight matrix
            net_inputs = activations @ w

            # apply activation function to the net input
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        """
        Activation Function - Sigmoid is very common. there are many others like RELU, TANH etc
        Activation functions should be differentiable continuous functions because we need to take the gradient

        Args:
            x(float): input value to be processed
        Returns:
            y(float): output of the function
        """
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # create a multi-layer perceptron
    mlp = MLP(num_inputs=3, hidden_layers=[3, 5], num_outputs=2, rng=rng)

    # set random values for network input
    input = rng.random(mlp.num_inputs)

    # perform forward pass
    output = mlp.forward_propagate(input)

    # print results
    print(f"network input: {input} | network output: {output}")
