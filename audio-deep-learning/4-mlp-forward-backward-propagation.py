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

        # initialize derivatives
        self.derivatives = [
            np.zeros((layers[i], layers[i + 1])) for i in range(len(layers) - 1)
        ]

        # initialize activations
        self.activations = [np.zeros(layer_size) for layer_size in layers]

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

        # save activations for backpropagation
        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            # compute previous activation with the current weight matrix
            # h(xW) in the tutorial
            net_inputs = activations @ w

            # apply activation function to the net input
            # a = f(h) in the notes
            activations = self._sigmoid(net_inputs)

            # save activations
            # a_3 = s(h_3)
            # h_3 = a_2 * w_2
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error):
        """
        Backpropagate an error signal.
        Args:
            error (nd.array): The error to backprop
        Returns:
            error (nd.array): The final error of the input
        """
        # dE/dW_i = (y - a_[i + 1] * s'(h_[i + 1])) * a_i
        # s'(h_[i + 1]) = s(h_[i + 1]) * ( 1 - s(h_[i + 1]))
        # s(h_[i + 1]) = a_[i + 1]
        # dE/dW_[i-1] = (y - a_[i + 1]) * s'(h_[i + 1])) * W_i * s'(h_i) * a_[i - 1]
        for i in reversed(range(len(self.derivatives))):
            # activations for subsequent layer
            activations = self.activations[i + 1]

            # (y - a_[i + 1] * s'(h_[i + 1]))
            # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta = error * self._sigmoid_derivative(activations)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.outer(self.activations[i], delta)

            # backpropagate next error
            # (y - a_[i + 1]) * s'(h_[i + 1])) * W_i
            error = delta @ self.weights[i].T

        return error

    def train(self, inputs, targets, epochs, learning_rate=1.0):
        """
        Train model running forward propagation and backward propagation.
        """
        for epoch in range(1, epochs + 1):
            sum_errors = 0.0
            for x, target in zip(inputs, targets):
                # forward propagation
                output = self.forward_propagate(x)

                # calculate error
                error = target - output

                # backward propagation
                self.back_propagate(error)

                # gradient descent
                self.gradient_descent(learning_rate)

                # update total error
                sum_errors += self._mse(target, output)

            # report error
            avg_error = sum_errors / len(inputs)
            print(f"Epoch {epoch:2d}/{epochs} - MSE: {avg_error:.6f}")

        print("Training Finished!\n=============")

    def gradient_descent(self, learning_rate=1.0):
        """
        Update the weights by stepping down the gradient.
        Args:
            learning_rate: how fast to learn

        """
        for weight, derivative in zip(self.weights, self.derivatives):
            weight += derivative * learning_rate

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

    def _sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Args:
            x(float): input value to be processed
        Returns:
            y(float): output of the function
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """
        Mean Square Error Loss Function
        Args:
            target (nd.array): The ground truth
            output (nd.array)
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    items = rng.random((1000, 2)) * 0.5
    targets = items.sum(axis=1, keepdims=True)

    # train model
    mlp = MLP(num_inputs=2, hidden_layers=[5], num_outputs=1, rng=rng)
    mlp.train(items, targets, epochs=50, learning_rate=0.1)

    # inference
    test_input = np.array([0.3, 0.1])
    expected = test_input.sum()
    predicted = mlp.forward_propagate(test_input)

    # print results
    print(
        f"Input: {test_input[0]} + {test_input[1]} | "
        f"Expected: {expected:.2f} | "
        f"Predicted: {predicted[0]:.4f}"
    )
