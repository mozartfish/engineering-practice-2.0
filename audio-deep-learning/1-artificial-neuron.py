import math


def sigmoid(x):
    """
    Activation Function - Sigmoid is very common. there are many others like RELU, TANH etc
    Activation functions should be differentiable continuous functions because we need to take the gradient
    """
    return 1.0 / (1 + math.exp(-x))


def activate(inputs, weights):
    # net input
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w

    # perform activation
    return sigmoid(h)


if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    output = activate(inputs, weights)
    print(f"output: {output}")
