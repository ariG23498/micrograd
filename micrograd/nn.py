import random
from typing import List

from .engine import Value


class Module(object):
    """
    The parent class for all neural network modules.
    """

    def zero_grad(self):
        # Zero out the gradients of all parameters.
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # Initialize a parameters function that all the children will
        # override and return a list of parameters.
        return []


class Neuron(Module):
    """
    A single neuron.
    Parameters:
        nin (int): number of inputs
        nonlin (bool): whether to apply ReLU nonlinearity
    """

    def __init__(self, nin: int, nonlin: bool = True):
        # Create weights for the neuron. The weights are initialized
        # from a random uniform distribution.
        self.w = [Value(data=random.uniform(-1, 1)) for _ in range(nin)]

        # Create bias for the neuron.
        self.b = Value(data=0.0)
        self.nonlin = nonlin

    def __call__(self, x: List["Value"]) -> "Value":
        # Compute the dot product of the input and the weights. Add the
        # bias to the dot product.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # If activation is mentioned apply ReLU to it.
        return act.relu() if self.nonlin else act

    def parameters(self):
        # Get the parameters of the neuron. The parameters of a neuron
        # is its weights and bias.
        return self.w + [self.b]

    def __repr__(self):
        # Print a better representation of the neuron.
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"


class Layer(Module):
    """
    A layer of neurons.
    Parameters:
        nin (int): number of inputs
        nout (int): number of outputs
    """

    def __init__(self, nin: int, nout: int, **kwargs):
        # A layer is a list of neurons.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List["Value"]) -> List["Value"]:
        # Iterate over all the neurons and compute the output of each.
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        # The parameters of a layer is the parameters of all the neurons.
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        # Print a better representation of the layer.
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    The Multi-Layer Perceptron (MLP) class.
    Parameters:
        nin (int): number of inputs.
        nouts (List[int]): number of outputs in each layer.
    """

    def __init__(self, nin: int, nouts: List[int]):
        # Get the number of input and all the number of outputs in
        # a single list.
        sz = [nin] + nouts

        # Build layers by connecting each layer to the previous one.
        self.layers = [
            # Do not use non linearity in the last layer.
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x: List["Value"]) -> List["Value"]:
        # Iterate over the layers and compute the output of
        # each sequentially.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Get the parameters of the MLP
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        # Print a better representation of the MLP.
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
