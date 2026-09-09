"""
Layers built on top of the scalar engine. Neuron -> Layer -> MLP.

Note how little there is here. Once backward() works, a "network" is just
a lot of Values wired together - the engine never needs to know the shape.
"""
import random
from nn.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        # Gradients ACCUMULATE (Lesson 2, Part C). Forget this and every step
        # gets the sum of all previous steps' gradients. Classic silent killer.
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, nonlin="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "tanh": return act.tanh()
        if self.nonlin == "relu": return act.relu()
        return act                                     # linear output

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """nin inputs -> hidden layers (nonlinear) -> final layer (LINEAR)."""

    def __init__(self, nin, nouts, nonlin="tanh"):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(nonlin if i < len(nouts) - 1 else "linear"))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def hidden(self, x):
        """Activations of the first hidden layer - the start of Network B's job."""
        return self.layers[0]([Value(v) if not isinstance(v, Value) else v for v in x])

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
