"""Layers on the tensor engine. This is what Network A will actually be built from."""
import numpy as np
from nn.tensor import Tensor


class Linear:
    """y = x @ W + b.  One matmul replaces an entire Layer of scalar Neurons."""

    def __init__(self, nin, nout, bias=True):
        # He/Kaiming init - Lesson 7 explains why the scale matters this much.
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin), label="W")
        self.b = Tensor(np.zeros(nout), label="b") if bias else None

    def __call__(self, x):
        out = x @ self.W
        return out + self.b if self.b is not None else out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class MLP:
    def __init__(self, nin, nouts, nonlin="tanh"):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i + 1]) for i in range(len(nouts))]
        self.nonlin = nonlin

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:                  # last layer stays linear
                x = x.tanh() if self.nonlin == "tanh" else x.relu()
        return x

    def hidden(self, x):
        h = self.layers[0](x)
        return h.tanh() if self.nonlin == "tanh" else h.relu()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


def bce_with_logits(logits, y):
    """Numerically stable binary cross-entropy. Lesson 5 derives the stability trick."""
    p = logits.sigmoid()
    eps = 1e-12
    return -((p + eps).log() * y + (1 - p + eps).log() * (1 - y)).mean()
