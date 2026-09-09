"""
The same autograd engine, but every node holds a numpy ARRAY instead of one float.

Nothing conceptually new happens here. It is still: forward value, list of
children, a local _backward. What changes is that each _backward now has to
answer a harder question - "what SHAPE is my child's gradient?" - and getting
that wrong is the most common bug in hand-written deep learning code.
"""
import numpy as np


def unbroadcast(g, shape):
    """Sum a gradient back down to `shape`.

    THE most important six lines in this file. numpy silently expands a (3,)
    bias across a (64, 3) batch on the way forward. On the way back, gradient
    arrives with shape (64, 3) but the bias only has 3 numbers. The 64 copies
    were all THE SAME parameter, so by the multivariable chain rule (Lesson 2,
    Part C) their gradients must be SUMMED.

    Forward broadcast  <->  backward sum. They are duals. Always.
    """
    while g.ndim > len(shape):          # axes that broadcasting prepended
        g = g.sum(axis=0)
    for i, s in enumerate(shape):       # axes that were size 1 and got stretched
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    # -- shape-preserving elementwise ops -----------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Tensor(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    # -- the one that isn't elementwise -------------------------------------
    def __matmul__(self, other):
        """Y = X @ W.

        The two backward rules look like magic but fall straight out of shapes:

            X: (n, d)   W: (d, m)   Y: (n, m)   dL/dY: (n, m)

            dL/dX must be (n, d).  From (n,m) and (d,m) the only way to get
                                   (n,d) is  dL/dY @ W.T
            dL/dW must be (d, m).  From (n,d) and (n,m) the only way is
                                   X.T @ dL/dY

        There is exactly one legal arrangement in each case. That is not a
        coincidence - it is the chain rule, and shape-matching finds it for you.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            self.grad  += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad
        out._backward = _backward
        return out

    # -- reductions ---------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        """Backward of sum is BROADCAST - the exact mirror of unbroadcast."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(m, (self,), "max")
        def _backward():
            mm = self.data.max(axis=axis, keepdims=True)
            mask = (self.data == mm).astype(np.float64)
            mask /= mask.sum(axis=axis, keepdims=True)      # split ties evenly
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += mask * g
        out._backward = _backward
        return out

    # -- nonlinearities -----------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    # -- shape ops ----------------------------------------------------------
    @property
    def T(self):
        out = Tensor(self.data.swapaxes(-1, -2), (self,), "T")
        def _backward():
            self.grad += out.grad.swapaxes(-1, -2)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Indexing. This is how an embedding table works - Lesson 8."""
        out = Tensor(self.data[idx], (self,), "idx")
        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)     # .at, not [idx] +=, so repeats accumulate
            self.grad += g
        out._backward = _backward
        return out

    # -- the sweep (identical to the scalar engine) -------------------------
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if id(v) in seen: return
            seen.add(id(v))
            for c in v._prev: build(c)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # -- boilerplate --------------------------------------------------------
    def __neg__(self):        return self * -1
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-(o if isinstance(o, Tensor) else Tensor(o)))
    def __rsub__(self, o):    return (-self) + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Tensor) else Tensor(o) ** -1)
    @property
    def shape(self):          return self.data.shape
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"
