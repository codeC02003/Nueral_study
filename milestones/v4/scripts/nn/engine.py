"""
A scalar reverse-mode automatic differentiation engine.

This is the heart of every deep learning framework, PyTorch included, stripped
down to something you can hold in your head. ~90 lines of real code.

THE IDEA
    When you write  L = (w*x + b - y)**2  in Python, Python computes a number
    and throws away how it got there. We instead build a GRAPH as a side effect
    of computing: every Value remembers which Values produced it, and HOW.

    That memory is what lets us walk backwards later and hand every parameter
    its derivative - all of them, in one sweep.
"""
import math


class Value:
    """A single number that remembers where it came from."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)   # the number itself (forward pass)
        self.grad = 0.0           # dL/d(self)  -- filled in by backward(). Starts
                                  # at 0 = "as far as we know, I don't affect L"
        self._prev = set(_children)   # the Values that produced me (graph edges)
        self._op = _op                # what operation produced me (for display)
        self.label = label            # human name, for the visualizer

        # The local rule for pushing gradient from me to my children.
        # Set by each operation below. Leaves (inputs/params) push to nobody.
        self._backward = lambda: None

    # -- operations ---------------------------------------------------------
    # Each one does TWO things:
    #   1. compute the forward value
    #   2. define _backward: given MY grad, how much grad do my children get?

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1. Addition is a gradient ROUTER:
            # it copies its incoming gradient to both children, unchanged.
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b. Multiplication is a gradient SWITCH:
            # each child receives the gradient scaled by the OTHER child's value.
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad      # d(e^x)/dx = e^x = out.data
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad   # the p(1-p) from Lesson 1
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # Gradient passes through untouched if active, is BLOCKED if not.
            # No shrinking factor -> this is why ReLU beat sigmoid for depth.
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # -- the sweep ----------------------------------------------------------

    def backward(self):
        """Fill in .grad for every Value that this one depends on."""

        # A node's gradient is only final once EVERY consumer of it has pushed.
        # So we need an order where children come after parents: a topological
        # sort of the graph, reversed.
        topo, visited = [], set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)       # appended only after all its children -> children
        build(self)              # appear EARLIER in topo, so reversed() is correct

        # Seed the sweep. dL/dL = 1: "if L increases by 1, L increases by 1."
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    # -- boilerplate --------------------------------------------------------
    def __neg__(self):          return self * -1
    def __radd__(self, o):      return self + o
    def __sub__(self, o):       return self + (-o)
    def __rsub__(self, o):      return (-self) + o
    def __rmul__(self, o):      return self * o
    def __truediv__(self, o):   return self * (o ** -1 if not isinstance(o, Value) else o ** -1)
    def __rtruediv__(self, o):  return (self ** -1) * o
    def __repr__(self):
        n = f"{self.label}=" if self.label else ""
        return f"Value({n}{self.data:.4f}, grad={self.grad:.4f})"


def trace(root):
    """Collect every node and edge reachable from root. Used by the visualizer."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges
