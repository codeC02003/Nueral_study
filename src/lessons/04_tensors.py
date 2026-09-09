"""
LESSON 4 - Tensors: the same engine, on arrays.

Run me:  python3 lessons/04_tensors.py

No new ideas about differentiation appear in this lesson. Every _backward still
answers the same local question. What changes is that gradients now have SHAPE,
and shape is where the bugs live.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nn.tensor import Tensor, unbroadcast
from nn.layers import MLP, bce_with_logits

def rule(t=""):
    print("\n" + "=" * 70)
    if t: print(t); print("=" * 70)

# ---------------------------------------------------------------------------
rule("PART A - why we are doing this at all")

from nn.engine import Value
from nn.mlp import MLP as ScalarMLP
import random

NIN, NHID, BATCH, STEPS = 8, 16, 32, 15
Xd = np.random.RandomState(0).randn(BATCH, NIN)
Yd = (Xd[:, 0] * Xd[:, 1] > 0).astype(float)

random.seed(0)
smlp = ScalarMLP(NIN, [NHID, 1])
t0 = time.time()
for _ in range(STEPS):
    loss = Value(0.0)
    for row, yv in zip(Xd, Yd):
        o = smlp([Value(v) for v in row])
        p = o.sigmoid()
        loss = loss + -((p.log() * yv) + ((1 - p).log() * (1 - yv)))
    loss = loss * (1.0 / BATCH)
    smlp.zero_grad(); loss.backward()
    for p_ in smlp.parameters(): p_.data -= 0.1 * p_.grad
t_scalar = time.time() - t0

np.random.seed(0)
tmlp = MLP(NIN, [NHID, 1])
Xt, Yt = Tensor(Xd), Tensor(Yd.reshape(-1, 1))
t0 = time.time()
for _ in range(STEPS):
    loss = bce_with_logits(tmlp(Xt), Yt)
    tmlp.zero_grad(); loss.backward()
    for p_ in tmlp.parameters(): p_.data -= 0.1 * p_.grad
t_tensor = time.time() - t0

nvals = BATCH * (NIN * NHID + NHID + NHID + 1)
print(f"  same network ({NIN}->{NHID}->1), same batch ({BATCH}), {STEPS} steps\n")
print(f"    scalar engine : {t_scalar*1000:>8.1f} ms   ~{nvals*STEPS:,} Value objects created")
print(f"    tensor engine : {t_tensor*1000:>8.1f} ms   {len(tmlp.parameters())} arrays")
print(f"    speedup       : {t_scalar/t_tensor:>8.0f}x")
print(f"""
  Scale that up. GPT-2's forward pass touches ~10^11 numbers. At one Python
  object each, you would run out of RAM before you finished a single step.

  The engine is unchanged. We just stopped putting one number in each box.""")

# ---------------------------------------------------------------------------
rule("PART B - matmul backward, derived by shape-matching")
print("""  Y = X @ W       X:(n,d)   W:(d,m)   Y:(n,m)   dL/dY:(n,m)

  You do not need to remember the rules. Ask what shapes are even LEGAL:

    dL/dX must be (n,d). You hold dL/dY:(n,m) and W:(d,m).
        dL/dY @ W      -> (n,m)@(d,m)  illegal
        dL/dY @ W.T    -> (n,m)@(m,d)  = (n,d)   the only option that fits

    dL/dW must be (d,m). You hold X:(n,d) and dL/dY:(n,m).
        X   @ dL/dY    -> (n,d)@(n,m)  illegal
        X.T @ dL/dY    -> (d,n)@(n,m)  = (d,m)   again, the only fit

  In both cases exactly one arrangement type-checks, and it is the right one.
  That is not luck; it is what the chain rule looks like when it has shapes.""")

X = Tensor(np.random.RandomState(1).randn(4, 3))
W = Tensor(np.random.RandomState(2).randn(3, 5))
Y = X @ W
Y.backward()
print(f"\n  X{X.shape} @ W{W.shape} = Y{Y.shape}")
print(f"  dL/dX {X.grad.shape}   dL/dW {W.grad.shape}   <- shapes always match their tensor")

# ---------------------------------------------------------------------------
rule("PART C - broadcasting: the bug that does not crash")

class BrokenTensor(Tensor):
    """Identical, except __add__ forgets to unbroadcast."""
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else BrokenTensor(other)
        out = BrokenTensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  = self.grad  + out.grad      # no unbroadcast
            other.grad = other.grad + out.grad      # no unbroadcast
        out._backward = _backward
        return out

print("""  A bias has one number per output feature. A batch has one row per example.
  numpy silently stretches the bias across the batch on the way forward:

        x @ W  :  (4, 5)          b  :  (5,)          x @ W + b  :  (4, 5)

  So on the way back, gradient arrives at b with shape (4,5) - but b only has
  5 numbers. Those 4 copies were THE SAME parameter used 4 times, so by the
  multivariable chain rule (Lesson 2, Part C) their gradients must be SUMMED.

      forward broadcast  <->  backward sum.   They are duals. Always.
""")
xw = np.random.RandomState(3).randn(4, 5)
for cls, tag in [(Tensor, "correct  "), (BrokenTensor, "broken   ")]:
    b = cls(np.zeros(5))
    out = cls(xw) + b
    out.backward()
    print(f"    {tag} b.shape {b.shape}   dL/db.shape {b.grad.shape}"
          f"   {'OK' if b.grad.shape == b.shape else '<-- MISMATCH'}")

print("""
  Now the part that makes this dangerous. Nothing raised. So watch what one
  training step does to the broken bias:""")
b_ok, b_bad = Tensor(np.zeros(5)), BrokenTensor(np.zeros(5))
for step in range(1, 4):
    for b in (b_ok, b_bad):
        b.zero_grad()
        o = type(b)(xw) + b
        o.backward()
        b.data = b.data - 0.1 * b.grad
    print(f"    after step {step}:   correct b.shape {b_ok.shape}      broken b.shape {b_bad.shape}")

print(f"""
  The bias quietly became a {b_bad.shape} matrix - a separate bias per batch
  POSITION. It is no longer a bias; it is 20 numbers memorising which row of
  the batch an example happened to land in. Change the batch size and the model
  breaks, or worse, silently degrades. No exception is ever raised.

  This is the tensor-scale version of the '=' vs '+=' bug from Lesson 2, and it
  is the single most common hand-written-backprop error there is. Six lines in
  nn/tensor.py fix it: `unbroadcast`.""")

# ---------------------------------------------------------------------------
rule("PART D - verification against PyTorch")
import torch

def check(name, build_np, build_torch, shapes):
    arrs = [np.random.RandomState(i + 10).randn(*s) for i, s in enumerate(shapes)]
    ours = [Tensor(a) for a in arrs]
    build_np(*ours).sum().backward()
    ts = [torch.tensor(a, requires_grad=True) for a in arrs]
    build_torch(*ts).sum().backward()
    errs = [np.abs(o.grad - t.grad.numpy()).max() for o, t in zip(ours, ts)]
    ok = all(o.grad.shape == t.grad.shape for o, t in zip(ours, ts))
    print(f"  {name:<34} max err {max(errs):.2e}   shapes {'match' if ok else 'DIFFER'}")

check("matmul + broadcast bias + tanh",
      lambda X, W, b: (X @ W + b).tanh(),
      lambda X, W, b: (X @ W + b).tanh(), [(6, 4), (4, 3), (3,)])
check("two layers, relu, sum-reduce",
      lambda X, W1, W2: ((X @ W1).relu() @ W2).sum(axis=1),
      lambda X, W1, W2: ((X @ W1).relu() @ W2).sum(axis=1), [(5, 4), (4, 7), (7, 2)])
check("param reused twice (accumulation)",
      lambda X, W: (X @ W) * (X @ W),
      lambda X, W: (X @ W) * (X @ W), [(4, 3), (3, 3)])
check("transpose + column broadcast",
      lambda X, c: (X.T + c).exp(),
      lambda X, c: (X.T + c).exp(), [(4, 3), (3, 1)])
print("\n  Every gradient matches PyTorch, and every SHAPE matches too.")

# ---------------------------------------------------------------------------
rule("PART E - the negation network, rebuilt on tensors")
DATA_X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=float)
DATA_Y = np.array([[1.], [0.], [0.], [1.]])
np.random.seed(3)
A = MLP(2, [4, 1])
Xt, Yt = Tensor(DATA_X), Tensor(DATA_Y)
t0 = time.time()
for step in range(2001):
    loss = bce_with_logits(A(Xt), Yt)
    A.zero_grad(); loss.backward()
    for p in A.parameters(): p.data -= 0.5 * p.grad
    if step % 500 == 0:
        pred = 1 / (1 + np.exp(-A(Xt).data))
        acc = ((pred > .5) == (DATA_Y > .5)).sum()
        print(f"  step {step:>4}   loss {loss.data:.6f}   correct {acc}/4")
print(f"\n  {time.time()-t0:.2f}s for 2000 steps. The scalar version took ~15s for 400.")
pred = 1 / (1 + np.exp(-A(Xt).data))
for (x, y, p) in zip(DATA_X, DATA_Y.ravel(), pred.ravel()):
    print(f"    x={x.astype(int)}  target {y:.0f}   predicted {p:.6f}")
print("""
  Same task, same answer, one matmul instead of 4 loops over 17 Value objects.
  Network A can now be something real.""")

print("\n" + "=" * 70)
print("NEXT: PyTorch, a code corpus, and a training loop that does not stop.")
print("=" * 70)
