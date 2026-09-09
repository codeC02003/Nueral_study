"""
LESSON 2 - Backpropagation: the chain rule as a graph algorithm.

Run me:  python3 lessons/02_backprop.py

Lesson 1 ended with a promise: all P derivatives from one forward pass and one
cheap sweep back. This lesson makes good on it, in about 90 lines of engine.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.engine import Value, trace

def rule(t=""):
    print("\n" + "=" * 70)
    if t: print(t); print("=" * 70)

# ---------------------------------------------------------------------------
# PART A - Computing something builds a graph, for free.
# ---------------------------------------------------------------------------
rule("PART A - the graph is a side effect of the arithmetic")

x = Value(3.0, label="x")      # "great" count, from Lesson 1
w = Value(0.3, label="w")
b = Value(0.0, label="b")

wx = w * x;      wx.label = "w*x"
z = wx + b;      z.label = "z"
p = z.sigmoid(); p.label = "p"

print(f"z = {z.data:.4f}   p = {p.data:.4f}   (same numbers as Lesson 1)")
print("""
But look what got recorded on the way:""")
for v in [p, z, wx]:
    kids = ", ".join(f"{c.label or c._op or f'{c.data:g}'}" for c in v._prev)
    print(f"  {v.label:>3} was produced by op '{v._op}' from ({kids})")
print("""
Nobody asked for that. `w * x` returned a Value that quietly kept a pointer to
w and to x, and a note saying "I am a product." Python normally discards that
history. Keeping it is the entire trick.""")

# ---------------------------------------------------------------------------
# PART B - Do the backward sweep by hand, one node at a time.
# ---------------------------------------------------------------------------
rule("PART B - the sweep, in slow motion")

y = Value(1.0, label="y")
diff = p - y;      diff.label = "p-y"
L = diff ** 2;     L.label = "L"
print(f"L = {L.data:.6f}\n")

print("Everything starts at zero: nothing is known to affect L yet.")
print(f"  L.grad={L.grad}  p.grad={p.grad}  w.grad={w.grad}\n")

L.grad = 1.0
print("SEED:  dL/dL = 1.   'If L rises by 1, L rises by 1.' Trivially true,")
print("       and it is the only fact the whole sweep needs to get started.\n")

print("Now walk backwards. At each node, ONE local question:")
print("  'given how much I affect L, how much does each of my children?'\n")

for node, name in [(L, "L    "), (diff, "p-y  "), (p, "p    "),
                   (z, "z    "), (wx, "w*x  ")]:
    node._backward()
    print(f"  after {name}._backward():  "
          f"p-y.grad={diff.grad:+.5f}  p.grad={p.grad:+.5f}  "
          f"z.grad={z.grad:+.5f}  w.grad={w.grad:+.5f}  b.grad={b.grad:+.5f}")

print(f"""
Watch how the gradient flowed right-to-left, one hop per line, and how
w.grad stayed 0 until the wave physically reached it.

  w.grad = {w.grad:+.8f}   <- Lesson 1 Part F got -0.35639980 by hand. Same number.

Note it took FIVE hops to get there, one per node on the path. I originally
wrote this loop with only four and w.grad stayed stuck at 0.000000 - I had
silently skipped the w*x node. That is the characteristic backprop bug: no
crash, no warning, just a parameter that never learns.

Each node only ever knew a local rule ('I am a product', 'I am a sigmoid').
None of them knew what L was. Locality is what makes this scale.""")

# ---------------------------------------------------------------------------
# PART C - The bug that silently corrupts everything: += vs =
# ---------------------------------------------------------------------------
rule("PART C - why gradients must ACCUMULATE (the classic bug)")

print("""What if a value is used more than once? Say a word appears in two
features, so 'a' feeds two different paths to the loss:

        a ---> a*b ---.
         \\              +---> L
          `--> a*c ---'

Multivariable chain rule: when a quantity influences the output along SEVERAL
paths, its total derivative is the SUM over paths. Miss the sum and you get a
wrong gradient with no error message - the model just trains badly forever.
""")

class Buggy(Value):                    # identical, except '=' instead of '+='
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Buggy(other)
        out = Buggy(self.data * other.data, (self, other), "*")
        out._ordered = (self, other)
        def _backward():
            self.grad  = other.data * out.grad     # BUG: overwrites
            other.grad = self.data  * out.grad     # BUG: overwrites
        out._backward = _backward
        return out
    def __add__(self, other):
        other = other if isinstance(other, Value) else Buggy(other)
        out = Buggy(self.data + other.data, (self, other), "+")
        out._ordered = (self, other)
        def _backward():
            self.grad  = out.grad                  # BUG
            other.grad = out.grad                  # BUG
        out._backward = _backward
        return out


def sweep(root, flip):
    """Same backward pass, but we control which branch is visited first."""
    topo, seen = [], set()
    def build(v):
        if id(v) in seen: return
        seen.add(id(v))
        kids = list(getattr(v, "_ordered", ()))
        for c in (reversed(kids) if flip else kids): build(c)
        topo.append(v)
    build(root)
    for v in topo: v.grad = 0.0
    root.grad = 1.0
    for v in reversed(topo): v._backward()
    return topo


print("Same graph, same numbers. The ONLY thing we change is which branch the")
print("sweep walks first - something you never control and never think about:\n")
print(f"  {'':<22}{'a.grad':>9}   verdict")
for flip in (False, True):
    a, bb, c = Buggy(2.0), Buggy(3.0), Buggy(5.0)
    sweep(a * bb + a * c, flip)
    bad = a.grad
    a2, b2, c2 = Value(2.0), Value(3.0), Value(5.0)
    out = a2 * b2 + a2 * c2; out.backward()
    order = "a*c branch first" if flip else "a*b branch first"
    print(f"  buggy  (= )  {order:<10}{bad:>+9.1f}   WRONG")
    print(f"  correct(+=)  {order:<10}{a2.grad:>+9.1f}   right\n")

print("""True answer: d(ab + ac)/da = b + c = 3 + 5 = 8.

The buggy engine gives +5 or +3 - never 8 - and WHICH wrong answer you get
depends on traversal order. It keeps whichever path it happened to visit last
and silently overwrites the other.

In the real engine `_prev` is a Python set, whose iteration order comes from
object hashes, i.e. memory addresses. So in practice that order is stable
within a session but can change between them. (Writing this lesson I saw this
exact example print +3.0 on one run and +5.0 on twelve others.)

That is the worst class of bug there is: same code, same input, an occasionally
different wrong gradient. Nothing raises. Nothing warns. Your model just trains
slightly worse than it should, forever, and you never find out why.

`+=` makes the result a SUM, and a sum does not care what order it is added in.
Order-independence is the whole reason that one character is correct.""")

# ---------------------------------------------------------------------------
# PART D - Why the order of the sweep matters.
# ---------------------------------------------------------------------------
rule("PART D - topological order")

print("""You cannot process a node until every consumer of it has already pushed
its gradient down - otherwise you'd hand a child an unfinished number.

So backward() first sorts the graph topologically (children before parents),
then walks it in REVERSE. Here's the actual order used for our neuron:
""")
nodes, edges = trace(L)
topo, seen = [], set()
def build(v):
    if v in seen: return
    seen.add(v)
    for ch in v._prev: build(ch)
    topo.append(v)
build(L)
order = " -> ".join((v.label or v._op or f"{v.data:g}") for v in reversed(topo))
print(f"  {order}")
print("""
Run the file again and this order will likely differ - DFS over a set explores
children in whatever order it gets. That is fine. MANY valid topological orders
exist, and the sweep is correct under every one of them. That guarantee is
precisely what the topological sort buys you, and it is why the '+=' from Part C
has to be an order-independent SUM.
""")
print(f"  {len(nodes)} nodes, {len(edges)} edges. Total cost of the sweep:")
print("  one visit per node. That is O(graph size) - NOT O(parameters^2),")
print("  and NOT one forward pass per parameter. This is the four-thousand-")
print("  years-to-two-seconds step from Lesson 1.")

# ---------------------------------------------------------------------------
# PART E - Prove it. Against numerical gradients, and against PyTorch.
# ---------------------------------------------------------------------------
rule("PART E - verification (never trust an autograd you didn't test)")

def f_ours(vals):
    a, bb, c = [Value(v) for v in vals]
    out = (a * bb + c.tanh()) * (a - c).sigmoid() + (bb ** 3) / a
    out.backward()
    return out.data, [a.grad, bb.grad, c.grad]

def f_plain(vals):
    a, bb, c = vals
    return (a * bb + math.tanh(c)) * (1/(1+math.exp(-(a - c)))) + (bb ** 3) / a

vals = [2.0, -3.0, 0.5]
out_ours, g_ours = f_ours(vals)

h = 1e-6                                        # numerical, central difference
g_num = []
for i in range(3):
    up, dn = list(vals), list(vals)
    up[i] += h; dn[i] -= h
    g_num.append((f_plain(up) - f_plain(dn)) / (2 * h))

import torch                                     # and PyTorch, the real thing
t = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals]
a, bb, c = t
o = (a * bb + torch.tanh(c)) * torch.sigmoid(a - c) + (bb ** 3) / a
o.backward()
g_torch = [x.grad.item() for x in t]

print(f"f(a,b,c) = (a*b + tanh(c)) * sigmoid(a-c) + b^3/a   at a=2, b=-3, c=0.5")
print(f"forward:  ours {out_ours:.10f}   torch {o.item():.10f}\n")
print(f"{'':<6}{'ours':>16}{'numerical':>16}{'pytorch':>16}{'err vs torch':>15}")
for n, a_, b_, c_ in zip("abc", g_ours, g_num, g_torch):
    print(f"df/d{n:<3}{a_:>16.10f}{b_:>16.10f}{c_:>16.10f}{abs(a_-c_):>15.2e}")
print("\nOur 90-line engine agrees with PyTorch to machine precision.")

# ---------------------------------------------------------------------------
# PART F - Your Lesson 1 exercise, answered: MSE vs cross-entropy.
# ---------------------------------------------------------------------------
rule("PART F - the saturation fix (Lesson 1's cliffhanger)")

def train(loss_kind, w0=-3.0, steps=80, lr=1.0):
    """Start CONFIDENTLY WRONG: w=-3 => p ~ 0.0001 while the truth is y=1."""
    w, b = Value(w0, label="w"), Value(0.0, label="b")
    hist = []
    for s_ in range(steps):
        x, y = Value(3.0), 1.0
        p = (w * x + b).sigmoid()
        L = (p - y) ** 2 if loss_kind == "mse" else -(p.log())
        w.grad = b.grad = 0.0          # ALWAYS zero grads first: they accumulate!
        L.backward()
        hist.append((p.data, w.grad))
        w.data -= lr * w.grad
        b.data -= lr * b.grad
    return hist

m, ce = train("mse"), train("ce")
print("""Saturation does not bite when the model is nearly right - it bites when
the model is CONFIDENTLY WRONG. So start at w = -3: the neuron screams
"negative!" (p = 0.0001) when the truth is y = 1. Maximum possible error.
""")
print(f"{'step':>5} | {'MSE  p':>10} {'dL/dw':>11} | {'CE   p':>10} {'dL/dw':>11}")
for s_ in [0, 1, 5, 20, 79]:
    print(f"{s_:>5} | {m[s_][0]:>10.6f} {m[s_][1]:>11.2e} | "
          f"{ce[s_][0]:>10.6f} {ce[s_][1]:>11.2e}")

print(f"""
At step 0 both are as wrong as it is possible to be. Their gradients:

    MSE : {m[0][1]:.2e}        CE : {ce[0][1]:.2e}        ratio {abs(ce[0][1]/m[0][1]):,.0f}x

MSE is handed a gradient of {abs(m[0][1]):.0e} - indistinguishable from "you are correct,
change nothing." After 80 steps it has crawled to p = {m[79][0]:.4f}. It is not
learning slowly; it is functionally DEAD. Cross-entropy fixed it by step 1.

WHY, in one line of algebra. For y=1, L = -log(p) and p = sigmoid(z):

    dL/dz = dL/dp * dp/dz = (-1/p) * p(1-p) = -(1-p) = p - 1

The p CANCELS. Cross-entropy is built so its derivative annihilates the
sigmoid's vanishing factor. What survives is (prediction - target): exactly
proportional to how wrong you are, and it never saturates.

Compare MSE, whose dL/dz = 2(p-y) * p(1-p) keeps that fatal p(1-p) factor.
The more confident and wrong it is, the LESS it wants to change. That is
backwards, and it is why nobody trains classifiers with squared error.

That cancellation is not luck. It is why this pairing sits at the output of
every classifier on earth, Network A's final layer included.""")

print("\n" + "=" * 70)
print("NEXT: Lesson 3 - tensors. Same engine, but on whole matrices at once,")
print("plus the thing that breaks everyone: how broadcasting backpropagates.")
print("=" * 70)
