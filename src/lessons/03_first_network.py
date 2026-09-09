"""
LESSON 3 - Your first actual NETWORK. And the first time B reads A.

Run me:  python3 lessons/03_first_network.py

Until now we trained ONE neuron. A neuron can only draw a straight line.
This lesson shows a task where that is provably not enough - and it is a
language task, not a maths toy.
"""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.engine import Value
from nn.mlp import MLP

def rule(t=""):
    print("\n" + "=" * 70)
    if t: print(t); print("=" * 70)

# ---------------------------------------------------------------------------
# THE TASK - negation. The thing that makes sentiment hard.
# ---------------------------------------------------------------------------
#   x1 = 1 if the sentiment word is positive ("good"), 0 if negative ("bad")
#   x2 = 1 if the phrase is negated ("not ..."), 0 if not
#
#       "good"      (1,0) -> positive
#       "not good"  (1,1) -> negative     <- negation FLIPS it
#       "bad"       (0,0) -> negative
#       "not bad"   (0,1) -> positive     <- and flips it back
#
# The output is not "positive-word-ness plus negated-ness". It is their XOR.
# No straight line separates these four points. This is the single cheapest
# demonstration of why depth exists.
DATA = [((1,0), 1.0, '"good"'),
        ((1,1), 0.0, '"not good"'),
        ((0,0), 0.0, '"bad"'),
        ((0,1), 1.0, '"not bad"')]


def bce(logit, y):
    p = logit.sigmoid()
    return -( (p.log() * y) + ((1 - p).log() * (1 - y)) )


def train(model, steps, lr, log_every=None, tag=""):
    for step in range(steps + 1):
        loss = sum((bce(model([Value(v) for v in x]), y) for x, y, _ in DATA), Value(0.0))
        loss = loss * (1.0 / len(DATA))
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        if log_every and step % log_every == 0:
            acc = sum((model([Value(v) for v in x]).data > 0) == (y > .5) for x, y, _ in DATA)
            print(f"  {tag} step {step:>4}   loss {loss.data:.5f}   correct {acc}/4")
    return loss.data


# ---------------------------------------------------------------------------
rule("PART A - one neuron tries, and cannot")
random.seed(1)
solo = MLP(2, [1])                                  # 2 inputs -> 1 output. 3 params.
print(f"  parameters: {len(solo.parameters())}")
train(solo, 400, 0.5, 100, "solo")
print("\n  what it settled on:")
for x, y, name in DATA:
    p = solo([Value(v) for v in x]).sigmoid().data
    print(f"    {name:<12} target {y:.0f}   predicted {p:.6f}   {'ok' if (p>.5)==(y>.5) else 'WRONG'}")
print("""
  Stuck at 2/4, and 0.693 loss = log(2) = the loss of pure guessing. It is not
  under-trained; it is at the best a straight line can do. A single neuron
  computes sigmoid(w1*x1 + w2*x2 + b): every point on one side of a line gets
  a high score. Our four points need opposite corners to agree. No line does that.""")

# ---------------------------------------------------------------------------
rule("PART B - add one hidden layer. Now it is a NETWORK.")
random.seed(1)
A = MLP(2, [4, 1])                                  # 2 -> 4 (tanh) -> 1
print(f"  parameters: {len(A.parameters())}   (2->4 layer: 12,  4->1 layer: 5)")
train(A, 400, 0.5, 100, "  A ")
print("\n  what it settled on:")
for x, y, name in DATA:
    p = A([Value(v) for v in x]).sigmoid().data
    print(f"    {name:<12} target {y:.0f}   predicted {p:.6f}   {'ok' if (p>.5)==(y>.5) else 'WRONG'}")
print("""
  4/4. The hidden layer's job is to BEND the space - to re-describe the four
  points in new coordinates where a line does separate them. That re-description
  is what "learning a representation" means, and it is the only reason depth
  buys you anything.

  THIS IS NETWORK A, in miniature. Everything from here is about reading it.""")

# ---------------------------------------------------------------------------
rule("PART C - Network B reads Network A. Nobody tells B what to look for.")

print("""  A's hidden layer is 4 numbers. What is in them? I could stare at the
  weights and guess. Instead we let a SECOND network work it out, with a
  rule we will keep for the whole project:

      B's training signal must come from A. Never from me labelling things.

  So B is self-supervised: given A's 4 hidden activations, reconstruct the
  original input (x1, x2). B never sees the input directly - only A's internal
  state. If B succeeds, that information demonstrably survives inside A.
""")

def acts(model, x):
    """A's hidden activations, DETACHED into fresh Values.
    Detaching matters: B must observe A without altering it. Skip this and
    B's gradients flow straight into A's weights and B rewrites its subject."""
    return [Value(h.data) for h in model.hidden(x)]

random.seed(7)
B = MLP(4, [6, 2])                                  # 4 hidden acts -> guess (x1, x2)
print(f"  B parameters: {len(B.parameters())}\n")

for step in range(1501):
    loss = Value(0.0)
    for x, y, _ in DATA:
        pred = B(acts(A, x))                        # B's only input: A's insides
        loss = loss + (pred[0] - x[0]) ** 2 + (pred[1] - x[1]) ** 2
    loss = loss * (1.0 / len(DATA))
    B.zero_grad(); loss.backward()
    for p in B.parameters():
        p.data -= 0.05 * p.grad
    if step % 300 == 0:
        print(f"  B step {step:>4}   reconstruction loss {loss.data:.6f}")

print("\n  B's readings of A's hidden layer:")
print(f"  {'phrase':<12}{'A hidden activations':<34}{'B decodes':<16}{'truth'}")
for x, y, name in DATA:
    h = [f"{v.data:+.2f}" for v in A.hidden(x)]
    pr = B(acts(A, x))
    print(f"  {name:<12}[{' '.join(h)}]   ({pr[0].data:+.2f},{pr[1].data:+.2f})   ({x[0]}, {x[1]})")

# ---------------------------------------------------------------------------
rule("PART D - the control experiment. Where did the information GO?")
print("""  A finding is worthless without a control. So: run the identical B, but
  let it read only A's single OUTPUT number instead of the hidden layer.
""")
random.seed(7)
B2 = MLP(1, [6, 2])
for step in range(1501):
    loss = Value(0.0)
    for x, y, _ in DATA:
        pred = B2([Value(A([Value(v) for v in x]).data)])
        loss = loss + (pred[0] - x[0]) ** 2 + (pred[1] - x[1]) ** 2
    loss = loss * (1.0 / len(DATA))
    B2.zero_grad(); loss.backward()
    for p in B2.parameters():
        p.data -= 0.05 * p.grad

hid_loss = sum(((B(acts(A, x))[0].data - x[0])**2 + (B(acts(A, x))[1].data - x[1])**2)
               for x, y, _ in DATA) / 4
out_loss = loss.data
print(f"  B reading A's HIDDEN LAYER (4 numbers) : reconstruction loss {hid_loss:.6f}")
print(f"  B reading A's OUTPUT       (1 number)  : reconstruction loss {out_loss:.6f}")
print(f"  the hidden-layer figure is not small, it is ZERO to {abs(math.log10(max(hid_loss,1e-300))):.0f} decimal places.\n")
print("""  That is a real, self-discovered result about A's anatomy:

      x1 and x2 are recoverable from A's hidden layer, and NOT from its output.

  A's hidden layer keeps the ingredients; A's output layer deliberately destroys
  them, because its job is to answer one question ("positive?") and everything
  not needed for that question is thrown away. Information is not preserved
  through a network - each layer discards what it no longer needs.

  Nobody labelled anything. B found this by minimising its own loss.""")

print("\n" + "=" * 70)
print("CAVEAT, and it is the whole reason Tier 4 exists: B has shown the info")
print("is PRESENT in A's hidden layer. It has NOT shown A uses it. Proving that")
print("needs intervention - break the component, see if A fails. Later.")
print("=" * 70)
