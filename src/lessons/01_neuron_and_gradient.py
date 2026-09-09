"""
LESSON 1 - The neuron, and what a gradient actually is.

Run me:  python3 lessons/01_neuron_and_gradient.py

Big question of this lesson:
    A network "learns". Learns *how*? Nobody hands it the right answer for its
    weights. So how does a pile of numbers know which direction to move?
"""
import math

def rule(t=""):
    print("\n" + "=" * 68)
    if t: print(t); print("=" * 68)

# ---------------------------------------------------------------------------
# PART A - A neuron is three things: a weighted sum, a bias, a squash.
# ---------------------------------------------------------------------------
rule("PART A - the forward pass")

# Toy sentiment task. Two hand-made features from a movie review:
#   x1 = how many times the word "great" appears
#   x2 = how many times the word "terrible" appears
# Label y = 1 if the review is positive, 0 if negative.
#
# Real NLP never uses features this crude - but everything you learn about the
# *machinery* here is identical in a 100-billion-parameter transformer.

def sigmoid(z):
    """Squash any real number into (0, 1) so we can read it as a probability."""
    return 1.0 / (1.0 + math.exp(-z))

def neuron(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 + b     # (1) weighted sum  -> the "pre-activation"
    return sigmoid(z)             # (2) squash        -> the "activation"

# Weights start as arbitrary junk. The network knows nothing yet.
w1, w2, b = 0.30, -0.10, 0.00

x1, x2, y = 3.0, 0.0, 1.0     # "great great great" -> definitely positive
p = neuron(x1, x2, w1, w2, b)
print(f"review: great x{x1:.0f}, terrible x{x2:.0f}   true label y = {y}")
print(f"z  = w1*x1 + w2*x2 + b = {w1}*{x1} + {w2}*{x2} + {b} = {w1*x1+w2*x2+b:.4f}")
print(f"p  = sigmoid(z)        = {p:.4f}      <- the network's belief")

# ---------------------------------------------------------------------------
# PART B - Loss: turn "how wrong are we" into ONE number.
# ---------------------------------------------------------------------------
rule("PART B - the loss")

def loss_fn(p, y):
    return (p - y) ** 2       # squared error. Deliberately the WRONG choice
                              # for a probability - Lesson 4 shows what it costs.

L = loss_fn(p, y)
print(f"L = (p - y)^2 = ({p:.4f} - {y})^2 = {L:.4f}")
print("""
This is the whole game, stated precisely:
    L is a function of the weights.  L = L(w1, w2, b)
    x and y are FIXED (they're the data). The weights are the only free knobs.
    Training = find the knob settings that make L small.
""")

# ---------------------------------------------------------------------------
# PART C - Discover the derivative by poking it.
# ---------------------------------------------------------------------------
rule("PART C - what is a gradient? (poke the knob and watch)")

def total_loss(w1, w2, b, x1=x1, x2=x2, y=y):
    return loss_fn(neuron(x1, x2, w1, w2, b), y)

base = total_loss(w1, w2, b)
h = 1e-5
bumped = total_loss(w1 + h, w2, b)

print(f"L at w1 = {w1}          : {base:.10f}")
print(f"L at w1 = {w1+h}   : {bumped:.10f}   (nudged w1 up by h={h})")
print(f"change in L          : {bumped - base:+.10f}")
print(f"change in L per unit w1 : ({bumped:.8f} - {base:.8f}) / {h} = {(bumped-base)/h:+.6f}")
print("""
That last number IS the derivative dL/dw1. It is not a formula you memorize,
it is a MEASUREMENT: "if I push this knob up by 1, the loss responds by this."

  negative  -> pushing w1 UP makes loss go DOWN  -> so increase w1
  positive  -> pushing w1 UP makes loss go UP    -> so decrease w1

Either way: to reduce the loss, step OPPOSITE the derivative.
That single sentence is gradient descent. There is nothing else to it.
""")

def numerical_grad(w1, w2, b, h=1e-5):
    """dL/dw for each knob, by literally poking each one."""
    L0 = total_loss(w1, w2, b)
    return (
        (total_loss(w1 + h, w2, b) - L0) / h,
        (total_loss(w1, w2 + h, b) - L0) / h,
        (total_loss(w1, w2, b + h) - L0) / h,
    )

g = numerical_grad(w1, w2, b)
print(f"gradient = (dL/dw1, dL/dw2, dL/db) = ({g[0]:+.6f}, {g[1]:+.6f}, {g[2]:+.6f})")
print("dL/dw2 is 0.0 because x2 = 0: 'terrible' never appeared, so that knob")
print("could not have affected this prediction. The math knows that already.")

# ---------------------------------------------------------------------------
# PART D - Train it. Watch a thing learn for the first time.
# ---------------------------------------------------------------------------
rule("PART D - gradient descent")

w1, w2, b = 0.30, -0.10, 0.00
lr = 5.0     # learning rate: how big a step we take. More on this in Lesson 5.

print(f"{'step':>4} {'loss':>10} {'p':>8} {'w1':>8} {'w2':>8} {'b':>8}")
for step in range(41):
    gw1, gw2, gb = numerical_grad(w1, w2, b)
    if step % 5 == 0:
        p = neuron(x1, x2, w1, w2, b)
        print(f"{step:>4} {total_loss(w1,w2,b):>10.6f} {p:>8.4f} {w1:>8.4f} {w2:>8.4f} {b:>8.4f}")
    w1 -= lr * gw1        # <- step OPPOSITE the gradient
    w2 -= lr * gw2
    b  -= lr * gb

print(f"\nfinal belief p = {neuron(x1,x2,w1,w2,b):.4f}, wanted {y}. It learned.")

# ---------------------------------------------------------------------------
# PART E - Why this approach is doomed, and why backprop exists.
# ---------------------------------------------------------------------------
rule("PART E - the catastrophe hiding in Part D")

print("""
Count the work. numerical_grad() called total_loss() once per parameter,
plus one baseline. With P parameters that is P+1 forward passes for ONE step.

  our neuron   P = 3            ->            4 forward passes / step
  small MLP    P = 100,000      ->      100,001 forward passes / step
  GPT-2        P = 124,000,000  ->  124,000,001 forward passes / step

GPT-2 trained for ~1,000,000 steps. That is 1.2 x 10^14 forward passes.
At a generous 1000 passes/second that is roughly four thousand years.

So the entire field rests on one question:
    can we get ALL P derivatives from ONE forward pass and one cheap sweep back?

Yes. It's called backpropagation, it costs about 2x a forward pass REGARDLESS
of P, and it is just the chain rule applied to a graph. That is Lesson 2.
Below is a preview of why it works, on our single neuron.
""")

# ---------------------------------------------------------------------------
# PART F - The chain rule, by hand, verified against the measurement.
# ---------------------------------------------------------------------------
rule("PART F - the analytic gradient (the chain rule's first appearance)")

w1, w2, b = 0.30, -0.10, 0.00   # reset

# Forward, but SAVING every intermediate. This saving is the whole trick.
z = w1 * x1 + w2 * x2 + b
p = sigmoid(z)
L = (p - y) ** 2

# Now walk backwards. Each step asks: "how does MY output affect the loss?"
#
#   L <- p <- z <- w1
#
dL_dp  = 2 * (p - y)          # d/dp of (p-y)^2
dp_dz  = p * (1 - p)          # d/dz of sigmoid(z)  <- memorize this one, it's lovely
dz_dw1 = x1                   # d/dw1 of (w1*x1 + w2*x2 + b)
dz_dw2 = x2
dz_db  = 1.0

# Chain rule = multiply the local links along the path.
dL_dw1 = dL_dp * dp_dz * dz_dw1
dL_dw2 = dL_dp * dp_dz * dz_dw2
dL_db  = dL_dp * dp_dz * dz_db

num = numerical_grad(w1, w2, b)
print(f"{'':<10}{'analytic':>14}{'numerical':>14}{'abs err':>12}")
for name, a, n in [("dL/dw1", dL_dw1, num[0]),
                   ("dL/dw2", dL_dw2, num[1]),
                   ("dL/db",  dL_db,  num[2])]:
    print(f"{name:<10}{a:>14.8f}{n:>14.8f}{abs(a-n):>12.2e}")

print("""
They agree to ~7 decimals (the gap is just floating-point error in the poke).

Now look at what we spent: ONE forward pass, then three multiplications reusing
the SAME shared prefix (dL_dp * dp_dz). Adding a fourth weight would cost one
more multiply, not one more forward pass. The cost stopped depending on P.

That is the entire reason deep learning is possible.
""")
