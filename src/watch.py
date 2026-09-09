#!/usr/bin/env python3
"""
Watch the networks train, live.

    python3 watch.py            # normal speed, ~30s
    python3 watch.py fast       # no delay, as fast as your CPU
    python3 watch.py slow       # slow enough to actually study

Phase 1: Network A learns negation ("not good" is negative, "not bad" is positive).
Phase 2: Network B learns to read A's hidden layer. Nobody tells B what to look for.
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn.engine import Value
from nn.mlp import MLP

SPEED = (sys.argv[1] if len(sys.argv) > 1 else "normal").lower()
DELAY = {"fast": 0.0, "normal": 0.045, "slow": 0.15}.get(SPEED, 0.045)
TTY = sys.stdout.isatty()

C = dict(g="\033[32m", r="\033[31m", d="\033[2m", b="\033[1m",
         c="\033[36m", y="\033[33m", m="\033[35m", x="\033[0m")
if not TTY:
    C = {k: "" for k in C}

SPARK = "▁▂▃▄▅▆▇█"
DATA = [((1, 0), 1.0, '"good"'),
        ((1, 1), 0.0, '"not good"'),
        ((0, 0), 0.0, '"bad"'),
        ((0, 1), 1.0, '"not bad"')]


def bar(v, w=22):
    n = max(0, min(w, int(round(v * w))))
    return "█" * n + C["d"] + "░" * (w - n) + C["x"]


def spark(vals, w=46):
    if len(vals) < 2:
        return ""
    step = max(1, len(vals) // w)
    v = vals[::step][-w:]
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1e-9
    return "".join(SPARK[min(7, int((x - lo) / rng * 7.99))] for x in v)


def frame(lines):
    if TTY:
        sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
        sys.stdout.flush()
        time.sleep(DELAY)
    else:
        print("\n".join(lines))


def bce(logit, y):
    p = logit.sigmoid()
    return -((p.log() * y) + ((1 - p).log() * (1 - y)))


# ===========================================================================
# PHASE 1 - Network A
# ===========================================================================
random.seed(1)
A = MLP(2, [4, 1])
STEPS_A = 400
hist_a, hidden_snap = [], []

for step in range(STEPS_A + 1):
    loss = sum((bce(A([Value(v) for v in x]), y) for x, y, _ in DATA), Value(0.0))
    loss = loss * (1.0 / len(DATA))
    A.zero_grad(); loss.backward()
    for p in A.parameters():
        p.data -= 0.5 * p.grad
    hist_a.append(loss.data)

    if step % (1 if TTY else 100) == 0:
        preds = [(name, A([Value(v) for v in x]).sigmoid().data, y) for x, y, name in DATA]
        acc = sum((p > .5) == (y > .5) for _, p, y in preds)
        L = []
        L.append(f"{C['b']}{C['c']}NETWORK A{C['x']}  {C['d']}learning negation — a task one neuron provably cannot do{C['x']}")
        L.append("")
        L.append(f"  step {C['b']}{step:>4}{C['x']} / {STEPS_A}      "
                 f"loss {C['y']}{loss.data:.6f}{C['x']}      "
                 f"correct {C['g' if acc == 4 else 'r']}{acc}/4{C['x']}")
        L.append("")
        L.append(f"  {C['d']}loss{C['x']}  {C['y']}{spark(hist_a)}{C['x']}  {C['d']}{hist_a[0]:.2f} → {loss.data:.4f}{C['x']}")
        L.append("")
        for name, p, y in preds:
            ok = (p > .5) == (y > .5)
            mark = f"{C['g']}✓{C['x']}" if ok else f"{C['r']}✗{C['x']}"
            L.append(f"   {name:<12} {bar(p)} {p:.4f}   target {y:.0f}  {mark}")
        L.append("")
        L.append(f"  {C['d']}hidden layer — the 4 features A is inventing for itself{C['x']}")
        for (x, y, name) in DATA:
            h = "  ".join(f"{v.data:+.3f}" for v in A.hidden(x))
            L.append(f"   {C['d']}{name:<12}{C['x']} [ {h} ]")
        frame(L)

hidden_snap = list(L)

# ===========================================================================
# PHASE 2 - Network B reads Network A
# ===========================================================================
def acts(model, x):
    """DETACHED copies. B observes A; B must not rewrite A."""
    return [Value(h.data) for h in model.hidden(x)]

random.seed(7)
B = MLP(4, [6, 2])
STEPS_B = 1200
hist_b = []

for step in range(STEPS_B + 1):
    loss = Value(0.0)
    for x, y, _ in DATA:
        pred = B(acts(A, x))
        loss = loss + (pred[0] - x[0]) ** 2 + (pred[1] - x[1]) ** 2
    loss = loss * (1.0 / len(DATA))
    B.zero_grad(); loss.backward()
    for p in B.parameters():
        p.data -= 0.05 * p.grad
    hist_b.append(loss.data)

    if step % (3 if TTY else 300) == 0:
        L = []
        L.append(f"{C['b']}{C['m']}NETWORK B{C['x']}  {C['d']}reading A's hidden layer. Nobody told it what to look for.{C['x']}")
        L.append("")
        L.append(f"  {C['d']}B sees ONLY A's 4 hidden numbers. Its job: recover the original input.{C['x']}")
        L.append(f"  {C['d']}It never sees the input. If it succeeds, that info survives inside A.{C['x']}")
        L.append("")
        L.append(f"  step {C['b']}{step:>4}{C['x']} / {STEPS_B}      "
                 f"reconstruction loss {C['m']}{loss.data:.6f}{C['x']}")
        L.append("")
        L.append(f"  {C['d']}loss{C['x']}  {C['m']}{spark(hist_b)}{C['x']}  {C['d']}{hist_b[0]:.2f} → {loss.data:.6f}{C['x']}")
        L.append("")
        L.append(f"   {C['d']}{'phrase':<12}{'A is thinking':<34}{'B decodes':<18}truth{C['x']}")
        for x, y, name in DATA:
            h = " ".join(f"{v.data:+.2f}" for v in A.hidden(x))
            pr = B(acts(A, x))
            near = abs(pr[0].data - x[0]) < .15 and abs(pr[1].data - x[1]) < .15
            mk = f"{C['g']}✓{C['x']}" if near else f"{C['d']}·{C['x']}"
            L.append(f"   {name:<12} [{h}]   "
                     f"({pr[0].data:+.2f}, {pr[1].data:+.2f})  {mk}   ({x[0]}, {x[1]})")
        frame(L)

# ===========================================================================
print(f"\n{C['b']}RESULT{C['x']}")
print(f"  A  final loss {hist_a[-1]:.6f}   solved negation with {len(A.parameters())} parameters")
print(f"  B  final loss {hist_b[-1]:.6f}   recovered A's inputs from A's hidden layer alone")
print(f"\n{C['d']}  Neither was told what to learn. A minimised its loss; B minimised its own.")
print(f"  B has shown the information is PRESENT in A. Not that A USES it —")
print(f"  proving that needs intervention, which is Tier 4 in PROJECT.md.{C['x']}\n")
