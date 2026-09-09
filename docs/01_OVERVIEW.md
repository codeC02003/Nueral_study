# A neural network that dissects another neural network

> **HISTORICAL DOCUMENT — reflects the project state as of V2/V3, before the A₂ replication and the bootstrap CIs. Do not use for
> current claims.** Kept as a record of what was believed and planned at the time.
> The authoritative claims are in [`03_CLAIMS.md`](03_CLAIMS.md).


## The idea

Two networks. **A** learns a task. **B** learns *A*.

A is a small transformer trained to write Python. B never sees Python — it only
sees A's insides, and has to work out how A does what it does. The name for this
is **mechanistic interpretability**, and the field borrows its metaphors from
anatomy: dissecting circuits, lesion studies, the *biology* of a model.

The governing rule is that **both networks are self-learning.** Neither is told
what to know. A minimises its prediction error; B minimises its own loss. Nobody
labels a single feature. If a human has to tell B what to look for, B isn't
learning — the human is, and B is just plumbing.

## The fundamentals it rests on

**1. A gradient is a measured sensitivity, not a formula.** "If I nudge this
weight up by one, the loss responds by this much." Step opposite it and the loss
falls. That is all of training.

**2. Backpropagation makes it affordable.** Measuring each weight's gradient by
poking it costs one forward pass per weight — 124 million passes per step for a
GPT-2-sized model, roughly four thousand years of training. Backprop gets *every*
gradient from one forward pass and one backward sweep, because sensitivity
multiplies along a path and paths share prefixes. Cost stops depending on
parameter count. Without this, none of deep learning exists.

**3. Depth exists to bend space.** One neuron can only draw a straight line, so it
cannot learn that "not good" is negative while "good" is positive — the answer is
the XOR of the two inputs, and no line separates those corners. A hidden layer
re-describes the inputs in coordinates where a line *does* work. That
re-description is what "learning a representation" means.

**4. The unit of meaning is a direction, not a neuron.** This is the one that
decides whether the project works. Networks store more features than they have
neurons, as *overlapping directions* — accepting interference in exchange for
capacity. It's called **superposition**, and it means individual neurons are
polysemantic: one neuron fires for legal language, the letter Q, and DNA
sequences. Verified here in a toy: a 5-neuron layer reliably stored 20 features.
So B cannot "read neuron 3." It has to find directions.

**5. B's method: an overcomplete sparse basis.** B is a sparse autoencoder — 128
inputs, 1,024 features, but only ~15 allowed to fire at once. Its loss is
reconstruction error plus a sparsity penalty, and nothing else. Since superposition
is compression *into* a small dense space, forcing a large sparse one runs the
compression backwards.

**6. Correlation is not causation, and this is the field's main failure mode.**
B can decode information from a layer at 100% accuracy while A completely ignores
that information downstream. Proving A *uses* something requires intervention:
break the component, see if A fails. That step is not built yet, and everything
below the line marked "verified" depends on it.

## What is actually running

| | Network A — the subject | Network B — the anatomist |
|---|---|---|
| what it is | 4-layer character transformer | sparse autoencoder |
| parameters | 819,968 | 263,296 |
| neurons | 2,784 | 1,024 features |
| trained on | 9.2M chars of Python stdlib | A's layer-2 residual stream |
| supervision | next character | reconstruct A's activations, sparsely |
| steps so far | 83,850 | 362,921 |

Also built from scratch, verified against PyTorch to machine precision
(`4.44e-16`): a scalar autograd engine, then a tensor version 95x faster. The
transformer uses PyTorch, but nothing in it is unexamined.

## Results so far

A went from letter soup to structurally valid Python:

```
step    250   def y areewaretin u)   dele    suob.e()
step 56,500   def connection(self):    return self._internal_connec
```

Correctly spelled identifiers, closed quotes and parens, indented method bodies,
plausible private-attribute naming. **1.28 bits per character** — better
compression than gzip on Python source, which means it learned structure, not
letter frequency.

B, with no labels, independently found features that fire on:
exception class names mid-word (`AttributeE`, `TypeE`), the position right before
`None`, assignment operators, closing parens of call signatures, `self` as a first
parameter, byte-string literals, and runs of repeated characters.

One measured result about A's anatomy: **A's attention stops reorganising after
~44,000 steps.** Circuit drift falls monotonically (0.0027 -> 0.0015) and then
flattens. Everything after that is weight refinement inside a fixed circuit. So a
system meant to watch a circuit *form* has to be watching in the first ~15,000
steps.

## Limitations — what this cannot do

**A will not write code you ask for.** 820K parameters on a laptop learns Python's
*shape*, not its meaning. It produces `msg = msg` three times in a row and
`def commands(self, stop_ is not None)`. Working code needs roughly four orders of
magnitude more compute. A small A is the right choice anyway, because B can
actually be checked against it.

**B's features are verified causal; B's *understanding* is not.** Ablation now
shows B's features do 3.5x more damage per unit removed than matched random
directions (Cohen's d = 1.19), with 67% of each feature's damage landing on a
single character versus 8% for random. Specific circuits are identified: f515
predicts the `r` in `return`, f384 completes `None`, f336 the dot after `self`.
See `findings/TIER4_STEP1.md`.

But **B did not find this - the ablations were run by hand.** B has no prediction
head and was not scored. That is evidence about A's anatomy, not evidence that B
understands A. And the circuits are identified, not traced: we know f515 is
required for `return`, not which heads and neurons implement it.

**B only reads one layer, one way.** It sees layer 2's residual stream and nothing
else — not the weights, not the attention heads, not the MLPs. Of the five planned
tiers, one is built.

**B can now intervene, and predict.** Ablation shows B's features do 3.5x more
damage than matched random directions, and B predicts the damage for features it has
never ablated at R2 = 0.33 (Pearson r = 0.58). It systematically under-predicts the
largest effects, so it knows *which* directions matter better than *how much*.
See findings/TIER4_STEP1.md and TIER4_STEP3.md.

**No hypothesis loop.** The design calls for B to form a hypothesis, design a test,
run it, and update. B currently minimises a reconstruction loss. That is a long way
from doing science.

**The visualisation is a snapshot,** not a live feed. A training process writes
checkpoints to disk; a browser cannot watch a `.pt` file change. Live numbers come
from the terminal dashboard.

## What it consumed

**Compute** — 20.2 hours wall clock on an M-series laptop GPU (MPS), both networks
running concurrently. A averaged 1.2 steps/s with B competing for the same GPU
(~7 steps/s alone). No cloud, no API calls, no external service. Everything ran
locally.

**Data** — 9.2M characters from 410 Python files already on the machine. A processed
343 million characters, about 37 passes over the corpus. Nothing was downloaded.

**Disk — 219 MB total**, after pruning from 1.2 GB:

| | |
|---|---|
| checkpoints (48 snapshots + 2 working + cache) | 197 MB |
| corpus (text + encoded) | 18 MB |
| training logs | 1.8 MB |
| visualisations | 1.4 MB |
| generated code samples (335 files) | 0.1 MB |
| the engines and lessons | 0.1 MB |

The 48 snapshots are not spares: they are A's development history, kept by
measuring where A's attention actually changed rather than at fixed intervals.
287 redundant ones were deleted. Left unmanaged, checkpoints grow ~1 GB/hour.

**Honest note on cost of the approach:** B ran 4.3x more steps than A and produced
findings that are still unverified. That ratio is normal for interpretability —
understanding a network is more expensive than training one — but it is worth
stating plainly rather than implying the analysis is cheap.
