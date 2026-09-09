# RUNBOOK — how to see the networks running

Everything runs from the repo root with plain `python3`. No install, no venv, no GPU.

```bash
cd ~/nueral
```

---

## 0. CONTINUOUS MODE — A learns Python forever, B reads A while it learns

```bash
./src/run_all.sh
```

Starts three things: A training on 9.2M characters of real Python, B reading A's
insides, and the live dashboard. Both networks resume from where they left off,
so you can stop and restart freely.

```bash
python3 src/dashboard.py          # reattach to the live view any time
python3 src/dashboard.py --once   # one frame, then exit
tail -f logs/a.log            # A's raw output
tail -f logs/b.log            # B's raw output
pkill -f train_a.py           # stop A
pkill -f train_b.py           # stop B
```

Or run the pieces yourself:

```bash
python3 src/train_a.py                 # forever. Ctrl-C stops; rerun resumes.
python3 src/train_a.py --steps 5000    # bounded
python3 src/train_b.py --layer 2       # which of A's 4 layers B dissects
```

### What to watch

| Field | Healthy | Meaning |
|---|---|---|
| A `loss` | falling, `4.56 → <1.5` | 4.56 = `ln(96)` = random guessing over the vocab |
| A `bits/char` | falling below 2.0 | compression rate. Real Python is ~1.5 bits/char for a good model. |
| B `recon err` | `< 20%` | how much of A's activation B fails to explain |
| B `active/token` | `10–50` of 1024 | **the sparsity that defeats superposition.** 300+ means B is cheating. |
| B `dead` | low | features that never fire — wasted capacity |
| `A is N steps ahead` | any | B chases a moving target. That is intentional. |

### Where the results land

```
samples/a_step*.py        what A could write at each stage — read these in order
findings/features.json    B's readings: each feature + the text that triggers it
checkpoints/a_step*.pt    A's full development history, for studying how it LEARNED
```

### Honest expectations

A is 820K parameters on a laptop. It will learn Python's **shape** — indentation,
`def`/`self`/`return`, matching brackets, string literals, the look of a class body —
and will produce syntactically plausible nonsense. It will not write working code you
ask for; that needs ~10,000x more compute. A small A is the right choice anyway,
because B can actually be verified against it.

Rough milestones on this machine (~8 it/s):

| A step | ~time | bits/char | what A can do |
|---|---|---|---|
| 500 | 1 min | 3.2 | letter frequencies, some `def`/`self` fragments |
| 5,000 | 10 min | ~2.0 | real keywords, indentation, matched quotes |
| 25,000 | 1 hour | ~1.7 | plausible function bodies, consistent naming |
| 100,000+ | 4 hours | ~1.5 | docstrings, class structure, argument patterns |

---

## 1. The lesson demos — watch the small networks train, live

```bash
python3 src/watch.py
```

Runs about 30 seconds and animates in place. Two phases:

**Phase 1 — Network A learns negation.** Watch the four bars separate. At step 0 all
four sit near 0.50 (the network is guessing). By step ~120 they split into two pairs.
Watch the `hidden layer` block at the bottom: those 4 numbers are the features A is
*inventing for itself*, and you can see them move.

**Phase 2 — Network B learns to read A.** B is shown only A's 4 hidden numbers and has
to recover the original input. It never sees the input. Watch `B decodes` snap onto
`truth` around step ~300.

### Speed control

| Command | Delay | Use for |
|---|---|---|
| `python3 src/watch.py` | 45 ms/frame | normal watching |
| `python3 src/watch.py slow` | 150 ms/frame | actually studying the hidden layer |
| `python3 src/watch.py fast` | none | just want the result |

Piping to a file or `less` disables the animation and prints checkpoints instead:
```bash
python3 src/watch.py > run.log        # snapshots every 100 steps, no escape codes
```

### What "good" looks like

```
A  final loss 0.014184   solved negation with 17 parameters
B  final loss 0.000000   recovered A's inputs from A's hidden layer alone
```

---

## 2. Neural Anatomy — every neuron of both networks, running locally

```bash
./src/serve.sh
```

Snapshots both live networks, rebuilds the page, serves it at
**http://localhost:8777/anatomy.html** and opens your browser. Nothing leaves this
machine — the only outbound request is the Google webfont, and it degrades fine
without it. Set `PORT=9000 ./src/serve.sh` to use another port.

### Which is which

Two separate bordered regions, and they never mix:

- **NETWORK A** — cyan border, top. The subject. 2,784 neurons.
- **NETWORK B** — gold border, bottom. The anatomist. 1,024 feature neurons.
- Between them, the gold arrow marks **where B taps A**: layer 2's residual stream,
  128 numbers per position, detached so B cannot alter A.

### Where the neurons are

Every cell in every strip is **one real neuron**, coloured by its measured
activation at the scanning position. Cyan = positive, gold = negative.
**Hover any cell** and it names itself.

Read A bottom to top — that is the order data actually flows:

| Strip | Count | What these neurons are |
|---|---|---|
| `EMBED` | 128 | the character's vector, plus its position |
| `L0 ATTN` | 4 heads × 32 | each row is one head; brightness = how hard it reads that position |
| `L0 MLP` | 512 | the layer's feed-forward neurons — where features get computed |
| `L0 RESID` | 128 | the residual stream leaving layer 0 — the network's shared memory bus |
| `L1…L3` | same | the same three populations, three more times |
| `OUTPUT` | 96 | one neuron per vocabulary character. The bright one is A's prediction. |

B's single strip is its 1,024 features. Only ~13 fire at any position; the caption
counts them. That sparsity is the entire point — it is what defeating superposition
looks like.

### Controls

`Space` pause · `←` `→` step one position · click any token · `L0H0`…`L3H3` switches
which head the attention matrix shows. In that matrix, **row = the position doing the
reading, column = the position being read**; the lower triangle is the causal mask,
and gold marks the current row.

---

## 3. The backward-pass instrument

```bash
open viz/backprop_microscope.html
```

Runs straight from the file — the autograd engine is reimplemented in JavaScript
inside the page, so it is a live instrument, not a recording.

Two things to actually do:

1. Press **Play sweep**. Watch the gradient wave travel right-to-left. Every node sits
   at `∇ 0` until the wave physically reaches it — that is backpropagation, visible.
2. Click **"set confidently wrong · w = −3"**, then toggle the loss function.
   - squared error → `dL/dw ≈ −0.0007` (the bar is invisible)
   - cross-entropy → `dL/dw ≈ −3.0`

   Same model, same wrongness, **4,000× the signal**. That is Lesson 2 in one gesture.



---

## 4. The lessons — run in order

```bash
python3 src/lessons/01_neuron_and_gradient.py     # ~1s
python3 src/lessons/02_backprop.py                # ~2s
python3 src/lessons/03_first_network.py           # ~15s
python3 src/lessons/04_tensors.py                 # ~3s
python3 src/lessons/demo_superposition.py         # ~20s  (uses PyTorch)
```

These print and exit — they are teaching transcripts, not live views. Read them
alongside the source; the source is commented more heavily than the output.

### The numbers to check in each

| Script | Look for | Should be | Means |
|---|---|---|---|
| `01` | Part F, `abs err` column | `~1e-06` | hand-derived chain rule matches the measured derivative |
| `02` | Part E, `err vs torch` | `0.00e+00` to `4e-16` | **your engine matches PyTorch to machine precision** |
| `02` | Part C, buggy vs correct | `+3 / +5` vs `+8` | `=` instead of `+=` gives order-dependent wrong gradients |
| `02` | Part F, ratio | `4,053x` | cross-entropy vs squared error when maximally wrong |
| `03` | Part A, solo loss | `0.69315` | `= ln(2)` — exactly a coin flip. One neuron cannot do XOR. |
| `03` | Part B, correct | `4/4` | one hidden layer solves it |
| `03` | Part D, control | hidden `0.000000`, output `0.28` | info survives in A's hidden layer, destroyed at its output |
| `04` | Part A, speedup | `~95x` | why we vectorized |
| `04` | Part D, all four rows | `shapes match` | tensor engine matches PyTorch in value *and* shape |

---

## 5. Poke at it yourself

Open a REPL and drive the engine directly:

```bash
python3
```
```python
>>> from nn.engine import Value
>>> a = Value(2.0, label="a"); b = Value(-3.0, label="b")
>>> L = (a * b + a).tanh()
>>> L.backward()
>>> a.grad, b.grad
(-0.002681901366051731, 0.002681901366051731)
```

Train your own network in six lines:

```python
>>> from nn.mlp import MLP
>>> from nn.engine import Value
>>> net = MLP(2, [4, 1])                       # 2 inputs -> 4 hidden -> 1 out
>>> out = net([Value(1.0), Value(0.0)])
>>> out.backward()
>>> len(net.parameters())
17
>>> [round(p.grad, 4) for p in net.parameters()][:3]
[0.2539, 0.0, 0.2539]
```

That middle `0.0` is not a bug — it is the second input weight, and the second
input was `0.0`, so that weight could not have affected the output. A zero
gradient means "you were not responsible." (Lesson 1, Part C.)

**Two rules that will bite you:**
- Call `net.zero_grad()` before every `backward()`. Gradients **accumulate** — forget
  this and step 50 gets the sum of all 50 previous steps' gradients.
- When feeding one network's activations into another, wrap them in fresh `Value`s
  (`Value(h.data)`). Otherwise B's gradients flow into A and B rewrites its own subject.

---

## 6. Regenerate the visualizer data

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from nn.engine import Value
from nn.graphviz_export import export
x,w,b,y = Value(3,label='x'),Value(.3,label='w'),Value(0,label='b'),Value(1,label='y')
wx = w*x; wx.label='w*x'
z = wx+b; z.label='z'
p = z.sigmoid(); p.label='p'
L = (p-y)**2; L.label='L'
print(export(L, 'viz_data.json')['frames'][-1]['grads'])
"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nn` | not in the repo root | `cd ~/nueral` first |
| `watch.py` shows garbage like `[H[J` | terminal ignoring ANSI | `python3 src/watch.py fast`, or pipe to a file |
| boxes/blocks render as `?` | font lacks block glyphs | any modern terminal font fixes it; output is still correct |
| `03` feels slow | scalar engine — one Python object per number | expected. Lesson 04 vectorizes it with numpy. |
| `demo_superposition.py` fails | needs PyTorch | `python3 -c "import torch"` to check |

---

## Map of the repo

```
nueral/
├── run_all.sh                  ← START HERE. A + B + dashboard, continuous.
├── train_a.py                  ← Network A: 820K-param char transformer on Python
├── train_b.py                  ← Network B: sparse autoencoder reading A
├── dashboard.py                ← live view of both
├── watch.py                    ← the small lesson networks, animated
├── 09_RUNBOOK.md                  ← this file
├── README.md                   ← the syllabus
├── 02_PROJECT_HISTORY.md                  ← Network A + B spec, the self-learning rule, B's 5 tiers
├── nn/
│   ├── gpt.py                  ← Network A's architecture, instrumented for B
│   ├── sae.py                  ← Network B. Includes the calibrated L1 table.
│   ├── tensor.py               ← numpy autograd engine, 95x faster than scalar
│   ├── layers.py               ← Linear / MLP on the tensor engine
│   ├── engine.py               ← scalar autograd. ~90 lines. Matches PyTorch to 1e-16.
│   ├── mlp.py                  ← Neuron / Layer / MLP built on the engine
│   └── graphviz_export.py      ← dumps a graph + its backward sweep for the visualizer
├── lessons/
│   ├── 01_neuron_and_gradient.py
│   ├── 02_backprop.py
│   ├── 03_first_network.py     ← first real network + B v0 reading it
│   └── demo_superposition.py   ← why "read neuron 3" is a broken plan
└── viz/
    ├── backprop_microscope.html   ← the backward-pass instrument
    ├── hud_template.html          ← the HUD's source (data injected at build)
    ├── build_hud.py               ← template + snapshot -> hud.html
    ├── export_live.py             ← snapshots both live networks to JSON
    └── refresh_hud.sh             ← one command: re-snapshot + rebuild
```

## Current state

Everything below reflects the finished project. Authoritative claims:
[`03_CLAIMS.md`](03_CLAIMS.md).

| | |
|---|---|
| **A₁** | frozen, step 83,750, 1.28 bits/char |
| **A₂** | frozen, step 83,750, the preregistered replication |
| **B** | sparse autoencoders trained on all four layers of both models |
| **causal test** | done — 3.50x vs matched random controls |
| **prediction** | done — calibration slope ≈ 1.0 on held-out features |
| **experiment selection** | done — ~10-25x more efficient than random |
| **uncertainty** | done — three-level hierarchical bootstrap, 5,000 reps |
| **ground-truth benchmark** | **abandoned** — instrument invalid, B never run. `../milestones/v5/` |

Still not built: B reading A's **weights** (only activations so far), cross-layer
feature identity, and any validated claim that B's decomposition matches A's
algorithm. Deferred items with trigger conditions: [`08_NOT_NOW.md`](08_NOT_NOW.md).

