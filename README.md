# Neural Anatomy

Two neural networks. **A** learns to write Python. **B** learns *A* — with no human
labels, its only training signal coming from A's own loss.

## The standing claim

> **B finds a frozen language model's causally important mechanisms roughly 10–25x
> faster than random search, calibrated to within the resolution of the data.
> Replicated on a second, independently trained model.**

## What is explicitly not claimed

- **That B has discovered A's actual algorithm.** Untested. The ground-truth
  benchmark that would test it was abandoned when its own validity condition failed
  — see `milestones/v5/`.
- **Population-level generality.** n = 2 models: a clean preregistered replication,
  not statistical power over models.
- **Any depth trend** in discovery or calibration. Not resolvable at this sample size.
- **Any semantic reading of individual features.** Matched controls show the
  directions are causally load-bearing; they say nothing about what they *mean*.

## Evidence, at a glance

| | |
|---|---|
| B's features are causally used by A | 3.50x more damage than magnitude-matched random directions, Cohen d = 1.19 |
| damage is specialised | 67% of a feature's damage on one character, vs 8% for random |
| B predicts unseen interventions | calibration slope ≈ 1.0; 1 of 8 model-layer cells resolvably differs |
| B beats random search | A₁ layer 2: 16.0x [9.6, 24.5]; 30–40% of the oracle bound at 0.65% budget |
| holds at every layer | oracle % 29.6–39.4 across four layers |
| holds on a second model | A₂ preregistered: mean 29.4%, band 26.1–42.1, **PASS** |

All intervals are three-level hierarchical bootstraps (features, contexts,
acquisition seed).

## Read in this order

| | |
|---|---|
| [`docs/01_OVERVIEW.md`](docs/01_OVERVIEW.md) | the idea, the fundamentals it rests on, what it consumed |
| [`docs/02_PROJECT_HISTORY.md`](docs/02_PROJECT_HISTORY.md) | all 13 phases, **including every failure and retraction** |
| [`docs/03_CLAIMS.md`](docs/03_CLAIMS.md) | four claims, separately graded |
| [`docs/04_METRICS.md`](docs/04_METRICS.md) | both metrics in full, including the definition that failed |
| [`docs/05_PREREGISTRATION_A2.md`](docs/05_PREREGISTRATION_A2.md) | the replication protocol + its deviations |
| [`docs/06_PREREGISTRATION_V5.md`](docs/06_PREREGISTRATION_V5.md) | the ground-truth protocol, 5 revisions, all pre-training |
| [`docs/07_OPEN_ITEMS.md`](docs/07_OPEN_ITEMS.md) | what is still weak and what to try |
| [`docs/08_NOT_NOW.md`](docs/08_NOT_NOW.md) | deferred work, each with a trigger condition |
| [`docs/09_RUNBOOK.md`](docs/09_RUNBOOK.md) | how to run everything |

Results in chronological order live in [`findings/`](findings/), `01_` through `07_`.

## Milestones — frozen, do not edit

```
milestones/v3/   the original 12.45x discovery result
milestones/v4/   cross-model replication + bootstrap CIs   <- the standing claim
milestones/v5/   ground-truth benchmark ABANDONED / instrument invalid
```

Each carries a `MANIFEST.json` with its headline numbers, scope limits and
reproduction commands.

## Layout

```
docs/            the written record, numbered in reading order
findings/        results, numbered chronologically, with raw JSON
src/
  nn/            the engine, written from scratch: scalar autograd -> tensors -> GPT -> SAE
  lessons/       01-04, the teaching sequence that built nn/
  train_*.py     A, B, and the induction subject
  harvest.py     builds the (feature, context) causal dataset
  ablate.py      the causal test with matched random controls
  v2_*, v3_*     the discovery experiments
  validate_layer.py, bootstrap_*.py
  dashboard.py, watch.py, export_neurons.py, build_anatomy.py
milestones/      frozen results
data/            the corpus (regenerable: python3 src/build_corpus.py)
checkpoints/     A₁, A₂, A_ind, and B's SAEs
experiments/     the 452,219-row causal dataset (regenerable)
viz/             rendered visualisations
```

## Quick start

```bash
python3 src/lessons/02_backprop.py      # the autograd engine, verified against PyTorch
python3 src/lessons/03_first_network.py # first network + B reading it
./src/run_all.sh                        # train A and B live, with a dashboard
python3 src/ablate.py                   # the causal result, with controls
```

## A note on how this is written

The docs record the mistakes: two retracted claims, a metric definition that
produced a negative result, a scoring bug that read the wrong array index, and
training flags that overshot a preregistered target twice. That is deliberate. The
replication is only worth anything because the failures are in the record next to it.
