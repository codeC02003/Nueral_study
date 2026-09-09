# CLAIMS — the authoritative record

**This file is the single source of truth for what this project claims.** Where any
other document disagrees, this one wins. Last updated after V5's abandonment; it
reflects the final state, not any intermediate one.

---

## A note on the word "mechanism"

Used precisely throughout, because the distinction is the whole point:

| term | meaning | established here? |
|---|---|---|
| **feature / direction** | a direction in activation space that B identified | yes |
| **causally important** | removing it measurably degrades A, more than a matched random direction does | **yes** |
| **mechanism** | a description that corresponds to the algorithm A actually implements | **no — untested** |

So the headline says *features and interventions*, not *mechanisms*. "Mechanism" is
reserved for the stronger thing V5 was built to test and did not test.

---

## THE HEADLINE CLAIM

> **B, given no human labels, finds causally important internal features and
> interventions roughly 10–25x more efficiently than random search, and predicts the
> effect of interventions it has never tried. Replicated prospectively on a second,
> independently trained model.**

---

## CLAIM 1 — B predicts held-out causal intervention effects

**Status: established.**

Ground truth is direct: we perform the intervention on frozen A and measure A's
actual loss. The target is observable, not inferred, so held-out interventions *are*
the ground truth for this claim. It needs no known-circuit benchmark.

- calibration slope ≈ 1.0 on held-out contexts of features never acquired
- of 8 model-layer cells, only one resolvably differs from slope 1.0
  (A₁ layer 0: 1.28, 95% CI [1.04, 1.57])
- a linear baseline scores R² = −0.245, so the learned structure is genuinely
  nonlinear rather than a fitted trend

**Why this is stronger than salience ranking, stated carefully:** accurate held-out
causal-effect prediction requires generalising *quantitative counterfactual effects*
to unseen features and contexts — including predicting near-zero for a direction A
carries but does not use. That is substantially more demanding than ranking which
directions matter. It does **not** follow that B represents A's algorithm: a
predictor may exploit statistical regularities of A's activations without any
correspondence to A's internal computation. Distinguishing those is exactly what V5
was for, and V5 did not run.

## CLAIM 2 — B discovers high-impact directions far more efficiently than random

**Status: established.**

- A₁ layer 2: **16.0x random**, 95% CI **[9.6, 24.5]**
- 30–40% of the oracle bound (the best any selector could reach at that budget),
  against random's 2.4%
- lower bound above 2.9x at every layer of both models

**Report the interval, not the point estimate.** "12.45x" appeared in earlier
documents; the defensible figure is **~10–25x**.

## CLAIM 3 — B's features are causally used by A, not merely correlated

**Status: established.**

- 3.50x more damage per unit removed than **magnitude-matched random directions**
  (Cohen's d = 1.19)
- 67% of a feature's damage falls on a single character, vs 8% for random directions

The matched control is what makes this a claim; deleting anything degrades a network.

**Scope:** this establishes that the directions are load-bearing. It does **not**
validate any semantic reading of them. "f515 → the `r` in `return`" describes where
the damage lands; it is not a verified account of A's computation.

## CLAIM 4 — this is a property of the method, not of one layer

**Status: established.**

Frozen configuration applied unchanged to all four layers. Oracle % mean 34.1,
range 29.6–39.4 (9.8 pp), CV 11.2%. Layer 2 — where the configuration was developed
— ranked third of four.

**Scope:** four layers is a small sample. The claim is "appears at every layer
tested", not "layer-invariant". Per-layer *differences* are **not resolvable**: every
per-layer CI overlaps between models.

## CLAIM 5 — this reproduces on a second independently trained model

**Status: replicated once, prospectively.**

Same architecture, corpus and hyperparameters; seed the only difference. Pass/fail
fixed in writing before A₂ existed (`05_PREREGISTRATION_A2.md`).

- A₂ four-layer mean oracle % **29.4** (preregistered band 26.1–42.1) — **PASS**
- minimum layer 26.4% (floor 15%) — **PASS**
- 19 of 20 A₁-vs-A₂ CI comparisons overlap

**Scope:** n = 2 is a clean preregistered replication, **not** population-level
generality. No statistical claim about generalisation across models is made or
supported.

**A prediction of ours that this falsified:** we predicted on record that calibration
would again degrade with depth and layer 3 would again fail. It did not. Bootstrap
CIs later showed the original A₁ observation had no evidential basis either
(slope 0.80, CI [0.13, 1.51]).

## CLAIM 6 — B's decomposition corresponds to A's actual algorithm

**Status: UNTESTED. Not claimed, and not rejected.**

V5 built a known-circuit benchmark to test exactly this. Across three attempts the
subject model never produced the canonical circuit the test scores against, so its
validity precondition never held and **B was never run against it**. Abandoned per
its own protocol after three attempts.

The hypothesis is therefore **untested**, not falsified. See
[`../milestones/v5/`](../milestones/v5/) and
[`../findings/07_V5_ABANDONED.md`](../findings/07_V5_ABANDONED.md).

---

## Uncertainty

All intervals are **three-level hierarchical bootstraps**, 5,000 reps: resample
features (the cluster), then contexts within each, then the acquisition seed. Rows
sharing a feature are not independent, and the reported figures are means over
acquisition seeds — a naive row-level or single-trajectory bootstrap understates the
uncertainty in both respects.

Interventions are deterministic (repeating one differs by exactly `0.00e+00`), which
removes repeated-measurement noise but **not** sampling uncertainty over which
features and contexts enter the pool.

**Limitation:** the seed component is estimated from only 3 acquisition seeds.

---

## Claims we retracted

Both are preserved in the project history rather than removed.

1. **"Calibration degrades with depth; layer 3 fails."** Bootstrap slope for A₁
   layer 3 is 0.80, CI **[0.13, 1.51]** — the widest in the grid. No evidential
   basis at any point.
2. **"Per-layer detail does not replicate."** All per-layer CIs overlap. The
   differences are **not resolvable at this sample size**, which is weaker than
   either "replicates" or "fails to replicate".

---

## Honest limitations on the whole package

- **`mag80 + log` was selected post-hoc on A₁.** The A₂ preregistration covers A₂'s
  *evaluation* only; it does not make the project preregistered.
- **No related-work comparison has been done.** No novelty claim should be made
  until it is.
- **Two training-flag bugs** overshot a preregistered target during A₂; both were
  caught before any metric was computed, both runs discarded, both logged.

---

## What is open

| | |
|---|---|
| **Mechanistic correspondence** | Claim 6. Needs a ground-truth instrument that does not assume the canonical decomposition — see the V6 sketch in `../findings/07_V5_ABANDONED.md`. |
| **Population generality** | needs many more independently trained models than 2. |
| **Coverage** | B reads one residual stream per model. It has never read A's weights, attention heads, or MLP neurons directly. |
| **Related work** | required before any novelty statement. |

---

## Document status

| document | status |
|---|---|
| **this file** | **authoritative, current** |
| `../README.md` | current summary |
| `00_START_HERE.md` | current, plain-language |
| `02_PROJECT_HISTORY.md` | historical record — preserves superseded claims *as history* |
| `01_OVERVIEW.md`, `07_OPEN_ITEMS.md` | **HISTORICAL** — reflect V2/V3 state, banner at top |
| `04_METRICS.md` | current for metric definitions and results |
| `../findings/*` | dated results; each true as of its date |
| `../milestones/v{3,4,5}/` | frozen; see `v4/ERRATA.md` for a terminology correction |
