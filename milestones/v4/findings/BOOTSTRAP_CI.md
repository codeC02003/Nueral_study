# Uncertainty quantification — three-level hierarchical bootstrap

5,000 reps · both models · all four layers

## Method

Rows are **not** independent draws: every context of feature 515 shares that
feature's decoder direction, firing statistics and logit reach. A row-level
bootstrap would treat ~300 correlated observations as 300 independent ones. And the
reported figures are means over 3 acquisition seeds, so an interval conditional on
one trajectory is an interval on a *different estimator*. Three levels:

```
1. resample FEATURES with replacement                (the cluster)
2. resample CONTEXTS within each sampled feature
3. resample the ACQUISITION SEED with replacement    (matches the reported estimator)
```

The **oracle bound travels with the resample** — recomputed each rep as the top-N
rows by |Δloss| inside that resampled pool, at the same N.

**Stated limitation:** with only 3 seeds, the seed-variance component is itself
crudely estimated. Better than excluding it; not as good as 10 seeds.

## Adding acquisition-seed variance changed one family of metrics, not the other

| metric | layer | 2-level width | 3-level width | ratio |
|---|---|---|---|---|
| calib slope | 2 | 0.460 | 1.034 | **2.2x** |
| calib slope | 3 | 0.490 | 1.385 | **2.8x** |
| oracle % | 0-3 | 0.13-0.16 | 0.14-0.18 | ~1.0x |
| × random | 0-3 | 3.7-17.0 | 4.5-17.5 | ~1.0x |

**Discovery metrics are robust to which acquisition trajectory was run. Calibration
metrics are not** — they depend on the specific fitted model, which depends heavily
on which rows happened to be acquired. This is a substantive finding, not
bookkeeping: it is why the earlier two-level intervals were misleading precisely
where the retracted claims lived.

## Results

| metric | layer | A₁ mean [95% CI] | A₂ mean [95% CI] | overlap |
|---|---|---|---|---|
| **oracle %** | 0 | 39.7 [30.3, 48.0] | 26.7 [20.7, 32.7] | yes |
| | 1 | 36.6 [29.9, 44.0] | 29.4 [23.0, 35.8] | yes |
| | 2 | 34.7 [26.2, 41.8] | 35.8 [29.0, 42.9] | yes |
| | 3 | 27.5 [19.4, 35.4] | 28.3 [21.1, 35.8] | yes |
| **× random** | 0 | 4.9 [2.9, 7.3] | 7.9 [5.1, 11.3] | yes |
| | 1 | 11.9 [7.9, 16.8] | 11.9 [7.5, 17.6] | yes |
| | 2 | 16.0 [9.6, 24.5] | 19.6 [12.5, 29.8] | yes |
| | 3 | 15.1 [7.9, 25.4] | 13.8 [7.9, 21.9] | yes |
| **top-1% recall** | 0 | 13.5 [6.1, 23.8] | 12.0 [7.7, 17.2] | yes |
| | 1 | 21.1 [14.7, 28.7] | 15.8 [10.7, 21.5] | yes |
| | 2 | 18.4 [11.4, 25.6] | 19.6 [14.2, 26.0] | yes |
| | 3 | 10.8 [6.2, 15.8] | 11.0 [7.0, 15.8] | yes |
| **calib slope** | 0 | 1.28 [1.04, 1.57] | 0.84 [0.32, 1.33] | yes |
| | 1 | 1.11 [0.78, 1.56] | 0.97 [0.61, 1.31] | yes |
| | 2 | 1.03 [0.62, 1.65] | 1.22 [0.73, 1.86] | yes |
| | 3 | 0.80 [0.13, 1.51] | 0.79 [0.29, 1.51] | yes |
| **R²** | 0 | 0.21 [0.09, 0.32] | 0.03 [-0.02, 0.09] | **NO** |
| | 1 | 0.09 [0.03, 0.18] | 0.05 [0.00, 0.09] | yes |
| | 2 | 0.08 [0.01, 0.21] | 0.13 [0.05, 0.24] | yes |
| | 3 | 0.04 [-0.04, 0.10] | 0.04 [-0.05, 0.14] | yes |

**19 of 20 comparisons overlap.** The exception is R² at layer 0 — the
tail-dominated metric already known to be unreliable (0.1% of rows carry 24% of the
variance it is computed from).

## What the intervals establish

**1. The preregistered criterion holds.** A₁ four-layer mean oracle % 34.6, A₂ 30.0,
band 26.1-42.1.

**2. The discovery result is solid, and less precise than reported.**

```
A₁ layer 2   x random   point 13.76x   bootstrap 16.0x  [9.6, 24.5]
```

Report as **~10-25x**. Every layer of both models has a lower bound above 2.9x, so
the direction is not in question; the precision never was there.

**3. Almost nothing about calibration is resolvable.** Only **one** of eight cells
excludes 1.0: A₁ layer 0, 1.28 [1.04, 1.57], a mild over-prediction. Every other
slope interval contains 1.0.

## Two of my claims, now fully retracted

**"Per-layer detail does not replicate."** All per-layer CIs overlap. The 39.7 vs
26.7 gap at layer 0 sits inside sampling variation. The correct statement is
**"not resolvable at this sample size"** — weaker than either "replicates" or "fails
to replicate". I read structure into four noisy points, then read its absence into
the same four.

**"Calibration degrades with depth; layer 3 fails."** A₁ layer 3 is
**0.80 [0.13, 1.51]** — the widest interval in the grid and completely
uninformative. The claim had **no evidential basis at any point**. It was not
falsified by A₂ so much as never supported by A₁.

## Scope

These intervals cover sampling over features, contexts, and acquisition seed
*within* a model. They do **not** support cross-model inference: n = 2 models gives
a clean preregistered replication, not population-level power. Nothing here is a
test of generalisation across models.

Reproduce: `python3 bootstrap_prep.py --model {a,a2} --layer {0..3} --seeds 3`
then `python3 bootstrap_ci.py --reps 5000` · raw: `findings/bootstrap_ci.json`
