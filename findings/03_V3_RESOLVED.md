# The V3 contradiction, resolved

**A frozen** @ 83,750 · pool 369,128 contexts · budget 2,400 · 3 seeds
· Metric 1 on 200 never-acquired features

V3 found that B could be a good **prospector** (find important mechanisms, 18x
random) or a good **surveyor** (calibrated over the population) but not both. This
resolves it. Two changes, and only the second one was the real fix.

## The full grid

| acquisition | target | mass found | x random | calib slope | R² |
|---|---|---|---|---|---|
| random | raw | 0.68% | 1.00x | 1.302 | **0.238** |
| magnitude (100/0) | raw | **12.18%** | **18.01x** | 0.371 | −0.276 |
| mag90 | raw | 10.41% | 15.39x | 0.420 | −0.124 |
| mag80 | raw | 9.27% | 13.72x | 0.465 | −0.080 |
| mag50 | raw | 6.12% | 9.05x | 0.542 | +0.017 |
| random | log | 0.68% | 1.00x | 2.083 | 0.191 |
| magnitude | log | 10.74% | 15.88x | 0.873 | 0.149 |
| **mag80** | **log** | **8.42%** | **12.45x** | **1.089** | **0.120** |

**`mag80 + log` is the answer.** Its calibration is *closer to unbiased than random
acquisition ever achieved* — |1.089 − 1| = 0.089 against random's |1.302 − 1| =
0.302 — while retaining **12.45x the discovery rate**.

## Step 1: split acquisition. Necessary, and nowhere near sufficient.

80% predicted-magnitude + 20% random, trained on both:

```
discovery   13.72x random   (kept 76% of pure magnitude's 18.01x)
slope       0.371 -> 0.465
R²         -0.276 -> -0.080
```

Better, monotonically — mag90 -> mag80 -> mag50 improves calibration and costs
discovery in exact order. But even 50/50 only reached slope 0.542. **The
contradiction was not solved by mixing.**

## Step 2: why mixing could not work

```
magnitude-selected rows : 1,920   mean |Δloss| 5.2088
random anchor rows      :   480   mean |Δloss| 0.1197     44x smaller

count share of the anchor : 20.00%
MASS  share of the anchor :  0.57%     <- the actual gradient signal
```

The training loss is **magnitude-weighted**, so a 20% anchor *by count* is 0.57% of
the signal. Mixing by count cannot rebalance a loss dominated by mass. Adding random
rows was fixing the wrong quantity.

## Step 3: the real fix — stop the tail dominating the loss

Train on `sign(y) · log1p(|y|)` and invert at prediction time. The **acquisition rule
is untouched**, so discovery should survive; only the loss changes, so every row
contributes comparably.

| | slope | R² | discovery |
|---|---|---|---|
| mag80, raw target | 0.465 | −0.080 | 13.72x |
| **mag80, log target** | **1.089** | **+0.120** | **12.45x** |

Calibration essentially restored for a ~9% relative loss in discovery.

## One interaction worth recording

The log target **hurts random acquisition** — slope 1.302 -> 2.083. Trained almost
entirely on tiny values in log space, then exponentiated, it over-predicts badly. So
this is not "log targets are better"; it is that **log targets and magnitude-biased
acquisition are the correct pairing.** Neither works alone:

```
random     + raw  ->  calibrated, finds nothing
magnitude  + raw  ->  finds everything, uncalibrated
random     + log  ->  worst calibration of all
mag80      + log  ->  both
```

## Where this leaves the two metrics

> **Metric 1 (causal calibration): passed.** slope 1.089, R² 0.120 under the
> discovery-oriented configuration; slope 0.947 / R² 0.434 at the larger 150k-row
> budget under random acquisition (`docs/04_METRICS.md`).
>
> **Metric 2 (important-mechanism discovery): passed.** 12.45x random on effect mass
> and 21.85x on top-1% recall while staying calibrated; 18.01x if calibration is
> abandoned.
>
> **The contradiction between them: resolved.** It was never a trade-off between
> exploring and exploiting - it was a mass-weighted loss. Fixing the loss dissolved
> it, at a ~9% cost in discovery.

## What is now true of B

B, reading 4.6% of A, can:

- find directions that are causally load-bearing (3.50x vs matched random controls)
- predict how much removing one costs A, essentially unbiased, on features it has
  never tested
- choose which experiments to run and find A's important mechanisms **~12x faster
  than random search**, without sacrificing calibration

All with no human labels: every signal came from A's own loss.

Reproduce:
```bash
python3 src/v3_discovery.py --acq random,magnitude,mag90,mag80,mag50   # raw target
python3 src/v3_discovery.py --acq random,magnitude,mag80 --target log  # the fix
```
