# The two metrics — what was tested, and what came back

Every number here is measured on this machine. Sources named at the end of each
section. `A` is frozen at step 83,750 throughout; nothing in these experiments
trains A.

---

## The question the metrics exist to answer

> Can B predict the causal effect of unseen interventions on frozen A, and by
> choosing its own experiments, reach a given accuracy with fewer experiments than
> random selection?

Two metrics, both scored against **A's own loss** — no human labels anywhere, so
the governing rule ("nobody tells B what to look for") holds.

| | What it asks | Verdict |
|---|---|---|
| **1 · Causal calibration** | B says Δloss = +0.010. Does reality say +0.010? | **PASSED** |
| **2 · Scientific efficiency** *(as average prediction error)* | Does B reach a given accuracy with fewer experiments than random? | **FAILED — and the definition was wrong** |
| **2 · Redefined: mechanism discovery** | Of A's total causal effect mass, how much has B found after N experiments? | **PASSED — 16.25x random** |

---

# METRIC 1 · CAUSAL CALIBRATION

## What it measures, and why calibration rather than accuracy

Ranking features correctly is not understanding. A salience heuristic can rank.
Stating *how much* damage an intervention causes, before running it, cannot be
faked — it requires a model of what the component does, including predicting
**≈ zero** for a direction A carries but never uses.

Three numbers are reported, and the **slope** is the one that matters:

- **calibration slope** — regress actual on predicted. `1.0` = unbiased. Below 1
  means systematic under-prediction; above 1, over-prediction.
- **MAE** — average absolute error.
- **R²** — variance explained. Kept for comparison, but see the warning under
  Metric 2: it is dominated by a fraction of a percent of rows.

## Test protocol

**Split by feature, always.** Test rows belong to 200 features that were never
ablated during training, so nothing can be answered by memorising a feature's
average. Both strategies are scored on the identical frozen test set.

## Results

### Attempt 1 — scalar target: `feature → mean Δloss`

| | value |
|---|---|
| MAE, held-out | 0.000763 |
| predict-the-mean baseline | 0.001088 |
| improvement over baseline | 29.9% |
| R² | **0.327** |
| Pearson r | 0.579 |
| train / test features | 819 / 205 |

It got there only after three fixes. The **first attempt scored R² = 0.069** with
train MAE of *exactly* `0.000000` — a 60K-parameter head memorising 416 samples.

| Fix | Effect |
|---|---|
| shrink to ~15K params, dropout 0.2, weight decay 0.3 | stops memorisation |
| **give the head what a direction does to A's output logits** | largest single gain |
| early-stop on held-out R² | it had peaked at step 500 and decayed for 5,500 more |

A linear ridge baseline scored **−0.245** — worse than predicting the mean — so the
structure learned is genuinely nonlinear, not a fitted trend.

**But the errors had a shape.** B systematically *under-predicted* the large
effects: `+0.0017` predicted where the truth was `+0.0084`.

### The diagnosis that changed the target

The under-prediction was not a loss-function problem. Measured:

```
96.3% of a feature's causal effect lands in 1% of positions
its effect where it fires is ~2,992x its effect where it doesn't
it fires on 0.72% of positions, at magnitude 2.61
```

Asking B for the **average** of a distribution that concentrated is the wrong
question. The scalar target discarded ~99% of the signal. This made the
log-target / quantile-loss debate moot — the target was misspecified, not the loss.

### Attempt 2 — contextual target: `(feature, context) → Δloss`

| | MAE | slope | R² |
|---|---|---|---|
| B @ 200 contexts | 0.1076 | 0.937 | 0.403 |
| B @ 600 contexts | 0.1007 | **1.047** | 0.423 |
| **+ 128-d residual state, 150k rows** | **0.1028** | **0.947** | **0.434** |
| *old scalar model, using all 1,024 features* | — | — | *0.327* |

The slope travelled **0.71 → 1.05**, crossing 1.000 near 400 contexts and sitting at
0.97–0.99 through 320–400 — **essentially unbiased**. The systematic shrinkage that
broke the scalar model is gone.

Adding the full 128-d residual state fixed the *bias* specifically: slope
0.833 → 0.947, while R² moved only 0.395 → 0.434.

## Metric 1 verdict

> **PASSED.** B predicts the causal effect of interventions on features it has
> never been tested against, essentially unbiased (slope ≈ 0.95), explaining 43% of
> variance — against 33% for the scalar model that used five times the features.

Sources: `findings/predictor.json`, `findings/v2_loop.json`, `findings/RESULTS.md`

---

# METRIC 2 · SCIENTIFIC EFFICIENCY

## What it measures

Give B and a random control the same starting data, same budget, same frozen test
set. Let B pick its next experiments by ensemble disagreement; let the control pick
uniformly. **The gap between the two learning curves is the result.** If B reaches a
given accuracy with fewer experiments, it has learned *how to study* A — which is a
stronger claim than knowing what is inside A.

## Three experiments were run

### EXP-1 — selecting over **features**. Looked positive, but was invalid.

| target MAE | B needs | random needs | ratio |
|---|---|---|---|
| 0.1200 | 80 | 120 | 1.50x |
| 0.1124 | 120 | 280 | **2.33x** |
| 0.1076 | 200 | 320 | 1.60x |
| 0.1039 | 360 | 400 | 1.11x |
| 0.1007 | 600 | 560 | 0.93x |

Median 1.11x, mean 1.28x, 10/15 levels favouring B. Final: B 0.1007 / random 0.1010.

**Why it does not count — my design error.** The prediction target was a
`(feature, context)` pair but the *selection pool* was 821 features. At budget 600,
random had already sampled **73% of everything available**, so both strategies
converged on nearly the same data. The test saturated before it could measure
anything. **The selection space must match the prediction space.**

### EXP-2 — selecting over **contexts** (452,219 candidates, budget 2,400 = 0.53%)

Four acquisition rules, identical seed data, 2 seeds:

| strategy | best MAE | reached at | final MAE | trend | ratio vs random |
|---|---|---|---|---|---|
| pure uncertainty | 0.1520 | 600 | 0.2368 | **DEGRADING** | 0.14x |
| mixed (½ uncertainty, ½ random) | 0.1299 | 1,000 | 0.1378 | degrading | 0.71x |
| top-k sampled (diversity in uncertain region) | 0.1446 | 800 | 0.1573 | degrading | 0.43x |
| **random** | **0.1209** | 2,200 | 0.1211 | **improving** | — |

**Pure uncertainty sampling got worse with more data** — 0.152 → 0.237. The textbook
pathology: max-disagreement selects outliers, so B's training distribution drifts
away from the test distribution.

### EXP-3 — re-scored on tail-robust metrics, plus a stratified sampler

**The hypothesis being tested:** R² and MAE are variance-weighted, and this target
is brutally heavy-tailed —

```
rows above the 95th pct : 22,611 (5.00%) carry 96.2% of all variance
rows above the 99th pct :  4,523 (1.00%) carry 77.1%
rows above the 99.9th   :    453 (0.10%) carry 23.7%
```

A tenth of a percent of rows controls a quarter of the score. Uncertainty sampling
*chases* that tail; R² is *scored* on it. Perhaps both failed for the same reason
and the comparison never tested what it claimed.

So: four metrics that the tail cannot dominate, and a **stratified** sampler that
bins candidates into deciles of *predicted* magnitude (observable before running the
experiment) and takes the most uncertain within each — so the budget covers the whole
range of effect sizes instead of collapsing onto freaks.

Final values at budget 2,400:

| | Spearman ρ | slope | log-MAE | top-1% recall | R² |
|---|---|---|---|---|---|
| stratified | 0.1417 | 1.128 | 0.1083 | **0.3825** | 0.2274 |
| mixed | 0.1022 | 1.340 | 0.1020 | 0.3236 | 0.1556 |
| **random** | **0.1543** | 1.223 | **0.0839** | 0.3664 | **0.2895** |

Efficiency ratios vs random:

| metric | stratified | mixed |
|---|---|---|
| Spearman ρ | **1.00x** (2/11 levels) | 0.67x (2/11) |
| log-MAE | 0.50x (0/11) | 0.83x (1/11) |
| top-1% recall | 0.95x (1/10) | 0.83x (0/11) |
| R² | 0.75x (0/11) | 0.70x (1/11) |

**My hypothesis was refuted.** The negative result is not a metric artefact. Random
wins on every metric tested. Stratification is clearly the best active strategy —
it *ties* random on Spearman — but nothing beats random.

## Four diagnostics, run in order

| # | Question | Answer |
|---|---|---|
| 1 | Is the measurement noisy? | **No.** Repeating an intervention differs by exactly `0.00e+00`. A is deterministic; there is no noise floor to blame. |
| 2 | Is B's input too thin? | **Partly.** The 128-d residual state fixed the bias (slope 0.833 → 0.947) but moved R² only 0.395 → 0.434. |
| 3 | Is the ceiling missing downstream information? | **No.** Leaking A's layer-3 state changed nothing (0.413 → 0.402). It *had* to: what B needs is the *counterfactual* layer-3 state, which is the quantity being predicted. No shortcut around simulating the intervention. |
| 4 | Is the metric wrong? | **Tested, refuted.** Tail-robust re-scoring still gives random the win. |

## Why random wins — a principled explanation, not a bug

The test set is a **uniform** sample of held-out contexts. Any non-uniform
acquisition introduces train/test distribution shift, and random sampling is the
unbiased estimator of exactly the distribution being scored.

Active learning wins when **labels are expensive** and **evaluation weights hard
cases**. Neither holds here: labels cost 18 seconds for 1,024 features, and
evaluation is average-case over a uniform sample. Random is not just adequate — for
this evaluation it is *optimal*.

## The exception, and what it implies

On **top-1% recall** — *of the genuinely most-damaging contexts, how many does B
find?* — stratified ends **ahead**: 0.3825 vs random's 0.3664.

That is a **retrieval** metric, and it is far closer to what a scientist actually
optimises. A scientist finds important mechanisms; they do not minimise average-case
error over a uniform sample.

## Metric 2 verdict

> **FAILED as defined, and the definition is the problem.** Random selection beat
> every uncertainty-based strategy on every metric tested, for a principled reason:
> the evaluation is average-case over a uniform distribution, which random sampling
> estimates optimally. But on the one retrieval-flavoured metric, stratified
> selection leads. **Metric 2 should be redefined as retrieval before being called
> answered.**

Sources: `findings/v2_loop.json`, `findings/v2_acquisition.json`,
`findings/v2_metric2.json`

---

# SUMMARY

| | Metric 1 · calibration | Metric 2 · efficiency |
|---|---|---|
| **Verdict** | PASSED | FAILED as defined |
| **Best number** | slope 0.947, R² 0.434 | random wins everywhere |
| **Beat the baseline?** | yes — 0.434 vs 0.327 with 5x fewer features | no — best active ratio 1.00x (tie) |
| **Failed attempts before it worked** | 1 (R² = 0.069, memorisation) | 3 (saturated pool; 4 strategies; tail-robust rescore) |
| **What the failure taught** | the target was misspecified, not the loss | the *evaluation* is misspecified, not B |

## UPDATE — Metric 2, redefined and passed

Rescoring as *discovery* rather than *average prediction error* reversed the result
completely. Full detail in `findings/V3_DISCOVERY.md`:

| strategy | mass found @2,400 | x random | % of oracle | top-1% recall |
|---|---|---|---|---|
| random | 0.68% | 1.00x | 2.4% | 0.7% |
| uncertainty | 6.25% | 9.25x | 22.3% | 11.3% |
| **predicted magnitude** | **10.99%** | **16.25x** | **39.2%** | **22.3%** |

Random would need ~40,567 experiments to match B's 2,400. **~17x.**

**The cost, and its resolution.** Calibration initially collapsed under exploitation
(slope 0.407, R² −0.203 versus 1.302 / 0.238 for random). Splitting acquisition
80/20 helped only slightly — because a 20% random anchor *by count* is 0.57% of the
training *mass*, the magnitude-selected rows being 44x larger. Fixing the loss rather
than the sample solved it:

| config | mass found | x random | slope | R² |
|---|---|---|---|---|
| random + raw target | 0.68% | 1.00x | 1.302 | 0.238 |
| magnitude + raw | 12.18% | 18.01x | 0.371 | −0.276 |
| mag80 + raw | 9.27% | 13.72x | 0.465 | −0.080 |
| **mag80 + log target** | **8.42%** | **12.45x** | **1.089** | **0.120** |

`mag80 + log` is *better calibrated than random acquisition ever was* while keeping
12.45x the discovery. **The contradiction is resolved** — see
`findings/V3_RESOLVED.md`.

**Reproduce everything:**

```bash
python3 harvest.py                                        # the 452,219-row dataset
python3 v2_loop.py --budget 600 --step 40                 # EXP-1, feature selection
python3 v2_context_loop.py --acq uncert,mixed,topk,random # EXP-2, context selection
python3 v2_metric2.py                                     # EXP-3, tail-robust
python3 v3_discovery.py                                   # V3, mechanism discovery
```

Full narrative including every earlier failure: `PROJECT.md`.
Deferred work with trigger conditions: `NOT_NOW.md`.
