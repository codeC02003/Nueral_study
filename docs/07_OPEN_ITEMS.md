# V2 — the plan

> **HISTORICAL DOCUMENT — reflects the project state as of V2, before V3/V4 superseded this plan. Do not use for
> current claims.** Kept as a record of what was believed and planned at the time.
> The authoritative claims are in [`03_CLAIMS.md`](03_CLAIMS.md).


**One question:**

> Can B predict the causal effect of unseen interventions on frozen A, and by
> choosing its own experiments, reach a given accuracy with fewer experiments
> than random selection?

**Two metrics**, both scored against A's own loss — no human labels, so the
governing rule ("nobody tells B what to look for") holds:

1. **Causal calibration** — B says Δloss = +0.010; reality says +0.011. Report the
   calibration slope, mean absolute error, and correlation. The slope matters most:
   B currently ranks features roughly right but *shrinks* large effects, so its
   calibration line is flatter than y = x.
2. **Scientific efficiency** — experiments B needs to reach a given accuracy,
   divided by experiments random selection needs. This is the metric that tests
   whether B has learned *how to study* A.

**A is frozen** at step 83,750. It is a laboratory specimen. No competing GPU load.

---

## The four steps

### 1. Make the measurement trustworthy
- mean-ablation as the primary intervention (zero-ablation kept as a sanity check)
- 10+ probe batches per feature, with standard error and a bootstrap CI
- **store the per-token deltas** instead of averaging them away
- B sees the uncertainty, so it can tell "I predicted badly" from "the experiment is noisy"

### 2. Change what B predicts
From `feature -> Δloss` to **`(feature, context) -> Δloss`**.

Already justified by measurement, not guesswork:

```
96.3% of a feature's causal effect lands in 1% of positions
its effect where it fires is ~2,992x its effect where it doesn't
it fires on 0.72% of positions, at magnitude 2.61
```

Asking B for the average of a distribution that concentrated is asking the wrong
question. This is the diagnosis of the R² = 0.327 failure — **not** a loss-function
problem, so the log-target/quantile debate is moot.

### 3. Give B uncertainty
Five small prediction heads. Disagreement is the uncertainty estimate. No Bayesian
machinery required.

### 4. Let B choose
```python
while budget:
    u = ensemble.disagreement(candidates)
    exp = argmax(u)                    # vs random.choice(candidates)
    actual = intervene_on_A(exp)
    dataset.add(exp, actual)
    ensemble.fit(dataset)
```
Run both selection strategies at equal budgets, evaluate on the same frozen
held-out intervention set, and plot the two learning curves.

---

## Already settled — do not redo

| Question | Answer |
|---|---|
| Are B's features causally used by A? | **Yes.** 3.50x more damage per unit removed than magnitude-matched random directions; Cohen's d = 1.19 |
| Is the damage specialised? | **Yes.** 67% of a feature's damage lands on one character, vs 8% for random directions |
| Does the ranking survive a different intervention? | **Yes**, ρ = 1.000 zero vs mean — but only *because* features are sparse (they fire on 0.72% of positions, so the mean they clamp to is 0.016). Weaker evidence than it looks; not proof it survives resampling. |
| Is importance a scalar? | **No.** See step 2. |
| Is the prediction failure a loss-function problem? | **No.** The target was misspecified. |

## V2 status

| Step | State |
|---|---|
| 1. Trustworthy measurement | **done** — 452,219 rows, CIs, 77% of features significant |
| 2. `(feature, context)` target | **done** — slope 1.047, R² 0.423 (old: 0.327) |
| 3. Uncertainty via ensemble | **done** — 5 bootstrap heads |
| 4. B chooses experiments | **partial** — 2.33x efficient early, decays to 1.0x by 400 |

**Selection pool: fixed, and the result was negative.** Selecting over 452,219
contexts instead of 821 features, *every* uncertainty strategy lost to random
(0.14x, 0.71x, 0.43x). Pure uncertainty got worse with more data.

**Diagnosed to a shared cause:** the target spans 0.007 to 13.5, and 0.1% of rows
carry 24% of the variance R² is scored on. Uncertainty sampling *chases* that tail;
R² is *scored* on that tail. Both fail for the same reason. Meanwhile the
measurement is exactly deterministic, and leaking A's layer-3 state changes nothing
- so it is not noise and not missing downstream information.

**The two open items now:**
1. Re-score on Spearman / calibration slope / log-magnitude instead of raw R².
2. Stratify acquisition - uncertainty *within* magnitude strata, not global argmax.

Neither needs new infrastructure. See `findings/01_RESULTS.md` for the full diagnosis.

## What is deliberately not in this plan

See `08_NOT_NOW.md` — ten deferred items, each with a trigger condition.
