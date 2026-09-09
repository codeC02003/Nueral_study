# Results

Verified findings, newest first. Reproduce with the commands at the end of each section.

## B predicts A's dependencies on features it has never ablated

**A** frozen @ 83,750 · **B** frozen @ 362,500 · layer 2 · 1,024 experiments
Split **by feature**: 819 train, 205 held out and never ablated during training.

## Result

| | value |
|---|---|
| B's prediction MAE (held-out) | **0.000763** |
| predict-the-mean MAE | 0.001088 |
| improvement over the trivial baseline | **29.9%** |
| R² | **0.327** |
| Pearson r | **0.579** |
| ridge (linear) baseline R² | −0.245 |

The head is never given a feature's index - only its properties (its decoder and
encoder directions, its firing statistics, and what it pushes A's output logits
toward). An index would let it memorise a lookup table. Properties force it to
learn what *kind* of direction A depends on, which is why it transfers to features
it has never seen ablated.

The ridge baseline scoring worse than the mean matters: the relationship between a
feature's properties and A's dependence on it is genuinely nonlinear, so the model
is learning structure rather than fitting a trend.

## Getting there took three fixes, all mine

The first attempt reached R² = 0.069 - barely above chance:

1. **The head was memorising.** Train MAE was *exactly* 0.000000 on 416 samples with
   a 60K-parameter network. Fix: shrink to ~15K params, dropout 0.2, weight decay 0.3.
2. **It was never told what a direction does to A's OUTPUT.** It got raw 128-d
   vectors and had to infer everything. Whether A depends on a direction is largely
   a question of what that direction pushes the logits toward, and the residual
   stream reaches the output directly. Adding logit reach was the single largest gain.
3. **No early stopping.** Test R² peaked at step 500 and decayed for 5,500 more
   steps. Fix: keep the best model by held-out R², not the last.

## What this establishes

B can now state, in advance, how much A depends on a direction it has never tested.
That is not a salience heuristic - a salience heuristic cannot predict *magnitude*,
and cannot correctly predict a small number for a direction A carries but ignores.

## What it does not establish

**R² = 0.327 is modest.** B gets the ordering roughly right (r = 0.58) but
systematically **underestimates the large effects** - it predicted +0.0017 for
feature 336 where the true cost was +0.0084. It knows which directions matter more
than others; it does not yet know how much.

Likely causes, untested: the target is heavily skewed (most features barely matter,
a few matter a lot) and a regularised regressor pulls toward the mean. A log or rank
target would probably help. Some overfitting remains (train 0.0001 vs test 0.0008).

**B still does not choose its own experiments.** It predicts outcomes for features
handed to it. Choosing what to test, and updating on surprise, is Tier 5.

## Position on the ladder

```
1. STRUCTURE   B finds candidate parts                    done
2. CAUSATION   are those parts used by A?                 done  (3.5x vs random)
3. MECHANISM   B predicts what removing one does          DONE  (R2 = 0.33, partial)
4. SCIENCE     B forms its own hypotheses and tests them  next
```

Reproduce: `python3 train_b_predict.py` · raw numbers in `findings/predictor.json`

---

## Verified: B's features are causally used by A

**A** frozen at step 83,750 · **B** at step 362,500 · tap = layer 2 residual stream
Baseline loss 0.9202 · 40 features vs 40 magnitude-matched random directions

## 1. The features are load-bearing

| | mean Δloss | damage per unit removed |
|---|---|---|
| B's features (n=40) | 0.00403 | **0.001020** |
| random controls (n=40) | 0.00119 | 0.000292 |

**3.50× more damage per unit removed. Cohen's d = 1.19** (a large effect).

The control is what makes this a claim. Deleting *any* direction hurts A, so
"ablating feature 507 raised the loss" proves nothing by itself. Random directions
of matched magnitude were removed from the same layer on the same batch, and B's
features did three and a half times more damage.

## 2. The damage is specialised, not diffuse

| | share of damage on its single most-affected character |
|---|---|
| B's features | **67.1%** |
| random controls | 8.0% |

**8.4× more concentrated.** A random direction degrades everything a little. One of
B's features degrades essentially one thing. That is what a functional part looks
like, as opposed to a piece of shared substrate.

## 3. What each feature is FOR

Ablating a feature and reading the contexts where A's prediction collapses:

| feature | context it protects | what it does |
|---|---|---|
| f336 | `self[.]` | predicts the dot after `self` |
| f873 | `sel[f]` | completes the word `self` |
| f515 | `retu[r]` | predicts the `r` in `return` |
| f384 | `Non[e]` | completes `None` |
| f691 | `'\n[']` | predicts a quote at a docstring boundary |
| f965 | `pro[g]` | continues the `prog...` stem |

These are **token-completion circuits**. Each is a specific "finish this word"
mechanism, and removing it breaks that word and almost nothing else.

## Why this matters

It closes the gap flagged as this project's main hazard. B's earlier findings were
*correlational* - a feature fires near `None`, which does not show A uses it. A
probe can decode information at perfect accuracy that the model ignores entirely.

Now it is causal: remove f384 and A can no longer finish `None`. The information
is not merely present in layer 2, it is **load-bearing**.

## What this does NOT establish

**B did not discover this. B found the candidates; the ablation was run by hand.**
B has no prediction head, so it cannot yet state what removing a feature will do,
and it was not scored on anything. This is evidence about A's anatomy, not evidence
that B understands A.

The full circuit is also unresolved. We know f515 is required to predict `r` in
`return` - not *how*: which attention heads feed it, which MLP neurons compute it,
how it reaches the output. That is a circuit trace, and it is not built.

## Position on the ladder

```
1. STRUCTURE   B finds candidate parts                    done
2. CAUSATION   are those parts actually used by A?        DONE - this document
3. MECHANISM   B predicts what removing one does          next
4. SCIENCE     B forms its own hypotheses and tests them  the goal
```

Reproduce: `python3 ablate.py` · raw numbers in `findings/ablation.json`

---

## V2 — can B predict causal effects and choose its own experiments?

**A** frozen @ 83,750 · 452,219 (feature, context) rows · 821 candidate features
· 200 test features never queried · 2 seeds · budget 600

### Metric 1 — causal calibration: PASSED

| | MAE | calibration slope | R² |
|---|---|---|---|
| B @ 600 features | 0.1007 | **1.047** | **0.423** |
| B @ 200 features | 0.1076 | 0.937 | 0.403 |
| old scalar model | — | — | 0.327 (using **all** 1,024 features) |

Slope went 0.71 → 1.05, crossing 1.000 near 400 features; at 320–400 it sits at
0.97–0.99, i.e. **essentially calibrated**. The old model's systematic
under-prediction of large effects is gone, and B beats its R² with a fifth of the
features. Changing the target from `feature -> Δloss` to `(feature, context) -> Δloss`
is what did it — as the 96.3%-in-1%-of-positions measurement predicted it would.

### Metric 2 — scientific efficiency: PARTIAL

| target MAE | B needs | random needs | ratio |
|---|---|---|---|
| 0.1200 | 80 | 120 | 1.50x |
| 0.1124 | 120 | 280 | **2.33x** |
| 0.1076 | 200 | 320 | 1.60x |
| 0.1039 | 360 | 400 | 1.11x |
| 0.1007 | 600 | 560 | 0.93x |

Median 1.11x, mean 1.28x, 10 of 15 accuracy levels favour B. **B is up to 2.33x
more efficient at low budgets, and the advantage decays to nothing by ~400.**

### Why the advantage decays — a flaw in the experiment, not in B

**The test is budget-saturated.** The pool holds 821 candidate features; a budget of
600 means random has already sampled 73% of everything available. Past that point
both strategies have seen nearly the same data, so no selection rule can help.

**The fix follows from the dataset already built:** B should select over
**(feature, context) pairs — 452,219 of them — not over 821 features.** A budget of
600 is then 0.13% of the pool rather than 73%, which is the regime where active
learning can actually demonstrate anything. The selection space should have matched
the prediction space; it did not.

### Verdict

> B predicts the causal effect of unseen interventions on frozen A, calibrated
> (slope ~1.0, R² 0.42), and is up to 2.33x more experiment-efficient than random
> in the unsaturated regime. Whether that efficiency holds at scale is untested,
> because the candidate pool was too small to test it.

Reproduce: `python3 harvest.py && python3 v2_loop.py --budget 600 --step 40`
Raw curves: `findings/v2_loop.json`

---

## V2 corrected — selecting over contexts. A negative result, and why.

Fixing the saturated pool (821 features -> 452,219 contexts; budget 2,400 = 0.53%
of the pool) produced a clean **negative** result on Metric 2, and diagnosing it
turned out to be the more valuable outcome.

### Every active-learning strategy lost to random

| strategy | best MAE | at | trend |
|---|---|---|---|
| pure uncertainty | 0.1520 | 600 | **DEGRADING** |
| mixed (half uncertainty, half random) | 0.1299 | 1000 | degrading |
| top-k sampled (diversity within uncertain) | 0.1446 | 800 | degrading |
| **random** | **0.1209** | 2200 | **improving** |

Efficiency vs random: uncertainty 0.14x, mixed 0.71x, top-k 0.43x. **All below 1.0.**
Pure uncertainty sampling actively got *worse* with more data - MAE 0.152 -> 0.237 -
the classic pathology: max-disagreement selects outliers, so the training
distribution drifts away from the test distribution.

### Four diagnostics, run in order

1. **Is the measurement noisy?** No. Repeating an intervention gives a difference of
   exactly `0.00e+00`. A is deterministic; there is no noise floor to blame.
2. **Is B's input too thin?** Partly. Adding the full 128-d layer-2 residual state
   moved R² 0.395 -> 0.434 and, more importantly, calibration slope 0.833 -> 0.947.
   Richer context fixes the *bias* but not the ceiling.
3. **Is the ceiling missing downstream information?** No. Leaking A's layer-3 state
   to B changed nothing (R² 0.413 -> 0.402). On reflection this had to be so: what
   B would need is the *counterfactual* layer-3 state, which is the quantity being
   predicted. There is no shortcut around simulating the intervention.
4. **Is R² measuring the right thing?** **No, and this is the finding.**

```
rows above the 95th pct : 22,611 (5.00%) carry 96.2% of all variance
rows above the 99th pct :  4,523 (1.00%) carry 77.1%
rows above the 99.9th   :    453 (0.10%) carry 23.7%
```

R² is variance-weighted, so **a tenth of a percent of rows control a quarter of the
score.** B can predict 99% of contexts well and still be pinned near R² = 0.4. The
metric is dominated by rare extremes.

### What this means

The negative result on Metric 2 is **partly an artefact of the metric**. Uncertainty
sampling and a variance-weighted score fail together for the same reason: both are
dominated by the extreme tail. Uncertainty sampling *chases* the tail; R² is
*scored* on the tail; and a model that fits the tail poorly looks bad on both while
being fine on the other 99%.

Two corrections follow, and they are not cosmetic:

- **Score on a rank or quantile metric**, or on log-magnitude, not raw R². Spearman
  correlation and calibration slope are the honest summaries for a target spanning
  0.007 to 13.5.
- **Stratify the acquisition.** Uncertainty *within* strata of effect magnitude,
  rather than global argmax, so B cannot spend its whole budget on freak rows.

### Verdict, stated plainly

> **Metric 1 (calibration): passed.** Slope 0.947 with the residual state included;
> R² 0.434, against 0.327 for the old scalar model.
>
> **Metric 2 (scientific efficiency): failed as measured.** Random selection beat
> every uncertainty-based strategy. But the measurement is confounded by a
> heavy-tailed target that both the acquisition rule and the score over-weight, so
> the question is not yet properly answered.

Reproduce: `python3 v2_context_loop.py --acq uncert,mixed,topk,random`
Raw curves: `findings/v2_acquisition.json`

---

## Metric 2, re-scored on tail-robust metrics — the negative result holds

My hypothesis was that R² is variance-weighted (0.1% of rows carry 23.7% of the
variance), so the negative result might be a metric artefact. **Tested. Refuted.**

Re-scored on Spearman rho, calibration slope, log-MAE and top-1% recall, and added a
**stratified** sampler (deciles of predicted magnitude, most-uncertain within each,
so the budget cannot collapse onto the freak tail).

| metric | strategy | median ratio vs random | levels favouring it |
|---|---|---|---|
| Spearman rho | stratified | 1.00x | 2/11 |
| Spearman rho | mixed | 0.67x | 2/11 |
| log-MAE | stratified | 0.50x | 0/11 |
| top-1% recall | stratified | 0.95x | 1/10 |
| R² | stratified | 0.75x | 0/11 |

Stratification is clearly the best active strategy — it *ties* random on Spearman —
but nothing beats random on any metric.

**Why, and it is principled rather than a bug:** the test set is a uniform sample of
held-out contexts. Any non-uniform acquisition introduces train/test distribution
shift, and random sampling is the unbiased estimator of exactly the distribution
being scored. Active learning wins when labels are expensive and evaluation weights
hard cases; here labels cost 18 seconds and evaluation is average-case.

**The exception that points at V3:** on top-1% recall — *does B find the genuinely
most-damaging contexts?* — stratified ends ahead, 0.3825 vs 0.3664. That is a
retrieval metric, and much closer to what a scientist actually optimises. The
conclusion is that **Metric 2 was defined wrong**: average-case predictive accuracy
over a uniform sample is not what "learning how to study A" should mean.

Reproduce: `python3 v2_metric2.py` · raw curves `findings/v2_metric2.json`
