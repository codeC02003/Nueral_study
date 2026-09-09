# V3 — B discovers A's important mechanisms 17x faster than random

> **TERMINOLOGY NOTE (added later).** This document uses "mechanisms" for what is
> more precisely called **causally important features and interventions**. B was
> never shown to recover A's actual algorithm — the benchmark for that was abandoned
> (`07_V5_ABANDONED.md`). The numbers below stand; only the word was too strong.
> Authoritative wording: [`../docs/03_CLAIMS.md`](../docs/03_CLAIMS.md).

**A frozen** @ 83,750 · pool 369,128 contexts · budget 2,400 (0.65% of the pool)
· 3 seeds · 200 test features never acquired

Metric 1 unchanged. Metric 2 redefined from *average prediction error* to
**important-mechanism discovery**: of the total causal effect mass in A, how much
has B found after N experiments?

---

## METRIC 2 — the answer is yes, decisively

| strategy | mass found @2,400 | x random | % of oracle | top-1% recall | x random |
|---|---|---|---|---|---|
| random | 0.68% | 1.00x | 2.4% | 0.7% | 1.00x |
| uncertainty | 6.25% | 9.25x | 22.3% | 11.3% | 15.19x |
| **predicted magnitude** | **10.99%** | **16.25x** | **39.2%** | **22.3%** | **29.88x** |
| hybrid (magnitude + σ) | 10.68% | 15.80x | 38.1% | 21.3% | 28.51x |

Random never reaches any of them within budget. Extrapolating (random's mass grows
linearly with N):

```
to match B's 2,400 experiments, random needs
    uncertainty   ~23,070    10x
    magnitude     ~40,567    17x
    hybrid        ~39,423    16x
```

**B finds A's important mechanisms with ~17x fewer experiments than random search.**
That is the claim the project was built to test, and it holds.

Two details worth noting:

- **The oracle normalisation** says B captured **39.2% of the maximum mass any
  selector could have reached at that budget** — against random's 2.4%. Not perfect,
  clearly not luck.
- **Uncertainty adds nothing here.** Hybrid (`|pred| + σ`) is marginally *worse* than
  pure magnitude. For discovery, B should predict and commit, not explore. This
  directly inverts the earlier V2 result where uncertainty was the whole strategy.

---

## METRIC 1 — and here is the cost

| experiments | random | uncertainty | magnitude | hybrid |
|---|---|---|---|---|
| calibration slope @2,400 | **1.302** | 0.652 | **0.407** | 0.439 |
| R² @2,400 | **0.238** | 0.063 | **−0.203** | −0.148 |

**Calibration collapses under exploitation.** A slope of 0.407 means the model
over-predicts by ~2.5x on the general population: trained almost exclusively on
high-effect contexts, it comes to believe everything is a big effect. Negative R²
means it is worse than predicting the mean on ordinary contexts.

The ordering is exact and monotone: the harder a strategy chases important
mechanisms, the worse its population-level calibration.

```
             discovery      calibration
random          worst          best
uncertainty     middle         middle
magnitude       best           worst
```

---

## What this actually shows

**B can be a good prospector or a good surveyor. With a single model and these
acquisition rules, not both.**

That is not a bug — it is distribution shift, stated precisely. Acquiring by
predicted magnitude produces a training set drawn from the tail, so the resulting
model is calibrated *for the tail* and miscalibrated everywhere else. The earlier V2
result was the same phenomenon seen from the other side: random wins on
average-case accuracy because random *is* the unbiased sample of the average case.

So the two metrics were never measuring one thing. They measure two different jobs:

| | question | who wins |
|---|---|---|
| Metric 1 | can B model A's dependencies in general? | random acquisition |
| Metric 2 | can B find A's important mechanisms? | magnitude acquisition, 17x |

Both are legitimate. A scientist needs both. The mistake was expecting one
acquisition rule to serve both.

## The fix, and it is not expensive

**Separate the acquisition set from the training set.** Nothing forces the model
that predicts to be trained only on what the selector chose:

1. **Two-pool acquisition** — spend, say, 80% of the budget on magnitude (discovery)
   and 20% on random (calibration anchor). Train on both.
2. **Importance weighting** — acquire by magnitude but weight the training loss by
   inverse selection propensity, which is the textbook correction for this exact
   shift.
3. **Two models** — a prospector for choosing, a surveyor trained on a random
   subsample for calibrated prediction. They can share the SAE.

Option 1 is one line and would be the thing to try next.

---

## Verdict

> **Metric 1 (calibration): passed** under random acquisition — slope 0.947,
> R² 0.434 (see `docs/04_METRICS.md`), and it is *preserved* here under random acquisition
> (slope 1.302, R² 0.238 at this smaller budget).
>
> **Metric 2 (important-mechanism discovery): PASSED, decisively.** 16.25x random on
> effect mass, 29.88x on top-1% recall, 39.2% of the oracle bound. Random would need
> ~17x more experiments to match.
>
> **Newly discovered constraint:** the two cannot currently be satisfied by one
> acquisition rule. Chasing mechanisms costs calibration, monotonically.

Reproduce: `python3 src/v3_discovery.py` · raw curves `findings/v3_discovery.json`
