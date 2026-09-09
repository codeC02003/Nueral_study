# What is claimed, and what each claim rests on

Written to keep separate claims separately graded. The failure mode this guards
against is letting evidence for a narrow claim quietly support a broad one.

---

## CLAIM 1 — B predicts intervention effects, and finds high-impact interventions faster than random

**Status: fairly strong.**

**This claim has objective ground truth.** We perform the intervention on frozen A
and measure A's actual loss. The target is directly observable, not inferred, so
held-out interventions *are* the ground truth for this claim. It does not depend on
any known-circuit benchmark.

Evidence:
- calibration slope ≈ 1.0 on held-out contexts of features never acquired
- 12.45x random on captured causal mass; 39.2% of the oracle bound at 0.65% budget
- controls: magnitude-matched random directions, 3.50x, Cohen d = 1.19

Limits: `mag80 + log` was selected post-hoc on A₁ (see Claim 3). No bootstrap CIs yet.

---

## CLAIM 2 — this is a property of the method, not of one layer

**Status: strong.**

The frozen configuration was applied unchanged to three layers it had never seen.

```
oracle %:  L0 39.4   L1 35.8   L2 31.4   L3 29.6      mean 34.1, CV 11.2%
```

Layer 2 - where the config was developed - ranked third of four. Also robust to a
9x change in SAE training budget, though that was tested on layer 2 only.

Caveat carried: n = 4 layers. "Appears at every layer tested", not "layer-invariant".

---

## CLAIM 3 — this generalises across independently trained models

**Status: pending A₂.**

`PREREGISTRATION_A2.md` is a **prospective replication preregistration** - it fixes
the evaluation of A₂ in advance. It is *not* a preregistration of the project:
`mag80 + log` was discovered post-hoc on A₁, and that ordering is on the record.

**With two models, no conventional significance test about cross-model
generalisation is appropriate.** n = 2 cannot support a population-level claim. What
A₂ can deliver is a clean, preregistered replication plus uncertainty intervals
*within* each model. Population-level generality needs many more independently
trained As, and is **not claimed**.

---

## CLAIM 4 — B has discovered A's actual mechanism / understands what A computes

**Status: NOT VALIDATED. This claim is not made.**

This is where a known-circuit benchmark becomes essential, and it is the reason the
induction task was in the original plan.

What we can support: B found directions that are **causally load-bearing** - removing
them damages A far more than removing comparable random directions, and the damage
is concentrated (67% on a single character vs 8% for random).

What we cannot support: that B's decomposition **corresponds to the true algorithm
inside A**. Matched controls establish that these directions matter causally. They
do **not** validate the semantic reading. `f515 -> the r in return` is a plausible
interpretation of where the damage lands; it is not a verified account of A's
mechanism.

To close it: train a small A on a task whose circuit is already established
(induction: previous-token head in layer 0 feeding an induction head in layer 1),
run the identical pipeline, and count what fraction of the known circuit B recovers.

---

## On uncertainty quantification

The interventions are **deterministic** - repeating one gives a difference of
exactly `0.00e+00`, so there is no repeated-measurement noise.

That does **not** remove sampling uncertainty. Which features and contexts enter the
pool, the train/test split, and the acquisition trajectory are all random draws.
**Bootstrap confidence intervals over features/contexts remain necessary** and are
not yet computed for the headline numbers.

## On oracle %

A metric constructed for this project. That is acceptable because its definition is
fixed and stated (captured causal mass / the maximum any selector could capture at
the same budget), and because raw mass, x random and top-1% recall are reported
alongside it and agree in direction. What would be illegitimate is changing the
metric repeatedly until a favourable result appears - the record shows the opposite:
Metric 2's first definition produced a negative result that was reported as such.

---

## Grading

| | status |
|---|---|
| causal discovery (Claim 1) | fairly strong |
| cross-layer robustness (Claim 2) | strong |
| cross-model generalisation (Claim 3) | pending A₂ |
| mechanistic correctness (Claim 4) | **unvalidated** |
| population-level generality | weak - too few independently trained As |
