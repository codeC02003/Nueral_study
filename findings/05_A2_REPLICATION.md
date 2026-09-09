# A₂ replication — confirmed on the primary criterion, and it falsified my prediction

**A₁** step 83,750 (val 1.0536) · **A₂** step 83,750 (val 1.0228) · seed 1234 the
only difference · identical architecture, corpus, hyperparameters · frozen
evaluation, nothing re-tuned · 40k-step SAE per layer · 3 seeds

Scored against `docs/05_PREREGISTRATION_A2.md`, written before A₂ existed.

## Result

| layer | oracle % A₁ | oracle % A₂ | ×rand A₁ | ×rand A₂ | slope A₁ | slope A₂ |
|---|---|---|---|---|---|---|
| 0 | 39.4% | 26.4% | 5.01x | 8.14x | 1.273 | 0.861 |
| 1 | 35.8% | 28.1% | 11.10x | 9.89x | 1.258 | 1.030 |
| 2 | 31.4% | 34.9% | 13.76x | 17.22x | 1.049 | 1.129 |
| 3 | 29.6% | 28.1% | 16.13x | 14.93x | 0.577 | 0.969 |
| **mean** | **34.1%** | **29.4%** | | | | |

### Primary criterion — PASSED

> Replication if A₂'s four-layer mean oracle % is within 26.1-42.1% and every layer
> exceeds 15%.

A₂ mean **29.4%** (in band). Minimum layer **26.4%** (above floor). **Confirmed.**

The discovery effect is not an artefact of one trained model.

### Secondary prediction — FAILED

> Stated in advance: calibration will again degrade with depth, and layer 3 will
> again fail it. If A₂'s layer 3 is well calibrated, the A₁ result was
> model-specific.

```
A₁ slopes:  1.273  1.258  1.049  0.577
A₂ slopes:  0.861  1.030  1.129  0.969
```

A₂'s layer 3 is at **0.969** — essentially unbiased, and better calibrated than
A₂'s layer 0. **The prediction is falsified.** Layer 3's collapse on A₁ was
model-specific, not a property of depth.

**Consequence:** the planned work on "deep-layer calibration" is cancelled. There is
no depth problem to fix. A₂ is also better calibrated overall (range 0.861-1.129
against A₁'s 0.577-1.273).

## The correction this forces on the earlier writeup

`V3_LAYER_VALIDATION.md` reported that oracle % **falls with depth**:

```
A₁  39.4 -> 35.8 -> 31.4 -> 29.6    monotone decline
A₂  26.4 -> 28.1 -> 34.9 -> 28.1    no trend
```

**It does not replicate.** Layer 0 moved 39.4 -> 26.4; layer 2 moved 31.4 -> 34.9.
With four layers and one model, that monotone pattern was noise read as structure.
The same applies to the `x random` trend, which also reverses direction at layer 0.

**Per-layer detail does not replicate. The aggregate does.** The supportable claim
is narrower: *B captures roughly 26-39% of achievable causal mass at every layer of
both models.* Not "shallower layers are easier."

## What is now established, and what is not

| claim | status |
|---|---|
| B predicts intervention effects, finds high-impact ones faster than random | **fairly strong** — replicated across 2 models, 8 model-layer pairs |
| property of the method, not one layer | **strong** |
| property of the method, not one model | **replicated once** — n=2, no population-level claim |
| calibration degrades with depth | **falsified** — and the A₁ observation it rested on was itself within noise (slope CI [0.56, 1.05] includes 1.0) |
| per-layer oracle % ordering | **not resolvable** at this sample size — all CIs overlap |
| B has discovered A's actual mechanism | **not validated, not claimed** — still needs a known-circuit benchmark |

## Deviations

Two, both logged in `docs/05_PREREGISTRATION_A2.md`, both caught before any metric was
computed: a relative-vs-absolute step-count flag that overshot to 130,000, and a
missing final checkpoint that left `a2_latest.pt` at 80,000. Both runs were
discarded and the root causes fixed. **No result was seen and set aside.**

Reproduce: `./src/run_a2_eval.sh` · raw: `findings/layers_a2/L{0,1,2,3}.json`
