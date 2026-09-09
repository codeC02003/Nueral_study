# V3 validated across all four layers of A

> **SUPERSEDED IN PART — see `05_A2_REPLICATION.md`.** The aggregate result
> here replicated on a second model. The **depth trends did not**: A₂ shows no
> monotone decline in oracle % and no calibration degradation with depth. Sections
> 2 and 4 below over-read four points from one model. Read them with that caveat.

**A frozen** @ 83,750 · every layer given an **identical 40,000-step SAE budget**
· frozen V3 config (`mag80` acquisition + log-magnitude target) · budget 2,400
· 3 seeds · hold-out split by feature

| layer | SAE recon | L0 | live feats | rows | mass ×rand | top-1% ×rand | **oracle %** | slope | R² |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.8% | 7.2 | 489 | 220,427 | 5.01x | 12.5x | **39.4%** | 1.273 | 0.199 |
| 1 | 8.0% | 11.3 | 1024 | 353,561 | 11.10x | 25.2x | **35.8%** | 1.258 | 0.095 |
| 2 | 13.5% | 15.3 | 1024 | 482,346 | 13.76x | 29.9x | **31.4%** | 1.049 | 0.073 |
| 3 | 12.7% | 25.3 | 1024 | 789,421 | 16.13x | 31.6x | **29.6%** | 0.577 | 0.015 |

```
oracle %      mean 34.1%   range 29.6-39.4%  (9.8 percentage points)   CV 11.2% (population sd 3.82; sample sd 4.42 gives 13.0%)
x random      mean 11.50x  range 5.01-16.13x
calib slope   mean 1.039   range 0.577-1.273
```

## 1. The discovery effect is not peculiar to layer 2

Oracle % — the fraction of the *achievable* causal mass B captured at the same
budget — has **mean 34.1%, range 29.6-39.4% (a spread of 9.8 percentage points),
coefficient of variation 11.2% (population sd 3.82; the sample-sd convention gives 13.0%)** across the four layers. Layer 2, where V3 was
measured, sits third of four: the median, not a lucky pick.

Four points is a small sample. The claim supported is "the effect appears at every
layer tested", not "the effect is layer-invariant".

## 2. `x random` is the wrong metric across layers, and it would have misled us

```
depth ->        L0      L1      L2      L3
x random      5.01x  11.10x  13.76x  16.13x     rises with depth
oracle %      39.4%   35.8%   31.4%   29.6%     falls with depth
```

They move in **opposite directions**. `x random` depends on how the mass happens to
be spread in that layer's pool — deeper layers have more rows and denser effects, so
random does worse and the ratio inflates. Reporting only `x random` would have
supported "deeper is better", which the normalised measure contradicts.

## 3. On layer 2, the result does not depend on the SAE training budget

**Scope: this was tested on layer 2 only.** It has not been shown for the other
three layers, each of which was run at 40k steps and nothing else.

| layer-2 SAE | steps | mass ×rand | oracle % | slope |
|---|---|---|---|---|
| frozen V3 | 362,000 | 12.45x | 30.0% | 1.089 |
| this run | 40,000 | 13.76x | 31.4% | 1.049 |

Essentially identical from **9x less** SAE training. The V3 headline was not an
artefact of over-training B.

## 4. ~~New finding — calibration degrades with depth~~ FALSIFIED BY A₂

**A₂ layer 3 slope is 0.969 — well calibrated.** This trend was specific to A₁.
The text below is kept as the record of what was observed, not as a finding.

```
slope   1.273  1.258  1.049  0.577
R²      0.199  0.095  0.073  0.015
```

Layer 3 has the **best** raw discovery (16.13x) and the **worst** calibration —
slope 0.577 means it over-predicts by ~1.7x, and R² 0.015 is barely above the mean.
It is also the densest layer (L0 = 25.3, 789k rows, deepest position in the
residual stream).

**B's discovery ability generalises across depth; B's predictive calibration does
not.** This was invisible from layer 2 alone.

Feature density (L0) and row count both rise with depth alongside the calibration
loss, so density **correlates** with the degradation. With four layers and no
intervention on density, **the cause is not established** - depth, density, row
count and position in the residual stream are all confounded here. Fixing it would
require varying density at fixed depth, which has not been done.

## What this validates, and what it still does not

**Validated:** the effect is a property of the *method*, not of layer 2, and not of
how long B's SAE was trained. Four layers, four independent SAEs, one frozen
configuration, consistent result.

**Not validated:** independence from *this particular A*. Every number here comes
from one model trained with one seed on one corpus. The remaining test is a second,
independently trained A — that is what would move the claim from "not peculiar to a
layer" to "not peculiar to a model".

Reproduce: `python3 src/validate_layer.py --layer {0,1,2,3} --sae-steps 40000 --seeds 3`
Raw: `findings/layers/L{0,1,2,3}.json`
