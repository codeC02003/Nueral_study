# Prospective replication preregistration — a second independently trained A

**Scope.** This preregisters the *evaluation of A₂*, not the project. `mag80 + log`
was discovered **post-hoc on A₁**; that ordering is on the record and this document
does not launder it. What is fixed in advance here is only how A₂ gets measured.

Written **before** A₂ exists. Everything below is fixed in advance so the result
cannot be shaped after the fact.

## The claim being tested

> B, given no human labels, finds A's causally important mechanisms far faster than
> random search, and this is a property of the method rather than of one trained
> model.

Every V3 number to date comes from **one A**: one seed, one corpus, one run.

## What changes, and what does not

| | A₁ (frozen) | A₂ |
|---|---|---|
| architecture | 4L × 4H × 128d, 819,968 params | **identical** |
| corpus | 9.2M chars, Python stdlib | **identical** |
| tokenizer / vocab | char-level, 96 symbols | **identical** |
| optimiser, LR, schedule, batch | AdamW 3e-4, warmup 200, batch 32 | **identical** |
| training steps | 83,750 | **83,750** |
| **init / shuffle seed** | unseeded (default) | **seed 1234** |

Exactly one thing differs: the random seed.

## The evaluation, fixed in advance

Frozen from V3. No component may be re-tuned for A₂.

```
SAE                40,000 steps · expansion 8 · l1 = 1e-2 · same calibration
acquisition        mag80  (80% predicted-magnitude, 20% random)
target             log-magnitude,  sign(y)·log1p(|y|)
experiment budget  2,400 contexts · step 200 · 5 heads · 300 epochs · 3 seeds
hold-out           split by feature, 20% never acquired
layers             0, 1, 2, 3 — all four
```

## Metrics, and what counts as replication

**Primary — oracle %** (budget-normalised captured causal mass). On A₁ the four
layers gave 39.4 / 35.8 / 31.4 / 29.6, mean **34.1%**.

> **Replication if A₂'s four-layer mean oracle % lands within 34.1 ± 8 pp
> (26.1–42.1%) and every individual layer exceeds 15%.**

**Secondary — calibration slope.** A₁ gave 1.273 / 1.258 / 1.049 / 0.577.

> Prediction, stated in advance: **calibration will again degrade with depth, and
> layer 3 will again fail it.** If A₂'s layer 3 is well calibrated, the A₁ result
> was model-specific, not a property of depth.

**Reported but not used for the replication call:** `x random` and top-1% recall.
Both depend on how mass is distributed in each pool and are not comparable across
models any more than across layers.

**On statistics.** With two independently trained models, no conventional
significance test about cross-model generalisation is appropriate — n = 2 cannot
support a population-level claim. A₂ delivers a clean preregistered replication plus
uncertainty *within* each model. Interventions are deterministic, which removes
repeated-measurement noise but **not** sampling uncertainty over which
features/contexts enter the pool — so bootstrap CIs across features remain
necessary and will be reported.

## Commitments

1. **A₂ is evaluated once, with the frozen configuration.** No tuning of the SAE,
   the acquisition rule, the target transform, or any threshold before the
   evaluation is run and recorded.
2. **Layer 3's calibration is not touched until after A₂.** Fixing it first would
   mean testing a different method than the one being replicated.
3. **A failed replication is reported as a failed replication**, not as a reason to
   adjust the protocol.
4. Any deviation forced by circumstance (crash, OOM, insufficient time) is recorded
   here with its reason.

## Deviations

**2026-09-08 — first A₂ run overshot, discarded and restarted.**
`train_a.py --steps` means "N *more* steps", not "until step N". Resuming from
75,000 with `--steps 83750` targeted 158,750; the run reached 130,000 before it was
caught — 55% more training than A₁.

Resolution: the run was **discarded, not used**. Rolling back was not clean either
(periodic snapshots do not store optimiser state, so resuming from 80,000 would have
restarted Adam mid-run). A₂ was retrained from scratch to exactly 83,750 with an
absolute `--until` flag added to make the error impossible to repeat. The overshot
run is archived at `checkpoints/a2_overshot/` and is **not** part of the
replication.

No evaluation was run on the overshot model, so no result was seen and discarded.

**2026-09-08 — final checkpoint was not saved; caught before any result.**
`train_a.py` only wrote checkpoints on multiples of `--ckpt-every`, and 83,750 is
not a multiple of 10,000. The retrained A₂ therefore stopped at 83,750 but left
`a2_latest.pt` at **80,000**. The four-layer evaluation was launched against that
80,000-step model and stopped ~90 seconds in, during layer 0's SAE training.

Resolution: the partial SAE was deleted; `train_a.py` now saves unconditionally on
exit; A₂ was resumed from `a2_latest.pt` (which does carry optimiser state, so Adam
continued rather than restarting) and completed to exactly **83,750, val 1.0228**.
**No metric was computed on the 80,000-step model** — the run was killed before any
harvest or discovery step, so nothing was seen.

---
Registered before A₂ training began. A₁ frozen at `milestones/v3/`, commit `a13d01d`.
