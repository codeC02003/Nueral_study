# V5 — GROUND-TRUTH BENCHMARK ABANDONED / INSTRUMENT INVALID

**FROZEN. This is not a mechanistic-validation milestone.**

## The frozen conclusion

> **V5 did not test whether B recovers A's true algorithm, because the preregistered
> ground-truth instrument failed its own validity condition. No
> mechanistic-understanding claim is supported or rejected.**

**B was never run.** Not once, at any stage. No B result was produced, seen, or
discarded.

## Why

The benchmark is gated on a validity precondition (section 3): the subject model
must contain the canonical circuit the test scores against — `|C| >= 2`, one member
per layer, threshold 0.5. Across three attempts that condition never held, so there
was nothing valid to score B against. Section 3 specifies abandonment after three
attempts. That is what happened.

## What was learned anyway

Attempt 3's model **did** learn content-based copying — verified by generalising to
held-out repeat periods (0.9045 vs a 1.1026 threshold), so it is matching content,
not memorising positions. And a **real two-layer circuit formed**: `L0H3 -> L1H3`,
ablation +11.75 and +12.11 against a baseline loss of 0.797, with `L1H3` carrying
the highest induction score at roughly 25x the uniform baseline.

But `L0H3` is not a previous-token head (0.0368), and neither head clears 0.5. The
model routes content matching in a **non-canonical** way.

Whether that is genuine circuit diversity or the wrong measuring instrument, these
three attempts **cannot distinguish**. Lowering the threshold after seeing three
failures was declined as post-hoc threshold-shopping.

## What is explicitly NOT concluded

- **Not** that B fails to recover circuits — B never ran.
- **Not** that the canonical induction circuit doesn't exist — it is well established
  in the literature on the standard fixed-offset task.
- **Not** that the 0.5 thresholds are miscalibrated — declining to move them was a
  methodological choice, not a measurement.

## V4 is untouched

V4's claim rests on interventions against frozen A₁/A₂ and needs no known-circuit
benchmark. Nothing here edits it. That separation is why V5 was made a distinct
milestone in the first place.

## Contents

```
MANIFEST.json          the frozen conclusion, all three attempts, the scorer bug
PREREGISTRATION_V5.md  5 revisions, every one written before training
findings/
  V5_ABANDONED.md      full report, ablation table, V6 requirements
  attempt{1,2,3}_groundtruth.json
  v5_ind.log, v5_ind2.log     raw training logs
checkpoints/
  aind_attempt1_tiled.pt          solved the task WITHOUT the circuit
  aind_attempt2_P_filler_P.pt     never learned the task
  aind_attempt3_step180000.pt     learned it, non-canonical circuit
scripts/               train_ind.py + nn/
```
