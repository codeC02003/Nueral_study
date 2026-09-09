# V4 — FROZEN

Cross-model replication with quantified uncertainty. **Do not modify.**
The induction ground-truth benchmark is a separate milestone (V5) and must be able
to fail without editing anything here.

## What is claimed

**B, given no human labels, finds a frozen language model's causally important
mechanisms roughly 10-25x faster than random search, calibrated to within the
resolution of the data. Replicated on a second independently trained model.**

## What is not claimed

- Population-level generality — n = 2 models.
- Any depth trend in calibration or discovery — not resolvable at this sample size.
- That B's decomposition corresponds to A's actual algorithm — **unvalidated**.
- Any semantic reading of individual features.

## Contents

```
MANIFEST.json         headline numbers, scope limits, reproduction commands
CLAIMS.md             four claims, separately graded
METRICS.md            both metrics in full, including the failed first definition
PREREGISTRATION_A2.md the A2 protocol, with both deviations logged
findings/             A2_REPLICATION, BOOTSTRAP_CI, V3_*, RESULTS, per-layer JSON
checkpoints/          A1, A2, and all 8 SAEs (4 layers x 2 models)
scripts/              everything needed to reproduce, including nn/
raw/                  the row-level arrays the bootstrap resamples
```
