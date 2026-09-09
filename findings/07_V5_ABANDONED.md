# V5 — ABANDONED per its own preregistration

`docs/06_PREREGISTRATION_V5.md` section 3: *"A_ind may be retrained at most 3 times before
the benchmark is abandoned and reported as such."* Three attempts, three VOID
preconditions. **B was never run.** The benchmark is abandoned, as specified.

## The three attempts

| # | task design | learned it? | generalised? | canonical circuit? | verdict |
|---|---|---|---|---|---|
| 1 | tiled period, random L | **yes**, ratio 0.000 | — | **no** — both critical heads in L0 | VOID |
| 2 | `[P][filler][P]` | no, ratio plateaued 0.786 | — | no circuit at all | VOID |
| 3 | `[filler][P][P]` | **yes**, ratio 0.188 | **yes**, held-out L 0.90 vs 1.10 | **no** — see below | VOID |

Attempt 3 satisfied conditions 1 and 3 comfortably. It failed only condition 2:
`|C| >= 2` with one member per layer.

```
best previous-token score   L0H1  0.1894      threshold 0.5
best induction score        L1H3  0.1823      threshold 0.5
C = []   |C| = 0
```

## What attempt 3 actually built

A two-layer circuit **did** form. By ablation on second-copy positions
(baseline loss 0.797):

| head | loss when ablated | increase |
|---|---|---|
| **L1H3** | 12.908 | **+12.11** |
| **L0H3** | 12.549 | **+11.75** |
| L0H1 | 5.794 | +5.00 |
| L0H0 | 5.280 | +4.48 |
| L0H2 | 4.843 | +4.05 |
| L1H1 | 3.350 | +2.55 |
| L1H0 | 1.440 | +0.64 |
| L1H2 | 1.008 | +0.21 |

`L0H3 -> L1H3` is a genuine two-layer critical pair, and `L1H3` is the head with the
highest induction score (0.1823 — about 25x the ~0.008 a uniform head would give).
But `L0H3`'s previous-token score is 0.0368.

**So the model learned a functional, content-based, two-layer copying circuit whose
layer-0 component is not a previous-token head.** It generalises to unseen periods,
so it is doing content matching rather than positional memorisation — it simply does
not route that matching the canonical way.

## Two bugs found and fixed along the way

1. **Task (Rev 4/5).** A tiled period let the model exploit *periodicity* instead of
   one-shot content matching. Attempt 1 solved the task perfectly with both critical
   heads in layer 0 and 0.0% of layer-1 attention on correct-token positions.
2. **Scoring.** When the layout changed to `[filler][P][P]`, the prior-occurrence
   index stayed at `t-(T-L)` instead of `t-L`, pointing into the filler — a position
   that can never be the answer. Fixing it moved `L1H3` from 0.0001 to **0.1823**, so
   the bug was material. Section 2's *definition* never changed; only its
   implementation was wrong.

The second bug is worth dwelling on: without it, the record would have said
"no induction head at all" when in fact there is one at 25x baseline.

## What this does and does not mean

**It does not mean B failed.** B never ran. No B result was produced, seen, or
discarded. The mechanistic-correctness hypothesis is **untested**, not falsified.

**V4 is unaffected.** Its claim — B finds causally load-bearing structure ~10-25x
faster than random, replicated across two models — rests on interventions against
frozen A₁/A₂ and needs no known-circuit benchmark.

**The honest finding is about the benchmark, not about B:** across three task
designs, a 2-layer attention-only transformer learned content-based copying (verified
by held-out-period generalisation) without producing a canonical previous-token /
induction-head pair at threshold 0.5. That is either a real fact about circuit
diversity in this regime, or evidence that these two scores at that threshold are the
wrong instrument for these task variants. **These attempts cannot distinguish those**,
and lowering the threshold now — after seeing three failures — is exactly the
post-hoc threshold-shopping the preregistration exists to prevent.

## Deviations, recorded

Section 3 permitted retraining **with a different seed**. I redesigned the task
twice (Rev 4, Rev 5). That exceeds what was preregistered and is recorded as a
deviation, not presented as a permitted retrain. Two facts bound its impact: B was
never run, and the evaluation (sections 4-9) was never touched.

## What a future attempt would need — not done here

Any V6 would need a **fresh preregistration**, written before seeing anything, and
would have to choose deliberately between:

- reproducing the literature's task exactly (fixed offset), accepting that the
  identifiability problem returns, and adding a separate positional-head control; or
- keeping the identifiable task and prespecifying an instrument that does not assume
  the canonical decomposition — e.g. scoring "does layer 0's output causally enable
  layer 1's content match", which the ablation above already suggests is the right
  question, rather than "is layer 0 a previous-token head".

Artefacts: `checkpoints/aind_latest.pt` (attempt 3, step 180,000),
`checkpoints/aind_void1.pt`, `checkpoints/aind_void2.pt`,
`findings/v5_groundtruth.json`, `findings/v5_void{1,2}_groundtruth.json`
