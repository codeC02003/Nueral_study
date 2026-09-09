# V5 preregistration — known-circuit ground-truth benchmark

**Written before A_ind exists.** Nothing below may change after any V5 result is
seen. V4 is frozen and this cannot edit it: if V5 fails, V4's causal-discovery
claim stands and only the stronger mechanistic claim falls.

---

## 0. The hypothesis under test

V4 established that B finds **causally load-bearing** structure. It did not
establish that B's decomposition **corresponds to the algorithm A actually runs**.

> **H1 (tested here):** B recovers the components of a known circuit, without being
> told what to look for.

## 1. The subject — A_ind

**Task.** Synthetic induction, **pattern-filler-pattern**. For each sequence:

```
L ~ Uniform{20..50}
P        = L distinct tokens sampled without replacement from a 64-symbol vocab
filler   = 128 - 2L tokens sampled from the vocab MINUS P
sequence = [ P ][ filler ][ P ]        length 128
```

Only the second copy of `P` is induction-predictable. Each of its tokens has
**exactly one** prior occurrence, at offset `128 - L`, which varies with `L`.

**Why the period must vary — this is not a robustness nicety.** With a fixed period
of 64, `token[t+1] == token[t-63]` always. A purely *positional* head that attends
to `t-63` would then produce **exactly the same attention pattern** as a genuine
induction head, and the `IND` score in section 2 would award it 1.0. The ground
truth would be unidentifiable and the benchmark would pass while measuring nothing.
Varying `L` per sequence means no fixed positional offset solves the task, so a
high `IND` score can only come from content matching.

**Why the base period must be duplicate-free.** Sampling `L` tokens *with*
replacement lets a token appear twice inside one period with two **different**
successors. The induction target would then be ambiguous, and a perfect induction
head could be penalised by `IND` for attending to the other equally valid match.
Sampling without replacement (possible since `L <= 60 < 64`) means every token
occurs exactly once per period, so each repeated token has exactly one correct
previous occurrence and exactly one correct successor.

**Held-out periods.** `L in {35, 45}` is excluded from training and used as a test
set, so the precondition below can check content-matching directly rather than
assuming it.

**Why the pattern is not tiled — Rev 4.** The first design tiled the base period
5-6 times across the sequence. A_ind solved that task perfectly (repeated-position
loss 0.0003) **without forming the induction circuit**: ablation showed both
critical heads were in layer 0 (+7.30 and +5.69 loss) with layer 1 contributing
almost nothing, and no layer-1 head attended to a position holding the correct next
token (0.0%). Repeating the period many times creates *periodicity*, which is a
different and easier structure to exploit than one-shot content matching. Exactly
one prior occurrence removes it.

**The four protections together:**

```
random L         -> a fixed positional offset cannot solve the task
distinct tokens  -> the content-match target is unambiguous
one occurrence   -> periodicity cannot substitute for induction
filler disjoint  -> no accidental matches outside the pattern
held-out L       -> content-matching is tested, not assumed
```

**Architecture.** 2 layers, 4 heads per layer, d = 128, **attention-only (no MLP)**.
Attention-only is chosen because the induction circuit's ground truth is exact in
that setting.

**The circuit that is known to form** (Elhage et al. 2021; Olsson et al. 2022):

```
layer 0   PREVIOUS-TOKEN HEAD   position t attends to t-1,
                                writing "the token before me" into the residual stream
              |
              |  K-composition
              v
layer 1   INDUCTION HEAD        query = current token; matches keys that encode
                                "token before me"; attends to j+1 where token[j] = token[t];
                                OV copies token[j+1] to the output
```

## 2. What counts as ground truth — fixed now, computed without B

Both scores are properties of A_ind's attention alone. They are computed **before**
B is run and never adjusted.

**Previous-token score**, head h in layer 0:
`PTS(h) = mean over positions t>=1 of attention(t -> t-1)`

**Induction score** (prefix-matching), head h in layer 1:
`IND(h) = mean over second-copy positions t of attention(t -> t_first+1)`
where `t_first = t - (128 - L)` is the token's unique earlier occurrence. Because
`L` varies per sequence, `t - t_first` is not constant, so this score cannot be
achieved by a fixed positional offset; and because the occurrence is unique, the
target is unambiguous.

**The ground-truth set C** = every layer-0 head with `PTS > 0.5`, plus every
layer-1 head with `IND > 0.5`.

**Threshold justification, fixed in advance:** 0.5 means the head places more than
half its attention mass on the single predicted position out of up to 128 — far
above the ~0.008 a uniform head would give.

## 3. Validity precondition — the test is void if this fails

A_ind must actually learn induction. Before B is run:

1. `loss(second-copy positions) < 0.5 x loss(first-copy + filler positions)`
2. `|C| >= 2`, with at least one layer-0 and one layer-1 member
3. **Content-matching, not positional.** Loss on the held-out periods
   `L in {35, 45}` must be within 1.25x of loss on trained periods. A model that
   memorised positional offsets would fail on unseen periods; a model doing content
   matching would not.

If any fails, **the run is void** — it tests the task setup, not B. It is reported
as void, not as a failure of B. A_ind may be retrained (different seed) at most 3
times before the benchmark is abandoned and reported as such.

**A void run permits fixing the SUBJECT, never the EVALUATION.** Sections 4-9 —
the feature-to-head mapping, primary score, controls, thresholds, pipeline and
failure interpretation — are frozen and were not touched by Rev 4.

## 4. Mapping B's output onto heads — fixed now

B ranks **(feature, context)** pairs; ground truth is about **heads**. The mapping
is prespecified so it cannot be chosen to flatter the result.

For each of the 8 heads h, and each SAE feature f, attribute by ablation:
`attr(f, h) = |mean activation of f| - |mean activation of f with head h ablated|`
Feature f is **assigned** to `argmax_h attr(f, h)`.

**B-attributed mass of head h** = the sum of |Δloss| over B's acquired experiments
whose feature is assigned to h.

Heads are then ranked by B-attributed mass.

## 5. Primary score — fixed now

> **precision@2 = |{top-2 heads by B-attributed mass} ∩ C| / min(|C|, 2)**

**Chance level, general form.** With 8 heads and `|C| >= 2`, drawing 2 heads at
random gives precision@2 = 1.0 with probability

```
P = C(|C|, 2) / C(8, 2) = C(|C|, 2) / 28
```

| \|C\| | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| chance | 3.6% | 10.7% | 21.4% | 35.7% |

`|C|` is fixed and recorded in section 2 **before B is run**, so the applicable
chance level is known in advance and is not chosen afterwards. If `|C| >= 5` the
benchmark is **weak by construction** (chance exceeds 1 in 3) and will be reported
as such rather than as a pass.

## 6. Controls

1. **Random directions** — ablate magnitude-matched random directions instead of SAE
   features (as in V4), to confirm the features are load-bearing on this task too.
2. **Random experiment selection** — the same pipeline with random acquisition,
   scored identically. Isolates the contribution of B's *selection*.
3. **Random head ranking** — the 3.6% chance baseline above, stated analytically.

## 7. Pass / fail thresholds — fixed now, before any result

A single random trajectory is a fragile comparator — one lucky ranking would decide
the verdict. So the criterion is a **recovery frequency across acquisition seeds**.

**5 acquisition seeds** (not V4's 3). The harvest is shared across seeds, so the
extra cost is two short trajectories; and since the primary score is now
frequency-based, seed count directly sets its resolution. This is a prespecified
strengthening of the evaluation, not a change to the method.

```
B_freq  = fraction of the 5 seeds in which B      reaches precision@2 = 1.0
R_freq  = fraction of the 5 seeds in which random reaches precision@2 = 1.0
```

| outcome | condition |
|---|---|
| **PASS** | `B_freq >= 0.8` **and** `B_freq - R_freq >= 0.4` |
| **RECOVERED, SELECTION IRRELEVANT** | `B_freq >= 0.8` **and** `R_freq >= 0.8` — the circuit is findable, but B's *selection* contributed nothing. Does **not** support H1's active-discovery component. |
| **PARTIAL** | `B_freq >= 0.4` **and** `B_freq > R_freq` |
| **FAIL** | `B_freq < 0.4` |
| **VOID** | the section-3 precondition fails |
| **WEAK BY CONSTRUCTION** | `\|C\| >= 5` (chance level above 1 in 3) |

Also reported, as continuous secondaries: **mean precision@2 across seeds** for both
B and random, and the per-seed head rankings.

## 8. Pipeline freeze

Identical to V4. No component may be re-tuned after seeing any V5 number:

```
SAE               40,000 steps · expansion 8 · l1 = 1e-2 · same calibration
acquisition       mag80  (80% predicted-magnitude, 20% random)
target            log-magnitude, sign(y)·log1p(|y|)
budget            2,400 contexts · step 200 · 5 heads · 300 epochs · **5 seeds**
split             by feature, 20% held out
intervention      mean-ablation
tap               layer-1 residual stream (the deepest, as A_ind has 2 layers)
uncertainty       three-level bootstrap: features, contexts, acquisition seed
```

Two things V4 taught that are carried in rather than rediscovered: **discovery
metrics are robust to acquisition seed, calibration metrics are not** (2.2-2.8x
wider), so calibration figures here will be reported with three-level CIs and
treated as low-resolution unless many more seeds are run.

## 9. How failure will be interpreted — fixed now

**If B finds causal effects but fails to recover the circuit:**
V4 stands. B demonstrably finds causally load-bearing structure and finds it faster
than random search. What fails is the stronger hypothesis that B's decomposition
corresponds to A's actual algorithm. That would be a real and publishable negative
result about SAE-based decomposition, not a failure of the V4 measurement.

**If the precondition fails:** void. Nothing is learned about B.

**If B recovers the circuit:** H1 is supported *on a two-layer attention-only model
with a single known circuit*. It does **not** generalise to A₁/A₂, which have MLPs,
four layers, and no known ground truth. That limitation is stated now so it cannot
be quietly dropped later.

## 10. Commitments

1. A_ind is trained **after** this document is final.
2. Ground-truth scores (PTS, IND, set C) are computed and **written down before B is
   run**.
3. One evaluation. No tuning of the SAE, acquisition rule, target transform, tap
   layer, or any threshold after seeing a V5 number.
4. A void, partial, or failed result is reported as such.
5. V4 is not edited by anything here.

## Revision history

**Rev 4, after a VOID run and before B was ever run.** The tiled-period task was
replaced with pattern-filler-pattern (exactly one prior occurrence). The first
A_ind solved the tiled task perfectly *without* the induction circuit — ablation
put both critical heads in layer 0 (+7.30, +5.69) and no layer-1 head attended to a
correct-token position (0.0%). Periodicity was substituting for content matching.

**This is a deviation beyond what section 3 allowed.** Section 3 permitted
retraining with a different seed, not redesigning the task. Recorded as such rather
than presented as a permitted retrain. Two facts bound its impact: **B had not been
run and no B result had been seen**, and the evaluation (sections 4-9) is unchanged
— only the subject was fixed.

**Rev 3, before A_ind existed.** Base period is now sampled **without
replacement**. With replacement, a token could appear twice in one period with two
different successors, making the induction target ambiguous and penalising a
correct head for choosing the other valid match. `L <= 60 < 64` so uniqueness is
always achievable.

**Rev 2, before A_ind existed.** Three fixes, all pre-training:
1. **Randomised repeat period.** The fixed 64-offset design made the ground truth
   unidentifiable — a positional head attending to `t-63` would have scored
   `IND = 1.0` identically to a real induction head. Added held-out periods to test
   content-matching directly.
2. **Chance level generalised** to `C(|C|,2)/28`; the `|C| = 2` figure of 3.6% only
   held for that case. Added a "weak by construction" outcome for `|C| >= 5`.
3. **Recovery frequency across 5 acquisition seeds** replaces a single fragile
   B-vs-random comparison.

## Outcome

**ABANDONED after 3 VOID attempts, per section 3.** B was never run; the
mechanistic hypothesis is untested, not falsified. See `findings/07_V5_ABANDONED.md`.

## Deviations

**2026-09-09 — VOID run 3 (Rev 5), benchmark abandoned.** Conditions 1 and 3
passed (ratio 0.188; held-out periods 0.90 vs 1.10) but `|C| = 0`: best
previous-token score 0.1894, best induction score 0.1823, threshold 0.5. Ablation
showed a real two-layer critical pair (L0H3 +11.75, L1H3 +12.11) whose layer-0
member is not a previous-token head.

**2026-09-09 — scoring bug fixed (not a definition change).** With the Rev 5
layout the prior-occurrence index remained `t-(T-L)` instead of `t-L`, pointing
into the filler. Fixing it moved L1H3 from 0.0001 to 0.1823. Section 2's definition
was unchanged; only its implementation was wrong.

**2026-09-08 — VOID run 2 (Rev 5 task change).** `[P][filler][P]` plateaued at
ratio 0.786 with no circuit. Layout changed to `[filler][P][P]` to shorten the
lookback from ~78-108 tokens to L. A second task redesign, again beyond section 3's
seed-retrain allowance.

**2026-09-08 — VOID run 1, task redesigned (Rev 4).** Detail above. The first
A_ind's checkpoint and ground truth are archived at
`findings/v5_void1_groundtruth.json` rather than deleted, since a model that solves
induction *without* an induction circuit is itself a result worth keeping.
