# NOT NOW

Good ideas, deferred on purpose. They are dangerous *because* they are good — each
one can quietly become its own research project and displace the single question
this project is actually trying to answer:

> Can B predict causal effects inside frozen A, and choose experiments that improve
> that prediction faster than random selection?

Nothing below is needed to answer that. Each entry has a **trigger** — the condition
that would make it worth picking up — so it can be judged rather than remembered.

| Deferred | Trigger to reconsider |
|---|---|
| **Crosscoders / cross-layer transcoders** | after per-layer SAEs work on L0–L3 and the "same feature or four features?" question actually blocks a conclusion |
| **Weight-space reader (old Tier 3)** | demoted from a required tier to a separate research question. Needs a population of 50–100 trained A's plus permutation invariance. Do last, or not at all. |
| **JumpReLU / Gated / BatchTopK SAEs** | only if *causal coverage* (not reconstruction error) proves to be the bottleneck |
| **HVP attribution correction** (arXiv:2606.09899) | only if exact patching gets slow. Trigger: one circuit trace exceeding ~1 hour; currently ~18 s. Note the paper flags SAE-encoder nonlinearity as an uncharacterised extra curvature source — which is exactly our setup. |
| **Full path patching / circuit tracing** | after B is calibrated. Tracing is worthless if the effect estimates it traces are wrong. |
| **Developmental replay across the 48 anchors** | after the loop works on one frozen A. Genuinely novel, and the anchors and tooling already exist — so it keeps. |
| **RL / bandit agent** | greedy argmax-uncertainty first. If greedy beats random, an agent is an optimisation, not a prerequisite. |
| **Ground truth (Tracr, induction task)** | *external validation after* the loop works, not a parallel subsystem. Tracr's A never self-learned its algorithm, so it can only ever be a microscope unit-test, never the main A. |
| **Scaling A** | not until the loop works. Being small is buying real methodological simplicity: exact ablation is affordable, so we get the gold-standard causal measurement that larger projects have to approximate. |
| **Human-readable explanations from B** | an open design question, not a task. A correct internal causal model may be sufficient. |

## Two design questions still open

1. **May B modify A during discovery**, or only observe and intervene transiently?
   (Currently: transient interventions only; A's weights are never touched.)
2. **Must B's output be human-readable**, or is a correct internal causal model
   enough even if B's latents are not individually interpretable?

Both shape what comes after V2. Neither blocks V2.
