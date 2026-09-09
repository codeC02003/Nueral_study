# ERRATA — V4

V4 is frozen. Its files are **not** edited. Corrections are recorded here instead,
the way an erratum accompanies a published paper rather than rewriting it.

## 1. Terminology — "mechanisms" overstated the claim

`README.md` and `MANIFEST.json` in this milestone describe B as finding *"causally
important mechanisms."*

**"Mechanism" implies that B's description corresponds to the algorithm A actually
implements. V4 does not establish that**, and says so elsewhere in its own scope
limits. The precise wording is:

> B finds causally important internal **features and interventions** roughly 10–25x
> more efficiently than random search.

"Mechanism" is reserved for the stronger claim that V5 was built to test and did not
test (V5 was abandoned — its instrument never became valid, and B was never run).

This is a wording correction. **No number in V4 changes.**

## 2. "Cannot be faked" overstated the prediction result

Some V4-era text argued that predicting intervention magnitude "cannot be faked" and
that "only a real model of A" could do it.

That is stronger than the evidence supports. A predictor can exploit statistical
regularities of A's activations without representing A's algorithm. The precise
statement is:

> Accurate held-out causal-effect prediction is substantially stronger evidence than
> salience ranking, because it requires generalising quantitative counterfactual
> effects to unseen features and contexts. It does not establish mechanistic
> correspondence.

## 3. Headline precision

Report the discovery advantage as **~10–25x** (A₁ layer 2: 16.0x, 95% CI [9.6, 24.5]),
not as the point estimate 12.45x that appears in earlier V3-era text.

---

Current authoritative claims: [`../../docs/03_CLAIMS.md`](../../docs/03_CLAIMS.md).
