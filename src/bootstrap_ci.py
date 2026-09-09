#!/usr/bin/env python3
"""
Hierarchical (cluster) bootstrap CIs for every headline number.

    python3 bootstrap_ci.py --reps 5000

WHY A CLUSTER BOOTSTRAP AND NOT A ROW BOOTSTRAP

The rows are not independent draws. Every context of feature 515 shares that
feature's decoder direction, its firing statistics and its logit reach, so a naive
row-level resample treats ~300 correlated observations as ~300 independent ones and
produces intervals that are far too narrow.

The data have two levels, so the resample does too:

    1. resample FEATURES with replacement          (the cluster)
    2. within each sampled feature, resample its CONTEXTS with replacement

Discovery metrics are recomputed on the resampled pool: the oracle bound is the
top-N rows by |delta-loss| within that pool at the same N, so the normalisation
travels with the resample rather than being held fixed.

    3. resample the ACQUISITION TRAJECTORY (seed) with replacement

The third level matters for a reason that is easy to miss. The reported figures are
means over 3 acquisition seeds, so an interval conditional on ONE trajectory is an
interval on a different estimator - it would understate the uncertainty of the
number actually being reported. Resampling seeds makes the CI an interval on the
same quantity. With only 3 seeds the seed-variance component is itself crudely
estimated; that is stated rather than hidden.

Calibration metrics are recomputed the same way on the held-out features, from the
stored per-row predictions.

WHAT THIS DOES AND DOES NOT ANSWER

It quantifies sampling variation over features and contexts *within* a model. It
does **not** turn n = 2 models into a population-level claim - no test here says
anything about generalisation across models. It only tells you whether a gap like
39.4% -> 26.4% is larger than the noise in either measurement.
"""
import os, sys, json, glob, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

def load(m, l):
    p = f"{ROOT}/experiments/boot/{m}_L{l}.npz"
    return np.load(p) if os.path.exists(p) else None

def _ragged(idx, feat):
    """Group row positions by feature into a flat array + offsets, so the two-level
    resample can be done with array ops instead of a Python loop per feature."""
    f = feat[idx]
    order = np.argsort(f, kind="stable")
    fs = f[order]
    uf, start, cnt = np.unique(fs, return_index=True, return_counts=True)
    return order.astype(np.int64), uf, start.astype(np.int64), cnt.astype(np.int64)

def _resample(order, start, cnt, nf, rng):
    """Level 1: sample nf features with replacement.
       Level 2: within each, sample cnt[f] contexts with replacement.
       Fully vectorised - the loop version was ~1000 rng calls per rep."""
    sel = rng.randint(0, len(cnt), nf)
    c = cnt[sel]
    total = int(c.sum())
    if total == 0: return None
    starts = np.repeat(start[sel], c)
    ns = np.repeat(c, c)
    picks = starts + (rng.random_sample(total) * ns).astype(np.int64)
    return order[picks]

def stats_for(z, rng, reps):
    feat, absy = z["feature"], z["absy"]
    NS = int(z["n_seeds"]) if "n_seeds" in z else 1
    # precompute the per-seed structures once
    S = []
    for sd in range(NS):
        pool, acq = z[f"pool{sd}"], z[f"acq{sd}"]
        rnd, te = z[f"rnd{sd}"], z[f"test{sd}"]
        po, _, ps, pc = _ragged(pool, feat)
        to, _, ts, tc = _ragged(te, feat)
        S.append(dict(po=po, ps=ps, pc=pc, to=to, ts=ts, tc=tc,
                      abs=absy[pool], acq=np.isin(pool, acq), rnd=np.isin(pool, rnd),
                      pred=z[f"pred{sd}"], actual=z[f"actual{sd}"]))

    out = {k: [] for k in ["oracle","xrand","top1","slope","r2"]}
    for _ in range(reps):
        d = S[rng.randint(0, NS)]          # level 3: resample the acquisition seed
        p_order, p_start, p_cnt = d["po"], d["ps"], d["pc"]
        t_order, t_start, t_cnt = d["to"], d["ts"], d["tc"]
        pool_abs, in_acq, in_rnd = d["abs"], d["acq"], d["rnd"]
        pred, actual = d["pred"], d["actual"]
        rows = _resample(p_order, p_start, p_cnt, len(p_cnt), rng)
        if rows is None or len(rows) < 50: continue
        a_ = pool_abs[rows]; am = in_acq[rows]; rm = in_rnd[rows]
        n_acq = int(am.sum())
        if n_acq < 10: continue
        # oracle bound recomputed inside the resampled pool, at the same count
        part = np.partition(a_, -n_acq)[-n_acq:]
        out["oracle"].append(a_[am].sum() / max(part.sum(), 1e-12))
        tot = max(a_.sum(), 1e-12)
        mr = a_[rm].sum() / tot
        if mr > 0: out["xrand"].append((a_[am].sum()/tot) / mr)
        k1 = max(1, int(0.01*len(rows)))
        top = np.argpartition(-a_, k1-1)[:k1]
        out["top1"].append(am[top].mean())

        trow = _resample(t_order, t_start, t_cnt, len(t_cnt), rng)
        if trow is None or len(trow) < 20: continue
        p, a = pred[trow], actual[trow]
        pc, ac = p-p.mean(), a-a.mean()
        den = float((pc**2).sum())
        if den > 1e-12: out["slope"].append(float(pc@ac/den))
        tt = float((ac**2).sum())
        if tt > 1e-12: out["r2"].append(1-float(((a-p)**2).sum())/tt)
    return {k: np.array([v for v in vs if np.isfinite(v)]) for k, vs in out.items()}

def ci(v):
    return (np.nanmean(v), np.nanpercentile(v,2.5), np.nanpercentile(v,97.5)) if len(v) else (np.nan,)*3

rng = np.random.RandomState(args.seed)
res = {}
print(f"  hierarchical bootstrap, {args.reps:,} reps  (resample features, then contexts within)\n")
for m in ("a","a2"):
    for l in range(4):
        z = load(m,l)
        if z is None: continue
        res[f"{m}_L{l}"] = {k: ci(v) for k, v in stats_for(z, rng, args.reps).items()}
        print(f"    {m}_L{l} done", flush=True)

NM = [("oracle","oracle %",100),("xrand","x random",1),("top1","top-1% recall",100),
      ("slope","calib slope",1),("r2","R2",1)]
for key, label, sc in NM:
    print(f"\n  {label}   mean [95% CI]")
    print(f"    {'layer':>6}   {'A1':^26}   {'A2':^26}   overlap?")
    for l in range(4):
        r1 = res.get(f"a_L{l}",{}).get(key); r2 = res.get(f"a2_L{l}",{}).get(key)
        if not r1 or not r2: continue
        ov = "yes" if (r1[1] <= r2[2] and r2[1] <= r1[2]) else "NO"
        f = lambda t: f"{t[0]*sc:7.2f} [{t[1]*sc:6.2f},{t[2]*sc:6.2f}]"
        print(f"    {l:>6}   {f(r1)}   {f(r2)}   {ov}")
json.dump({k:{m:[float(x) for x in t] for m,t in v.items()} for k,v in res.items()},
          open(f"{ROOT}/findings/bootstrap_ci.json","w"), indent=1)
print(f"\n  -> findings/bootstrap_ci.json")
