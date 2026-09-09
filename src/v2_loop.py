#!/usr/bin/env python3
"""
V2 STEPS 2-4 - B predicts contextual causal effects, and chooses its own experiments.

    python3 v2_loop.py                  # the full experiment
    python3 v2_loop.py --budget 400     # bigger experiment budget

Answers exactly one question:

    Can B predict the causal effect of unseen interventions on frozen A, and by
    selecting its own experiments, reach a given accuracy with FEWER experiments
    than random selection?

STEP 2 - what B predicts
    old:  feature            -> mean delta-loss        (R2 = 0.327, wrong target)
    new:  (feature, context) -> delta-loss at THAT position

    Justified by measurement: 96.3% of a feature's effect lands in 1% of positions,
    so the scalar average discarded ~99% of the signal.

STEP 3 - uncertainty
    Five small heads, differently initialised, on different bootstrap resamples.
    Their disagreement is the uncertainty estimate. No Bayesian machinery.

STEP 4 - agency
    B picks the experiment where its ensemble disagrees most, runs it, refits.
    The control picks uniformly at random. Same budget, same held-out test set.
    The gap between the two learning curves is the result.

SPLIT BY FEATURE, always. Test contexts belong to features B has never queried,
so nothing can be answered by memorising a feature's average.
"""
import os, sys, json, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=300, help="experiments each strategy may run")
ap.add_argument("--step", type=int, default=20, help="features acquired per round")
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=260, help="refit epochs per round")
ap.add_argument("--seeds", type=int, default=3, help="repeats, for error bars")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

# ------------------------------------------------------------------ data
z = np.load(f"{ROOT}/experiments/contextual.npz")
meta = json.load(open(f"{ROOT}/data/meta.json")); V = len(meta["vocab"])
sys.path.insert(0, ROOT)
from nn.gpt import GPT, Config
from nn.sae import SAE
cfg = Config(vocab_size=V)
B_sae = SAE(cfg.n_embd)
B_sae.load_state_dict(torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location="cpu")["model"])
A_gpt = GPT(cfg)
A_gpt.load_state_dict(torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location="cpu")["model"])

feat = z["feature"].astype(np.int64)
y    = z["d_mean"].astype(np.float32)

# ---- the CONTEXT part of (feature, context) ----
with torch.no_grad():
    Wu = A_gpt.head.weight                                  # (vocab, n_embd)
    Wdec = B_sae.W_dec                                      # (n_feat, n_embd)
    logit = Wdec @ Wu.T                                     # what each feature pushes toward
    FEAT_STATIC = torch.cat([
        Wdec, B_sae.W_enc.T, B_sae.b_enc.unsqueeze(1),
        Wdec.norm(dim=1, keepdim=True), logit.norm(dim=1, keepdim=True),
        logit.abs().max(dim=1, keepdim=True).values,
        logit.topk(4, dim=1).values,
    ], dim=1).numpy().astype(np.float32)                    # per-feature properties

def build_X(idx):
    """Per-row inputs: the feature's static properties + THIS context."""
    f = feat[idx]
    ctx = np.stack([
        z["act"][idx], np.log1p(z["act"][idx]),
        z["base"][idx],                                     # how hard A already finds this token
        z["pos"][idx] / cfg.block_size,
        logit[f, z["tgt"][idx].astype(np.int64)].numpy(),    # does the feature push TOWARD the answer?
        logit[f].numpy().max(1),
    ], axis=1).astype(np.float32)
    return np.concatenate([FEAT_STATIC[f], ctx], axis=1)

ALLF = np.unique(feat)
ROWS = {int(f): np.nonzero(feat == f)[0] for f in ALLF}

# ------------------------------------------------------------------ model
class Head(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d, 96), nn.GELU(), nn.Dropout(.1),
                            nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
    def forward(s, x): return s.n(x).squeeze(-1)

class Ensemble:
    """Disagreement across differently-seeded, differently-resampled heads = uncertainty."""
    def __init__(s, d, n, seed):
        s.h = []
        for k in range(n):
            torch.manual_seed(seed*1000 + k)
            s.h.append(Head(d).to(DEV))
        s.d = d
    def fit(s, X, Y, epochs):
        s.mu, s.sd = X.mean(0), X.std(0).clamp(min=1e-6)
        s.ym, s.ys = Y.mean(), Y.std().clamp(min=1e-9)
        Xn, Yn = (X-s.mu)/s.sd, (Y-s.ym)/s.ys
        for k, h in enumerate(s.h):
            g = torch.Generator(device="cpu").manual_seed(k)
            bi = torch.randint(0, len(Xn), (len(Xn),), generator=g).to(DEV)   # bootstrap
            o = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-2)
            h.train()
            for _ in range(epochs):
                o.zero_grad(set_to_none=True)
                F.smooth_l1_loss(h(Xn[bi]), Yn[bi], beta=.5).backward(); o.step()
    @torch.no_grad()
    def pred(s, X):
        Xn = (X-s.mu)/s.sd
        p = torch.stack([h.eval()(Xn) for h in s.h])
        return (p.mean(0)*s.ys + s.ym), (p.std(0)*s.ys)      # mean, disagreement

# ------------------------------------------------------------------ the experiment
def run(seed):
    rng = np.random.RandomState(seed)
    fs = ALLF.copy(); rng.shuffle(fs)
    test_f, pool_f = fs[:200], fs[200:]                      # test features NEVER queried
    te = np.concatenate([ROWS[int(f)] for f in test_f])
    Xte = torch.tensor(build_X(te), device=DEV); Yte = torch.tensor(y[te], device=DEV)

    def metrics(ens):
        p, _ = ens.pred(Xte)
        mae = (p-Yte).abs().mean().item()
        # calibration slope: regress actual on predicted. 1.0 = perfectly calibrated
        pc, ac = p-p.mean(), Yte-Yte.mean()
        slope = (pc@ac / pc.pow(2).sum().clamp(min=1e-12)).item()
        r = torch.corrcoef(torch.stack([p, Yte]))[0,1].item()
        r2 = 1 - (Yte-p).pow(2).sum().item()/max((Yte-Yte.mean()).pow(2).sum().item(),1e-12)
        return mae, slope, r, r2

    out = {}
    for strat in ("B chooses", "random"):
        acq = list(rng.choice(pool_f, args.step, replace=False))   # identical seed start
        curve = []
        while len(acq) <= args.budget and not stop:
            idx = np.concatenate([ROWS[int(f)] for f in acq])
            Xtr = torch.tensor(build_X(idx), device=DEV); Ytr = torch.tensor(y[idx], device=DEV)
            ens = Ensemble(Xtr.shape[1], args.heads, seed)
            ens.fit(Xtr, Ytr, args.epochs)
            m = metrics(ens); curve.append((len(acq),) + m)
            print(f"    seed {seed} {strat:<10} {len(acq):>4} features  MAE {m[0]:.4f}  slope {m[1]:.3f}  R2 {m[3]:+.3f}")
            if len(acq) >= args.budget: break
            rest = np.array([f for f in pool_f if f not in acq])
            if strat == "random":
                pick = rng.choice(rest, min(args.step, len(rest)), replace=False)
            else:
                # B tests where its heads disagree most. Score every candidate in
                # ONE batched pass - the per-feature Python loop was 50x slower.
                cand_rows = np.concatenate([ROWS[int(f)][:60] for f in rest])
                owner     = np.concatenate([np.full(min(60, len(ROWS[int(f)])), j)
                                            for j, f in enumerate(rest)])
                _, u = ens.pred(torch.tensor(build_X(cand_rows), device=DEV))
                u = u.cpu().numpy()
                sc = np.bincount(owner, weights=u, minlength=len(rest)) / \
                     np.bincount(owner, minlength=len(rest)).clip(min=1)
                pick = rest[np.argsort(-sc)[:args.step]]
            acq += list(pick)
        out[strat] = curve
    return out

print = __import__("functools").partial(print, flush=True)
print(f"V2 experiment   budget {args.budget} features   {args.heads} heads   {args.seeds} seeds   {DEV}")
print(f"dataset: {len(y):,} (feature, context) rows over {len(ALLF):,} features\n")
runs = []
for s in range(args.seeds):
    if stop: break
    t0 = time.time(); runs.append(run(s))
    print(f"  seed {s} done  {time.time()-t0:.0f}s")

def agg(strat, col):
    L = min(len(r[strat]) for r in runs)
    return np.array([[r[strat][i][col] for r in runs] for i in range(L)])

n = [runs[0]["B chooses"][i][0] for i in range(min(len(r["B chooses"]) for r in runs))]
print(f"\n{'':>10}{'B chooses':>26}{'random':>26}")
print(f"{'features':>10}{'MAE':>9}{'slope':>8}{'R2':>9}{'MAE':>9}{'slope':>8}{'R2':>9}")
bm, bs, br2 = agg("B chooses",1), agg("B chooses",2), agg("B chooses",4)
rm, rs, rr2 = agg("random",1), agg("random",2), agg("random",4)
for i, k in enumerate(n):
    print(f"{k:>10}{bm[i].mean():>9.4f}{bs[i].mean():>8.3f}{br2[i].mean():>9.3f}"
          f"{rm[i].mean():>9.4f}{rs[i].mean():>8.3f}{rr2[i].mean():>9.3f}")

# ---- scientific efficiency, measured at several accuracy levels ----
print(f"\n  METRIC 1 - causal calibration (held-out contexts, unseen features)")
print(f"    B's MAE {bm[-1].mean():.4f}   slope {bs[-1].mean():.3f}   R2 {br2[-1].mean():.3f}")
print(f"    (slope 1.000 = perfectly calibrated; the old scalar model had R2 = 0.327)")

def budget_for(curve_mae, target):
    """smallest budget at which this strategy reaches `target` MAE"""
    for i in range(len(n)):
        if curve_mae[i].mean() <= target: return n[i]
    return None

print(f"\n  METRIC 2 - scientific efficiency")
print(f"    {'accuracy (MAE)':>16}{'B needs':>10}{'random needs':>14}{'ratio':>9}")
ratios = []
for i in range(1, len(n)):
    tgt = bm[i].mean()
    b_n, r_n = n[i], budget_for(rm, tgt)
    if r_n is None:
        print(f"    {tgt:>16.4f}{b_n:>10}{'never':>14}{'>' + str(round(n[-1]/b_n,2)) + 'x':>9}")
    else:
        ratios.append(r_n / b_n)
        print(f"    {tgt:>16.4f}{b_n:>10}{r_n:>14}{r_n/b_n:>8.2f}x")
if ratios:
    print(f"\n    median efficiency ratio: {np.median(ratios):.2f}x")
    print(f"    -> B reaches a given accuracy with ~{np.median(ratios):.2f}x fewer experiments than random")
else:
    print(f"\n    random never matched B at any level tested")

json.dump({"n": n, "ratios": ratios, "B_mae": bm.mean(1).tolist(), "rand_mae": rm.mean(1).tolist(),
           "B_slope": bs.mean(1).tolist(), "B_r2": br2.mean(1).tolist(),
           "rand_r2": rr2.mean(1).tolist(), "seeds": args.seeds, "budget": args.budget},
          open(f"{ROOT}/findings/v2_loop.json","w"), indent=1)
print(f"\n  -> findings/v2_loop.json")
