#!/usr/bin/env python3
"""
V2 STEP 4, CORRECTED - B selects (feature, context) experiments, not features.

    python3 v2_context_loop.py                    # the experiment
    python3 v2_context_loop.py --budget 3000

WHAT WAS WRONG WITH v2_loop.py

The prediction target was (feature, context) -> delta-loss, but the SELECTION pool
was 821 features. At a budget of 600, random had already sampled 73% of everything
available, so both strategies converged on nearly the same data and no selection
rule could possibly help. The test saturated before it measured anything.

    old pool:      821 features        budget 600 = 73% of the pool
    correct pool:  452,219 contexts    budget 600 = 0.13% of the pool

The selection space now matches the prediction space. That is the only change.

HOLD-OUT IS STILL BY FEATURE. Test rows belong to features never queried under any
strategy, so nothing can be answered by memorising a feature's average - and both
strategies are scored on the identical frozen test set.
"""
import os, sys, json, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=2400, help="experiments (contexts) per strategy")
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--pool-cap", type=int, default=60000, help="candidates scored per round")
ap.add_argument("--acq", default="uncert,mixed,topk,random",
                help="acquisition strategies to compare")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

# ------------------------------------------------------------------ data
z = np.load(f"{ROOT}/experiments/contextual.npz")
meta = json.load(open(f"{ROOT}/data/meta.json")); V = len(meta["vocab"])
from nn.gpt import GPT, Config
from nn.sae import SAE
cfg = Config(vocab_size=V)
B_sae = SAE(cfg.n_embd); B_sae.load_state_dict(torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location="cpu")["model"])
A_gpt = GPT(cfg); A_gpt.load_state_dict(torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location="cpu")["model"])

feat = z["feature"].astype(np.int64); y = z["d_mean"].astype(np.float32)
with torch.no_grad():
    Wu, Wdec = A_gpt.head.weight, B_sae.W_dec
    logit = Wdec @ Wu.T
    FEAT_STATIC = torch.cat([
        Wdec, B_sae.W_enc.T, B_sae.b_enc.unsqueeze(1),
        Wdec.norm(dim=1, keepdim=True), logit.norm(dim=1, keepdim=True),
        logit.abs().max(dim=1, keepdim=True).values, logit.topk(4, dim=1).values,
    ], dim=1).numpy().astype(np.float32)
LOGIT = logit.numpy()

# Precompute the whole design matrix once - 452k x ~270 floats is ~490 MB in fp32,
# so build it in fp32 on CPU and move only the slices we need.
print("  building design matrix...")
CTX = np.stack([
    z["act"], np.log1p(z["act"]), z["base"], z["pos"]/cfg.block_size,
    LOGIT[feat, z["tgt"].astype(np.int64)], LOGIT[feat].max(1),
], axis=1).astype(np.float32)
X_ALL = np.concatenate([FEAT_STATIC[feat], CTX], axis=1)
print(f"  {X_ALL.shape[0]:,} candidate experiments x {X_ALL.shape[1]} inputs "
      f"({X_ALL.nbytes/1e6:.0f} MB)\n")

ALLF = np.unique(feat)

class Head(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d, 96), nn.GELU(), nn.Dropout(.1),
                            nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
    def forward(s, x): return s.n(x).squeeze(-1)

class Ensemble:
    def __init__(s, d, n, seed):
        s.h = []
        for k in range(n):
            torch.manual_seed(seed*1000 + k); s.h.append(Head(d).to(DEV))
    def fit(s, X, Y, epochs):
        s.mu, s.sd = X.mean(0), X.std(0).clamp(min=1e-6)
        s.ym, s.ys = Y.mean(), Y.std().clamp(min=1e-9)
        Xn, Yn = (X-s.mu)/s.sd, (Y-s.ym)/s.ys
        for k, h in enumerate(s.h):
            g = torch.Generator().manual_seed(k)
            bi = torch.randint(0, len(Xn), (len(Xn),), generator=g).to(DEV)
            o = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-2); h.train()
            for _ in range(epochs):
                o.zero_grad(set_to_none=True)
                F.smooth_l1_loss(h(Xn[bi]), Yn[bi], beta=.5).backward(); o.step()
    @torch.no_grad()
    def pred(s, X, chunk=100000):
        ms, us = [], []
        for i in range(0, len(X), chunk):
            Xn = (X[i:i+chunk]-s.mu)/s.sd
            p = torch.stack([h.eval()(Xn) for h in s.h])
            ms.append(p.mean(0)*s.ys + s.ym); us.append(p.std(0)*s.ys)
        return torch.cat(ms), torch.cat(us)

def run(seed):
    rng = np.random.RandomState(seed)
    fs = ALLF.copy(); rng.shuffle(fs)
    test_f = set(int(f) for f in fs[:200])
    is_test = np.isin(feat, list(test_f))
    te = np.nonzero(is_test)[0]
    pool = np.nonzero(~is_test)[0]                      # every context of every train feature
    Xte = torch.tensor(X_ALL[te], device=DEV); Yte = torch.tensor(y[te], device=DEV)

    def metrics(ens):
        p, _ = ens.pred(Xte)
        mae = (p-Yte).abs().mean().item()
        pc, ac = p-p.mean(), Yte-Yte.mean()
        slope = (pc@ac / pc.pow(2).sum().clamp(min=1e-12)).item()
        r2 = 1 - (Yte-p).pow(2).sum().item()/max((Yte-Yte.mean()).pow(2).sum().item(),1e-12)
        return mae, slope, r2

    out = {}
    seed_idx = rng.choice(pool, args.step, replace=False)   # identical start for all
    for strat in args.acq.split(","):
        acq = seed_idx.copy(); curve = []
        while len(acq) <= args.budget and not stop:
            Xtr = torch.tensor(X_ALL[acq], device=DEV); Ytr = torch.tensor(y[acq], device=DEV)
            ens = Ensemble(Xtr.shape[1], args.heads, seed); ens.fit(Xtr, Ytr, args.epochs)
            m = metrics(ens); curve.append((len(acq),) + m)
            print(f"    seed {seed} {strat:<10} {len(acq):>5} contexts  MAE {m[0]:.4f}  slope {m[1]:.3f}  R2 {m[2]:+.3f}")
            if len(acq) >= args.budget: break
            rest = np.setdiff1d(pool, acq, assume_unique=False)
            if strat == "random":
                pick = rng.choice(rest, min(args.step, len(rest)), replace=False)
            else:
                # score a large random slice - 452k candidates every round is wasteful
                cand = rng.choice(rest, min(args.pool_cap, len(rest)), replace=False)
                _, u = ens.pred(torch.tensor(X_ALL[cand], device=DEV))
                order = torch.argsort(u, descending=True).cpu().numpy()
                if strat == "uncert":
                    # PURE argmax uncertainty. Known to fail on heavy-tailed targets:
                    # the most uncertain points are the most extreme ones, so the
                    # training set drifts away from the test distribution.
                    pick = cand[order[:args.step]]
                elif strat == "mixed":
                    # half by uncertainty, half at random. The random half anchors
                    # the training distribution so it cannot drift.
                    h = args.step // 2
                    a = cand[order[:h]]
                    b = rng.choice(np.setdiff1d(rest, a), args.step - h, replace=False)
                    pick = np.concatenate([a, b])
                elif strat == "topk":
                    # diversity WITHIN the uncertain region: sample from the top 10x
                    # most uncertain rather than taking the single most extreme ones.
                    top = cand[order[:min(args.step*10, len(order))]]
                    pick = rng.choice(top, args.step, replace=False)
                else:
                    raise SystemExit(f"unknown strategy {strat}")
            acq = np.concatenate([acq, pick])
        out[strat] = curve
    return out

print(f"V2 corrected   budget {args.budget} contexts   pool {len(y):,}   "
      f"({args.budget/len(y)*100:.2f}% of the pool)   {args.seeds} seeds   {DEV}\n")
runs = []
for s in range(args.seeds):
    if stop: break
    t0 = time.time(); runs.append(run(s)); print(f"  seed {s} done  {time.time()-t0:.0f}s")

STR = args.acq.split(",")
L = min(min(len(r[k]) for k in r) for r in runs)
n = [runs[0][STR[0]][i][0] for i in range(L)]
def agg(k, c): return np.array([[r[k][i][c] for r in runs] for i in range(L)])
res = {k: (agg(k,1), agg(k,2), agg(k,3)) for k in STR}

hdr = f"{'contexts':>9}" + "".join(f"{k:>26}" for k in STR)
print("\n" + hdr)
print(f"{'':>9}" + "".join(f"{'MAE':>9}{'slope':>8}{'R2':>9}" for _ in STR))
for i, c in enumerate(n):
    line = f"{c:>9}"
    for k in STR:
        m, sl, r2 = res[k]
        line += f"{m[i].mean():>9.4f}{sl[i].mean():>8.3f}{r2[i].mean():>9.3f}"
    print(line)

print(f"\n  METRIC 1 - causal calibration (best point per strategy)")
print(f"    {'strategy':<10}{'MAE':>9}{'slope':>9}{'R2':>9}{'at':>8}{'trend':>10}")
for k in STR:
    m, sl, r2 = res[k]
    b = int(np.argmin(m.mean(1)))
    trend = "improving" if m[-1].mean() <= m[b].mean()*1.02 else "DEGRADING"
    print(f"    {k:<10}{m[b].mean():>9.4f}{sl[b].mean():>9.3f}{r2[b].mean():>9.3f}{n[b]:>8}{trend:>10}")

rm = res["random"][0] if "random" in res else None
if rm is not None:
    print(f"\n  METRIC 2 - scientific efficiency vs random")
    for k in STR:
        if k == "random": continue
        m = res[k][0]
        ratios = []
        for i in range(1, len(n)):
            t = m[i].mean()
            j = next((q for q in range(len(n)) if rm[q].mean() <= t), None)
            if j is not None: ratios.append(n[j]/n[i])
        if ratios:
            r = np.array(ratios)
            print(f"    {k:<10} median {np.median(r):>5.2f}x   mean {r.mean():>5.2f}x   "
                  f"({(r>1).sum()}/{len(r)} levels favour it)")
        else:
            print(f"    {k:<10} random never matched it")

json.dump({"n": n, "strategies": STR,
           **{f"{k}_{f}": res[k][j].mean(1).tolist()
              for k in STR for j, f in enumerate(["mae","slope","r2"])},
           "pool": int(len(y)), "budget": args.budget},
          open(f"{ROOT}/findings/v2_acquisition.json","w"), indent=1)
print(f"\n  -> findings/v2_acquisition.json")
