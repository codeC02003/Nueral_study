#!/usr/bin/env python3
"""
V2 METRIC 2, RE-SCORED - is B's experiment selection actually worse than random,
or was the previous answer an artefact of the metric?

    python3 v2_metric2.py

WHY RE-SCORE

The first attempt said random wins. But the diagnosis showed the target is
brutally heavy-tailed:

    rows above the 99.9th percentile: 0.10% of rows, 23.7% of all variance

R2 and MAE are variance-weighted, so a few hundred freak rows dominate the score.
Uncertainty sampling CHASES that tail; R2 is SCORED on that tail. Both failed for
the same reason, which means the comparison never tested what it claimed to.

So this run reports metrics that are not tail-dominated:

    Spearman rho   rank correlation - immune to the magnitude of the extremes
    slope          calibration; 1.0 = unbiased
    log-MAE        error on sign(y)*log1p(|y|) - the tail cannot swamp it
    top-1% recall  of the truly most-damaging contexts, how many does B find?
    R2             kept, so the artefact is visible rather than hidden

AND a stratified acquisition rule. Global argmax-uncertainty spends the whole
budget on freaks. Stratified splits candidates into deciles by PREDICTED magnitude
(observable before running the experiment) and takes the most uncertain within
each decile, so the budget covers the whole range of effect sizes.
"""
import os, sys, json, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=2400)
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--pool-cap", type=int, default=60000)
ap.add_argument("--acq", default="strat,mixed,random")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

z = np.load(f"{ROOT}/experiments/contextual.npz")
RES = np.load(f"{ROOT}/experiments/row_resid.npy")      # the richer context that fixed calibration
meta = json.load(open(f"{ROOT}/data/meta.json"))
from nn.gpt import GPT, Config
from nn.sae import SAE
cfg = Config(vocab_size=len(meta["vocab"]))
Bs = SAE(cfg.n_embd); Bs.load_state_dict(torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location="cpu")["model"])
Ag = GPT(cfg); Ag.load_state_dict(torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location="cpu")["model"])
feat = z["feature"].astype(np.int64); y = z["d_mean"].astype(np.float32)
with torch.no_grad():
    Wdec = Bs.W_dec; logit = Wdec @ Ag.head.weight.T
    STAT = torch.cat([Wdec, Bs.W_enc.T, Bs.b_enc.unsqueeze(1), Wdec.norm(dim=1, keepdim=True),
        logit.norm(dim=1, keepdim=True), logit.abs().max(1, keepdim=True).values,
        logit.topk(4, 1).values], 1).numpy().astype(np.float32)
L = logit.numpy()
CTX = np.stack([z["act"], np.log1p(z["act"]), z["base"], z["pos"]/cfg.block_size,
    L[feat, z["tgt"].astype(np.int64)], L[feat].max(1)], 1).astype(np.float32)
X_ALL = np.concatenate([STAT[feat], CTX, RES], 1)
print(f"  {X_ALL.shape[0]:,} candidate experiments x {X_ALL.shape[1]} inputs")
print(f"  budget {args.budget} = {args.budget/len(y)*100:.2f}% of the pool\n")
ALLF = np.unique(feat)

class Head(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d,128), nn.GELU(), nn.Dropout(.1),
                            nn.Linear(128,64), nn.GELU(), nn.Linear(64,1))
    def forward(s, x): return s.n(x).squeeze(-1)

class Ensemble:
    def __init__(s, d, n, seed):
        s.h = []
        for k in range(n):
            torch.manual_seed(seed*1000+k); s.h.append(Head(d).to(DEV))
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
            ms.append(p.mean(0)*s.ys+s.ym); us.append(p.std(0)*s.ys)
        return torch.cat(ms), torch.cat(us)

def rankdata(t):
    o = torch.argsort(t); r = torch.empty_like(o, dtype=torch.float32)
    r[o] = torch.arange(len(t), dtype=torch.float32, device=t.device)
    return r

def run(seed):
    rng = np.random.RandomState(seed)
    fs = ALLF.copy(); rng.shuffle(fs)
    ist = np.isin(feat, fs[:200])
    te = np.nonzero(ist)[0]; pool = np.nonzero(~ist)[0]
    Xte = torch.tensor(X_ALL[te], device=DEV); Yte = torch.tensor(y[te], device=DEV)
    lg_true = torch.sign(Yte)*torch.log1p(Yte.abs())
    k1 = max(1, int(.01*len(Yte)))
    true_top = set(Yte.abs().topk(k1).indices.tolist())
    ry = rankdata(Yte)

    def metrics(ens):
        p, _ = ens.pred(Xte)
        pc, ac = p-p.mean(), Yte-Yte.mean()
        slope = (pc@ac/pc.pow(2).sum().clamp(min=1e-12)).item()
        r2 = 1-(Yte-p).pow(2).sum().item()/max((Yte-Yte.mean()).pow(2).sum().item(),1e-12)
        rp = rankdata(p)
        rho = torch.corrcoef(torch.stack([rp, ry]))[0,1].item()
        lg_p = torch.sign(p)*torch.log1p(p.abs())
        lmae = (lg_p-lg_true).abs().mean().item()
        pred_top = set(p.abs().topk(k1).indices.tolist())
        rec = len(true_top & pred_top)/k1
        return rho, slope, lmae, rec, r2

    out = {}
    seed_idx = rng.choice(pool, args.step, replace=False)
    for strat in args.acq.split(","):
        acq = seed_idx.copy(); curve = []
        while len(acq) <= args.budget and not stop:
            Xtr = torch.tensor(X_ALL[acq], device=DEV); Ytr = torch.tensor(y[acq], device=DEV)
            ens = Ensemble(Xtr.shape[1], args.heads, seed); ens.fit(Xtr, Ytr, args.epochs)
            m = metrics(ens); curve.append((len(acq),)+m)
            print(f"    seed {seed} {strat:<7} {len(acq):>5}  rho {m[0]:.3f}  slope {m[1]:.3f}"
                  f"  logMAE {m[2]:.4f}  top1%rec {m[3]:.3f}  R2 {m[4]:+.3f}")
            if len(acq) >= args.budget: break
            rest = np.setdiff1d(pool, acq)
            if strat == "random":
                pick = rng.choice(rest, min(args.step, len(rest)), replace=False)
            else:
                cand = rng.choice(rest, min(args.pool_cap, len(rest)), replace=False)
                pm, u = ens.pred(torch.tensor(X_ALL[cand], device=DEV))
                u = u.cpu().numpy(); pm = pm.cpu().numpy()
                if strat == "mixed":
                    h = args.step//2
                    a = cand[np.argsort(-u)[:h]]
                    pick = np.concatenate([a, rng.choice(np.setdiff1d(rest,a), args.step-h, replace=False)])
                elif strat == "strat":
                    # STRATIFIED: deciles of PREDICTED magnitude (observable before
                    # the experiment), then most-uncertain within each decile. The
                    # budget therefore covers the whole range of effect sizes
                    # instead of collapsing onto the freak tail.
                    q = np.quantile(np.abs(pm), np.linspace(0,1,11))
                    per = args.step//10; sel = []
                    for d in range(10):
                        lo, hi = q[d], q[d+1]
                        m_ = np.nonzero((np.abs(pm)>=lo)&(np.abs(pm)<=hi))[0]
                        if len(m_)==0: continue
                        sel.append(cand[m_[np.argsort(-u[m_])[:per]]])
                    pick = np.concatenate(sel) if sel else cand[np.argsort(-u)[:args.step]]
                    if len(pick) < args.step:
                        pick = np.concatenate([pick, rng.choice(np.setdiff1d(rest,pick),
                                                                args.step-len(pick), replace=False)])
                else: raise SystemExit(strat)
            acq = np.concatenate([acq, pick])
        out[strat] = curve
    return out

runs = []
for s in range(args.seeds):
    if stop: break
    t0=time.time(); runs.append(run(s)); print(f"  seed {s} done  {time.time()-t0:.0f}s")

STR = args.acq.split(",")
Lm = min(min(len(r[k]) for k in r) for r in runs)
n = [runs[0][STR[0]][i][0] for i in range(Lm)]
def agg(k,c): return np.array([[r[k][i][c] for r in runs] for i in range(Lm)])
RES_ = {k: [agg(k,c) for c in range(1,6)] for k in STR}
NAMES = ["Spearman rho","calib slope","log-MAE","top1% recall","R2"]
BETTER = [1,None,-1,1,1]      # 1 = higher better, -1 = lower better

for j,nm in enumerate(NAMES):
    print(f"\n  {nm}")
    print(f"    {'contexts':>9}" + "".join(f"{k:>12}" for k in STR))
    for i,c in enumerate(n):
        print(f"    {c:>9}" + "".join(f"{RES_[k][j][i].mean():>12.4f}" for k in STR))

print(f"\n  METRIC 2 - efficiency vs random, on TAIL-ROBUST metrics")
print(f"    {'metric':<14}{'strategy':<9}{'median ratio':>13}{'levels favouring it':>22}")
for j,nm in enumerate(NAMES):
    if BETTER[j] is None: continue
    rr = RES_["random"][j]
    for k in STR:
        if k=="random": continue
        m = RES_[k][j]; ratios=[]
        for i in range(1,len(n)):
            t = m[i].mean()
            hit = (lambda q: rr[q].mean() >= t) if BETTER[j]==1 else (lambda q: rr[q].mean() <= t)
            q = next((w for w in range(len(n)) if hit(w)), None)
            if q is not None: ratios.append(n[q]/n[i])
        if ratios:
            r=np.array(ratios)
            print(f"    {nm:<14}{k:<9}{np.median(r):>12.2f}x{f'{(r>1).sum()}/{len(r)}':>22}")
        else:
            print(f"    {nm:<14}{k:<9}{'never matched':>13}{'':>22}")
json.dump({"n":n,"strategies":STR,"metrics":NAMES,
           **{f"{k}_{NAMES[j]}":RES_[k][j].mean(1).tolist() for k in STR for j in range(5)}},
          open(f"{ROOT}/findings/v2_metric2.json","w"),indent=1)
print(f"\n  -> findings/v2_metric2.json")
