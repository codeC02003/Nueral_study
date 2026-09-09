#!/usr/bin/env python3
"""
V3 - one clean experiment.

    python3 v3_discovery.py

A is FROZEN. Metric 1 is unchanged. Metric 2 is redefined.

WHY METRIC 2 IS REDEFINED
The old Metric 2 asked: does B reach lower average prediction error than random,
per experiment? Random won on every scoring - and for a principled reason. The test
set is a UNIFORM sample of held-out contexts, so random sampling is the unbiased
estimator of exactly the distribution being scored. Active learning cannot beat that
by construction.

But average-case predictive accuracy is not what a scientist optimises. A scientist
finds the important mechanisms. So:

    NEW METRIC 2 - important-mechanism discovery
    Of the total causal effect mass in A, how much has B FOUND after N experiments?

Scored on the experiments B actually ran, not on a held-out set. Three ways:

    mass found      sum |delta-loss| over acquired  /  sum over the whole pool
    vs oracle       mass found  /  the best any selector could have done at N
    top-1% recall   of the truly most-damaging contexts, how many has B run?

The oracle normalisation matters: it turns "0.34 of the mass" into "62% of what was
achievable", which is the number that means something.

FOUR SELECTORS
    random      uniform. The control.
    uncert      max ensemble disagreement. Pure exploration.
    magnitude   max predicted |delta-loss|. Pure exploitation.
    hybrid      |predicted| + kappa * uncertainty. Upper-confidence-bound.

Metric 1 is still reported every round on the 200 never-acquired test features, so
it is visible whether chasing important mechanisms wrecks calibration.
"""
import os, sys, json, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=2400)
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--kappa", type=float, default=1.0, help="hybrid: |pred| + kappa*sigma")
ap.add_argument("--target", default="raw", choices=["raw","log"],
                help="log: train on sign(y)*log1p(|y|) so the tail cannot dominate the loss")
ap.add_argument("--pool-cap", type=int, default=80000)
ap.add_argument("--acq", default="random,magnitude,mag90,mag80,mag50",
                help="magNN = NN%% of each round by predicted magnitude, rest random")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

# ------------------------------------------------------------------ inputs
z = np.load(f"{ROOT}/experiments/contextual.npz")
RES = np.load(f"{ROOT}/experiments/row_resid.npy")
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
ABS = np.abs(y)
ALLF = np.unique(feat)
print(f"  A frozen @ 83,750   pool {len(y):,} contexts   budget {args.budget} "
      f"({args.budget/len(y)*100:.2f}%)   {args.seeds} seeds   {DEV}\n")

class Head(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d,128), nn.GELU(), nn.Dropout(.1),
                            nn.Linear(128,64), nn.GELU(), nn.Linear(64,1))
    def forward(s, x): return s.n(x).squeeze(-1)

def T(v):    return torch.sign(v)*torch.log1p(v.abs())
def T_inv(v): return torch.sign(v)*(torch.expm1(v.abs()))

class Ensemble:
    """`--target log` is the fix for the mass-domination problem.

    A 20% random anchor is only 0.57% of the training MASS, because the
    magnitude-selected rows average 44x larger |delta-loss|. Mixing by count
    cannot rebalance a loss weighted by magnitude. Training on
    sign(y)*log1p(|y|) compresses the tail so every row contributes comparably,
    while the ACQUISITION rule is left untouched - so discovery should survive.
    """
    def __init__(s, d, n, seed):
        s.h = [Head(d).to(DEV) for _ in range(n)]
        for k, h in enumerate(s.h):
            torch.manual_seed(seed*1000+k)
            for m in h.modules():
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def fit(s, X, Y, epochs):
        s.log = (args.target == "log")
        Yt = T(Y) if s.log else Y
        s.mu, s.sd = X.mean(0), X.std(0).clamp(min=1e-6)
        s.ym, s.ys = Yt.mean(), Yt.std().clamp(min=1e-9)
        Xn, Yn = (X-s.mu)/s.sd, (Yt-s.ym)/s.ys
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
            m = p.mean(0)*s.ys+s.ym; sd = p.std(0)*s.ys
            if s.log:
                # invert to raw units, so every metric stays comparable
                lo, hi = T_inv(m-sd), T_inv(m+sd)
                m, sd = T_inv(m), (hi-lo).abs()/2
            ms.append(m); us.append(sd)
        return torch.cat(ms), torch.cat(us)

def run(seed):
    rng = np.random.RandomState(seed)
    fs = ALLF.copy(); rng.shuffle(fs)
    ist = np.isin(feat, fs[:200])
    te = np.nonzero(ist)[0]; pool = np.nonzero(~ist)[0]
    Xte = torch.tensor(X_ALL[te], device=DEV); Yte = torch.tensor(y[te], device=DEV)

    # ---- the discovery ground truth, over the acquirable pool ----
    pool_abs = ABS[pool]
    TOTAL = pool_abs.sum()
    order = np.argsort(-pool_abs)
    ORACLE = np.cumsum(pool_abs[order])                 # best possible mass at any N
    k1 = max(1, int(.01*len(pool)))
    TRUE_TOP = set(pool[order[:k1]].tolist())

    def m1(ens):
        p, _ = ens.pred(Xte)
        pc, ac = p-p.mean(), Yte-Yte.mean()
        slope = (pc@ac/pc.pow(2).sum().clamp(min=1e-12)).item()
        r2 = 1-(Yte-p).pow(2).sum().item()/max((Yte-Yte.mean()).pow(2).sum().item(),1e-12)
        return slope, r2

    out = {}
    seed_idx = rng.choice(pool, args.step, replace=False)
    for strat in args.acq.split(","):
        acq = seed_idx.copy(); curve = []
        while True:
            Xtr = torch.tensor(X_ALL[acq], device=DEV); Ytr = torch.tensor(y[acq], device=DEV)
            ens = Ensemble(Xtr.shape[1], args.heads, seed); ens.fit(Xtr, Ytr, args.epochs)
            found = ABS[acq].sum()
            mass = found/TOTAL
            orc  = found/ORACLE[len(acq)-1]
            rec  = len(TRUE_TOP & set(acq.tolist()))/k1
            sl, r2 = m1(ens)
            curve.append((len(acq), mass, orc, rec, sl, r2))
            print(f"    seed {seed} {strat:<9} {len(acq):>5}  mass {mass*100:6.2f}%  "
                  f"vs-oracle {orc*100:5.1f}%  top1%rec {rec*100:5.1f}%  slope {sl:.3f}  R2 {r2:+.3f}")
            if len(acq) >= args.budget or stop: break
            rest = np.setdiff1d(pool, acq)
            if strat == "random":
                pick = rng.choice(rest, min(args.step, len(rest)), replace=False)
            else:
                cand = rng.choice(rest, min(args.pool_cap, len(rest)), replace=False)
                pm, u = ens.pred(torch.tensor(X_ALL[cand], device=DEV))
                pm, u = pm.abs().cpu().numpy(), u.cpu().numpy()
                if strat.startswith("mag") and strat != "magnitude":
                    # SPLIT ACQUISITION. The selector chases important mechanisms,
                    # but a random slice is kept every round as a calibration
                    # anchor, so the training distribution cannot drift entirely
                    # into the tail. Acquisition set != training set is the point.
                    frac = int(strat[3:]) / 100.0
                    nm = int(round(args.step * frac))
                    a = cand[np.argsort(-pm)[:nm]]
                    b = rng.choice(np.setdiff1d(rest, a), args.step - nm, replace=False)
                    pick = np.concatenate([a, b])
                else:
                    sc = {"uncert": u, "magnitude": pm, "hybrid": pm + args.kappa*u}[strat]
                    pick = cand[np.argsort(-sc)[:args.step]]
            acq = np.concatenate([acq, pick])
        out[strat] = curve
    return out

runs = []
for s in range(args.seeds):
    if stop: break
    t0 = time.time(); runs.append(run(s)); print(f"  seed {s} done  {time.time()-t0:.0f}s")

STR = args.acq.split(",")
Lm = min(min(len(r[k]) for k in r) for r in runs)
n = [runs[0][STR[0]][i][0] for i in range(Lm)]
def agg(k,c): return np.array([[r[k][i][c] for r in runs] for i in range(Lm)])
R = {k: [agg(k,c) for c in range(1,6)] for k in STR}
NM = ["mass found","vs oracle","top-1% recall","calib slope","R2"]

for j,nm in enumerate(NM):
    print(f"\n  {nm}" + ("  (Metric 1 - should not collapse)" if j>=3 else "  (Metric 2)"))
    print(f"    {'experiments':>12}" + "".join(f"{k:>12}" for k in STR))
    for i,c in enumerate(n):
        f = (lambda v: f"{v*100:>11.2f}%") if j<3 else (lambda v: f"{v:>12.3f}")
        print(f"    {c:>12}" + "".join(f(R[k][j][i].mean()) for k in STR))

print(f"\n  METRIC 2 - important-mechanism discovery, vs random")
print(f"    {'strategy':<11}{'mass @2400':>12}{'x random':>10}{'oracle%':>10}{'top1% rec':>11}{'x random':>10}")
rm, rr = R["random"][0][-1].mean(), R["random"][2][-1].mean()
for k in STR:
    m, o, t = R[k][0][-1].mean(), R[k][1][-1].mean(), R[k][2][-1].mean()
    print(f"    {k:<11}{m*100:>11.2f}%{m/rm:>9.2f}x{o*100:>9.1f}%{t*100:>10.1f}%{t/max(rr,1e-9):>9.2f}x")

print(f"\n  budget random needs to match each strategy's mass at 2,400:")
for k in STR:
    if k == "random": continue
    tgt = R[k][0][-1].mean()
    j = next((q for q in range(len(n)) if R["random"][0][q].mean() >= tgt), None)
    print(f"    {k:<11}" + (f"random never reaches it within {n[-1]:,} "
          f"(random tops out at {R['random'][0][-1].mean()*100:.2f}%)" if j is None
          else f"random needs {n[j]:,} vs {n[-1]:,}  ->  {n[j]/n[-1]:.2f}x"))

json.dump({"n": n, "strategies": STR, "metrics": NM,
           **{f"{k}_{NM[j]}": R[k][j].mean(1).tolist() for k in STR for j in range(5)}},
          open(f"{ROOT}/findings/v3_discovery.json","w"), indent=1)
print(f"\n  -> findings/v3_discovery.json")
