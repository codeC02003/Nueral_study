#!/usr/bin/env python3
"""
Dump the row-level arrays a cluster bootstrap needs.

    python3 bootstrap_prep.py --model a  --layer 2
    python3 bootstrap_prep.py --model a2 --layer 2

The four-layer runs reported only scalars, so there is nothing to resample. This
re-runs ONE acquisition trajectory per (model, layer) using the SAE that already
exists - so it skips the 7-minute SAE training - and saves:

    feature[i]           which feature each row belongs to   (the CLUSTER id)
    absy[i]              |delta-loss|, the discovery target
    pool / acq / test    row index sets
    pred / actual        per-test-row prediction and truth   (for slope, R2)

Everything else about the configuration is unchanged: mag80 acquisition,
log-magnitude target, same budget, same 5-head ensemble, same feature-level split.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["a","a2"])
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--batches", type=int, default=10)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--budget", type=int, default=2400)
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seeds", type=int, default=3, help="acquisition trajectories to save")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
L, M = args.layer, args.model

from nn.gpt import GPT, Config
from nn.sae import SAE
meta = json.load(open(f"{ROOT}/data/meta.json"))
cfg = Config(vocab_size=len(meta["vocab"]))
data = np.load(f"{ROOT}/data/corpus.npy")
A = GPT(cfg).to(DEV).eval()
A.load_state_dict(torch.load(f"{ROOT}/checkpoints/{M}_latest.pt", map_location=DEV)["model"])
for q in A.parameters(): q.requires_grad_(False)
sp = f"{ROOT}/checkpoints/b_L{L}.pt" if M == "a" else f"{ROOT}/checkpoints/b_{M}_L{L}.pt"
B = SAE(cfg.n_embd).to(DEV); B.load_state_dict(torch.load(sp, map_location=DEV)["model"])
for q in B.parameters(): q.requires_grad_(False)
print(f"  {M.upper()} layer {L}   SAE from {os.path.basename(sp)}")

def mk(s):
    rs = np.random.RandomState(s)
    ix = rs.randint(0, len(data)-cfg.block_size-1, args.batch)
    X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    return X, Y
@torch.no_grad()
def ptl(X, Y, iv=None):
    lg,_,_ = A(X, intervene=iv)
    return F.cross_entropy(lg.view(-1, lg.size(-1)), Y.reshape(-1), reduction="none")

BAT = [mk(700+k) for k in range(args.batches)]
BASE, ACT, RESID, TOK, TGT, POS = [], [], [], [], [], []
with torch.no_grad():
    for X, Y in BAT:
        BASE.append(ptl(X, Y)); _,_,ac = A(X, record=True)
        r = ac["resid"][L].reshape(-1, cfg.n_embd)
        RESID.append(r); ACT.append(B.encode(r/B.scale))
        TOK.append(X.reshape(-1)); TGT.append(Y.reshape(-1))
        POS.append(torch.arange(cfg.block_size, device=DEV).repeat(args.batch))
MU = torch.cat(ACT).mean(0)

cols = {k: [] for k in ["feature","act","tgt","base","d","pos"]}
RR = []
for i in range(B.d_hidden):
    def hook(h, i=i):
        f = B.encode(h/B.scale); return h - ((f[...,i:i+1]-MU[i])*B.W_dec[i])*B.scale
    for k,(X,Y) in enumerate(BAT):
        sel = torch.nonzero(ACT[k][:, i] > 1e-3).flatten()
        if sel.numel()==0: continue
        d = (ptl(X,Y,{L:hook}) - BASE[k])[sel]
        cols["feature"].append(torch.full_like(sel,i)); cols["act"].append(ACT[k][sel,i])
        cols["tgt"].append(TGT[k][sel]); cols["base"].append(BASE[k][sel])
        cols["pos"].append(POS[k][sel]); cols["d"].append(d); RR.append(RESID[k][sel])
C = {k: torch.cat(v).float().cpu().numpy() for k,v in cols.items()}
RESM = torch.cat(RR).float().cpu().numpy()
feat = C["feature"].astype(np.int64); y = C["d"].astype(np.float32)
print(f"  {len(y):,} rows")

with torch.no_grad():
    Wd = B.W_dec.cpu(); lg = Wd @ A.head.weight.cpu().T
    STAT = torch.cat([Wd, B.W_enc.T.cpu(), B.b_enc.unsqueeze(1).cpu(), Wd.norm(dim=1,keepdim=True),
        lg.norm(dim=1,keepdim=True), lg.abs().max(1,keepdim=True).values, lg.topk(4,1).values],1).numpy().astype(np.float32)
LG = lg.numpy()
CTX = np.stack([C["act"], np.log1p(C["act"]), C["base"], C["pos"]/cfg.block_size,
    LG[feat, C["tgt"].astype(np.int64)], LG[feat].max(1)],1).astype(np.float32)
X_ALL = np.concatenate([STAT[feat], CTX, RESM],1)

def T(v): return torch.sign(v)*torch.log1p(v.abs())
def Ti(v): return torch.sign(v)*torch.expm1(v.abs())
class H(nn.Module):
    def __init__(s,d):
        super().__init__(); s.n=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.1),
            nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x): return s.n(x).squeeze(-1)
class Ens:
    def __init__(s,d,n,seed):
        s.h=[]
        for k in range(n): torch.manual_seed(seed*1000+k); s.h.append(H(d).to(DEV))
    def fit(s,X,Y,ep):
        Yt=T(Y); s.mu,s.sd=X.mean(0),X.std(0).clamp(min=1e-6)
        s.ym,s.ys=Yt.mean(),Yt.std().clamp(min=1e-9); Xn,Yn=(X-s.mu)/s.sd,(Yt-s.ym)/s.ys
        for k,h in enumerate(s.h):
            g=torch.Generator().manual_seed(k); bi=torch.randint(0,len(Xn),(len(Xn),),generator=g).to(DEV)
            o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-2); h.train()
            for _ in range(ep):
                o.zero_grad(set_to_none=True); F.smooth_l1_loss(h(Xn[bi]),Yn[bi],beta=.5).backward(); o.step()
    @torch.no_grad()
    def pred(s,X,chunk=100000):
        ms=[]
        for i in range(0,len(X),chunk):
            p=torch.stack([h.eval()((X[i:i+chunk]-s.mu)/s.sd) for h in s.h])
            ms.append(Ti(p.mean(0)*s.ys+s.ym))
        return torch.cat(ms)

# Save EVERY acquisition trajectory, not just one. The reported figures are
# 3-seed means, so a CI conditional on a single trajectory is an interval on a
# different estimator. The harvest above is shared across seeds, so this costs
# three short trajectories rather than three full runs.
save = {}
ALLF = np.unique(feat)
for sd in range(args.seeds):
    rng = np.random.RandomState(sd)
    fs = ALLF.copy(); rng.shuffle(fs)
    nte = max(1, int(.2*len(fs))); ist = np.isin(feat, fs[:nte])
    te = np.nonzero(ist)[0]; pool = np.nonzero(~ist)[0]
    acq = rng.choice(pool, args.step, replace=False)
    while len(acq) < args.budget:
        Xtr=torch.tensor(X_ALL[acq],device=DEV); Ytr=torch.tensor(y[acq],device=DEV)
        e=Ens(Xtr.shape[1],args.heads,sd); e.fit(Xtr,Ytr,args.epochs)
        rest=np.setdiff1d(pool,acq); cand=rng.choice(rest,min(80000,len(rest)),replace=False)
        pm=e.pred(torch.tensor(X_ALL[cand],device=DEV)).abs().cpu().numpy()
        nm=int(round(args.step*.8)); a1=cand[np.argsort(-pm)[:nm]]
        acq=np.concatenate([acq,a1,rng.choice(np.setdiff1d(rest,a1),args.step-nm,replace=False)])
    acq = acq[:args.budget]
    Xtr=torch.tensor(X_ALL[acq],device=DEV); Ytr=torch.tensor(y[acq],device=DEV)
    e=Ens(Xtr.shape[1],args.heads,sd); e.fit(Xtr,Ytr,args.epochs)
    pred = e.pred(torch.tensor(X_ALL[te],device=DEV)).cpu().numpy()
    rnd = rng.choice(pool, args.budget, replace=False)
    save.update({f"pool{sd}":pool.astype(np.int32), f"acq{sd}":acq.astype(np.int32),
                 f"rnd{sd}":rnd.astype(np.int32), f"test{sd}":te.astype(np.int32),
                 f"pred{sd}":pred.astype(np.float32), f"actual{sd}":y[te].astype(np.float32)})
    print(f"    trajectory seed {sd} saved")

od = f"{ROOT}/experiments/boot"; os.makedirs(od, exist_ok=True)
np.savez_compressed(f"{od}/{M}_L{L}.npz",
    feature=feat.astype(np.int32), absy=np.abs(y), y=y,
    n_seeds=np.array(args.seeds), **save)
print(f"  -> experiments/boot/{M}_L{L}.npz")
