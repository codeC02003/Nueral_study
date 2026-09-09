#!/usr/bin/env python3
"""
V3 VALIDATION - does the ~12x discovery result survive on a different layer of A?

    python3 validate_layer.py --layer 0
    python3 validate_layer.py --layer 0 --sae-steps 40000

Self-contained: trains an SAE on the chosen layer, harvests the (feature, context)
dataset in memory, and runs the frozen V3 configuration (mag80 acquisition +
log-magnitude target) against a random control. Nothing is layer-2 specific, and no
intermediate files are written, so every layer is measured identically.

WHY THIS MATTERS
The V3 headline - B finds A's important causal mass ~12x faster than random - was
measured on layer 2 only. If it is peculiar to layer 2, the claim is weak. Every
layer gets the SAME SAE step budget so the comparison is fair.
"""
import os, sys, json, time, argparse, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--model", default="a", help="checkpoint prefix: 'a' or 'a2'")
ap.add_argument("--sae-steps", type=int, default=40000)
ap.add_argument("--sae-l1", type=float, default=1e-2)
ap.add_argument("--batches", type=int, default=10)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--budget", type=int, default=2400)
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--heads", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--retrain-sae", action="store_true")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
L = args.layer
stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

from nn.gpt import GPT, Config
from nn.sae import SAE
meta = json.load(open(f"{ROOT}/data/meta.json"))
cfg = Config(vocab_size=len(meta["vocab"]))
data = np.load(f"{ROOT}/data/corpus.npy")
ack = torch.load(f"{ROOT}/checkpoints/{args.model}_latest.pt", map_location=DEV)
A = GPT(cfg).to(DEV).eval(); A.load_state_dict(ack["model"])
for q in A.parameters(): q.requires_grad_(False)
print(f"MODEL {args.model.upper()}  LAYER {L}   frozen @ {ack['step']:,}   {DEV}\n")

def mk(seed):
    rs = np.random.RandomState(seed)
    ix = rs.randint(0, len(data)-cfg.block_size-1, args.batch)
    X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    return X, Y

# ---------------------------------------------------------------- 1. the SAE
SP = f"{ROOT}/checkpoints/b_{args.model}_L{L}.pt"
B = SAE(cfg.n_embd, 8, args.sae_l1).to(DEV)
if os.path.exists(SP) and not args.retrain_sae:
    ck = torch.load(SP, map_location=DEV); B.load_state_dict(ck["model"])
    print(f"  loaded SAE for layer {L} @ step {ck['step']:,}")
else:
    print(f"  training SAE on layer {L} for {args.sae_steps:,} steps...")
    def acts_batch(bs):
        rs = np.random.RandomState()
        ix = rs.randint(0, len(data)-cfg.block_size-1, bs)
        X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
        with torch.no_grad(): _, _, ac = A(X, record=True)
        return ac["resid"][L].reshape(-1, cfg.n_embd)
    B.calibrate(acts_batch(16))
    opt = torch.optim.Adam(B.parameters(), lr=1e-3)
    fire = torch.zeros(B.d_hidden, device=DEV); t0 = time.time()
    for s in range(args.sae_steps):
        if stop: break
        a = acts_batch(16)
        _, f, loss, mse, _ = B(a)
        opt.zero_grad(set_to_none=True); loss.backward()
        B.W_dec.grad -= (B.W_dec.grad*B.W_dec.data).sum(-1, keepdim=True)*B.W_dec.data
        opt.step(); B.normalize_decoder(); fire += (f > 0).float().sum(0)
        if (s+1) % 10000 == 0:
            l0, fvu, _ = B.stats(a)
            print(f"    {s+1:>6,}  recon-err {fvu*100:5.1f}%  active/token {l0:5.1f}  "
                  f"dead {(fire==0).sum().item():>4}  {time.time()-t0:.0f}s")
    torch.save({"model": B.state_dict(), "step": args.sae_steps, "layer": L}, SP)
for q in B.parameters(): q.requires_grad_(False)
a = None
with torch.no_grad():
    X, _ = mk(700); _, _, ac = A(X, record=True)
    l0, fvu, _ = B.stats(ac["resid"][L].reshape(-1, cfg.n_embd))
print(f"  SAE ready: recon-err {fvu*100:.1f}%   active/token {l0:.1f}\n")

# ---------------------------------------------------------------- 2. harvest
print(f"  harvesting (feature, context) experiments on layer {L}...")
BAT = [mk(700+k) for k in range(args.batches)]
@torch.no_grad()
def ptl(X, Y, iv=None):
    lg, _, _ = A(X, intervene=iv)
    return F.cross_entropy(lg.view(-1, lg.size(-1)), Y.reshape(-1), reduction="none")
BASE, ACT, RESID, TOK, TGT, POS = [], [], [], [], [], []
with torch.no_grad():
    for X, Y in BAT:
        BASE.append(ptl(X, Y))
        _, _, ac = A(X, record=True)
        r = ac["resid"][L].reshape(-1, cfg.n_embd)
        RESID.append(r); ACT.append(B.encode(r/B.scale))
        TOK.append(X.reshape(-1)); TGT.append(Y.reshape(-1))
        POS.append(torch.arange(cfg.block_size, device=DEV).repeat(args.batch))
MU = torch.cat(ACT).mean(0)
NPOS = sum(b.numel() for b in BASE)

cols = {k: [] for k in ["feature","act","tok","tgt","base","d","pos"]}
RES_rows = []
t0 = time.time()
for i in range(B.d_hidden):
    if stop: break
    def hook(h, i=i):
        f = B.encode(h/B.scale)
        return h - ((f[..., i:i+1]-MU[i])*B.W_dec[i])*B.scale
    for k, (X, Y) in enumerate(BAT):
        sel = torch.nonzero(ACT[k][:, i] > 1e-3).flatten()
        if sel.numel() == 0: continue
        d = (ptl(X, Y, {L: hook}) - BASE[k])[sel]
        cols["feature"].append(torch.full_like(sel, i))
        cols["act"].append(ACT[k][sel, i]); cols["tok"].append(TOK[k][sel])
        cols["tgt"].append(TGT[k][sel]); cols["base"].append(BASE[k][sel])
        cols["pos"].append(POS[k][sel]); cols["d"].append(d)
        RES_rows.append(RESID[k][sel])
    if (i+1) % 256 == 0:
        print(f"    {i+1}/{B.d_hidden}  {time.time()-t0:.0f}s")
C = {k: torch.cat(v).float().cpu().numpy() for k, v in cols.items()}
RESM = torch.cat(RES_rows).float().cpu().numpy()
feat = C["feature"].astype(np.int64); y = C["d"].astype(np.float32); ABS = np.abs(y)
print(f"  {len(y):,} (feature, context) rows   {time.time()-t0:.0f}s\n")

# ---------------------------------------------------------------- 3. inputs
with torch.no_grad():
    Wdec = B.W_dec.cpu(); logit = Wdec @ A.head.weight.cpu().T
    STAT = torch.cat([Wdec, B.W_enc.T.cpu(), B.b_enc.unsqueeze(1).cpu(),
        Wdec.norm(dim=1,keepdim=True), logit.norm(dim=1,keepdim=True),
        logit.abs().max(1,keepdim=True).values, logit.topk(4,1).values],1).numpy().astype(np.float32)
LG = logit.numpy()
CTX = np.stack([C["act"], np.log1p(C["act"]), C["base"], C["pos"]/cfg.block_size,
    LG[feat, C["tgt"].astype(np.int64)], LG[feat].max(1)],1).astype(np.float32)
X_ALL = np.concatenate([STAT[feat], CTX, RESM], 1)
ALLF = np.unique(feat)
print(f"  design matrix {X_ALL.shape}   features {len(ALLF)}\n")

# ---------------------------------------------------------------- 4. the V3 run
def T(v): return torch.sign(v)*torch.log1p(v.abs())
def Ti(v): return torch.sign(v)*torch.expm1(v.abs())
class Head(nn.Module):
    def __init__(s,d):
        super().__init__(); s.n=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.1),
            nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x): return s.n(x).squeeze(-1)
class Ens:
    def __init__(s,d,n,seed):
        s.h=[]
        for k in range(n):
            torch.manual_seed(seed*1000+k); s.h.append(Head(d).to(DEV))
    def fit(s,X,Y,ep):
        Yt=T(Y); s.mu,s.sd=X.mean(0),X.std(0).clamp(min=1e-6)
        s.ym,s.ys=Yt.mean(),Yt.std().clamp(min=1e-9)
        Xn,Yn=(X-s.mu)/s.sd,(Yt-s.ym)/s.ys
        for k,h in enumerate(s.h):
            g=torch.Generator().manual_seed(k)
            bi=torch.randint(0,len(Xn),(len(Xn),),generator=g).to(DEV)
            o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-2); h.train()
            for _ in range(ep):
                o.zero_grad(set_to_none=True)
                F.smooth_l1_loss(h(Xn[bi]),Yn[bi],beta=.5).backward(); o.step()
    @torch.no_grad()
    def pred(s,X,chunk=100000):
        ms,us=[],[]
        for i in range(0,len(X),chunk):
            Xn=(X[i:i+chunk]-s.mu)/s.sd
            p=torch.stack([h.eval()(Xn) for h in s.h])
            m=p.mean(0)*s.ys+s.ym; sd=p.std(0)*s.ys
            lo,hi=Ti(m-sd),Ti(m+sd); ms.append(Ti(m)); us.append((hi-lo).abs()/2)
        return torch.cat(ms),torch.cat(us)

def run(seed):
    rng=np.random.RandomState(seed); fs=ALLF.copy(); rng.shuffle(fs)
    nte=max(1,int(.2*len(fs)))
    ist=np.isin(feat,fs[:nte]); te=np.nonzero(ist)[0]; pool=np.nonzero(~ist)[0]
    Xte=torch.tensor(X_ALL[te],device=DEV); Yte=torch.tensor(y[te],device=DEV)
    pa=ABS[pool]; TOTAL=pa.sum(); ORC=np.cumsum(pa[np.argsort(-pa)])
    k1=max(1,int(.01*len(pool))); TT=set(pool[np.argsort(-pa)[:k1]].tolist())
    out={}
    seed_idx=rng.choice(pool,args.step,replace=False)
    for strat in ("random","mag80"):
        acq=seed_idx.copy(); curve=[]
        while True:
            Xtr=torch.tensor(X_ALL[acq],device=DEV); Ytr=torch.tensor(y[acq],device=DEV)
            e=Ens(Xtr.shape[1],args.heads,seed); e.fit(Xtr,Ytr,args.epochs)
            p,_=e.pred(Xte); pc,acx=p-p.mean(),Yte-Yte.mean()
            sl=(pc@acx/pc.pow(2).sum().clamp(min=1e-12)).item()
            r2=1-(Yte-p).pow(2).sum().item()/max((Yte-Yte.mean()).pow(2).sum().item(),1e-12)
            fnd=ABS[acq].sum()
            curve.append((len(acq),fnd/TOTAL,fnd/ORC[len(acq)-1],
                          len(TT&set(acq.tolist()))/k1,sl,r2))
            if len(acq)>=args.budget or stop: break
            rest=np.setdiff1d(pool,acq)
            if strat=="random":
                pick=rng.choice(rest,min(args.step,len(rest)),replace=False)
            else:
                cand=rng.choice(rest,min(80000,len(rest)),replace=False)
                pm,_=e.pred(torch.tensor(X_ALL[cand],device=DEV)); pm=pm.abs().cpu().numpy()
                nm=int(round(args.step*.8)); a1=cand[np.argsort(-pm)[:nm]]
                pick=np.concatenate([a1,rng.choice(np.setdiff1d(rest,a1),args.step-nm,replace=False)])
            acq=np.concatenate([acq,pick])
        out[strat]=curve
    return out

runs=[]
for s in range(args.seeds):
    if stop: break
    t0=time.time(); runs.append(run(s)); print(f"  seed {s} done  {time.time()-t0:.0f}s")
Lm=min(min(len(r[k]) for k in r) for r in runs)
n=[runs[0]["random"][i][0] for i in range(Lm)]
def ag(k,c): return np.array([[r[k][i][c] for r in runs] for i in range(Lm)])
R={k:[ag(k,c) for c in range(1,6)] for k in ("random","mag80")}
mm,rr = R["mag80"][0][-1].mean(), R["random"][0][-1].mean()
tm,tr = R["mag80"][2][-1].mean(), R["random"][2][-1].mean()
res={"model":args.model,"layer":L,"sae_steps":args.sae_steps,"recon_err":float(fvu),"l0":float(l0),
     "rows":int(len(y)),"features":int(len(ALLF)),
     "mass_mag80":float(mm),"mass_random":float(rr),"discovery_x":float(mm/max(rr,1e-12)),
     "oracle_pct":float(R["mag80"][1][-1].mean()),
     "top1_mag80":float(tm),"top1_x":float(tm/max(tr,1e-12)),
     "slope":float(R["mag80"][3][-1].mean()),"r2":float(R["mag80"][4][-1].mean()),
     "slope_random":float(R["random"][3][-1].mean()),"r2_random":float(R["random"][4][-1].mean())}
print(f"\n  LAYER {L} RESULT   (mag80 + log target, budget {args.budget})")
print(f"    SAE               recon-err {fvu*100:.1f}%   active/token {l0:.1f}   rows {len(y):,}")
print(f"    mass found        {mm*100:.2f}%  vs random {rr*100:.2f}%   ->  {mm/max(rr,1e-12):.2f}x")
print(f"    oracle bound      {R['mag80'][1][-1].mean()*100:.1f}%")
print(f"    top-1% recall     {tm*100:.1f}%  vs random {tr*100:.2f}%   ->  {tm/max(tr,1e-12):.1f}x")
print(f"    calib slope       {res['slope']:.3f}   (random acquisition: {res['slope_random']:.3f})")
print(f"    R2                {res['r2']:.3f}   (random acquisition: {res['r2_random']:.3f})")
od = f"{ROOT}/findings/layers" if args.model == "a" else f"{ROOT}/findings/layers_{args.model}"
os.makedirs(od, exist_ok=True)
json.dump(res, open(f"{od}/L{L}.json","w"), indent=1)
print(f"\n  -> {od}/L{L}.json")
