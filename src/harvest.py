#!/usr/bin/env python3
"""
V2 STEP 1 - build a trustworthy (feature, context) -> delta-loss dataset.

    python3 harvest.py                 # resumable, Ctrl-C safe
    python3 harvest.py --batches 20    # more precision

WHAT CHANGED FROM THE OLD HARVEST

The old version recorded one number per feature: its mean delta-loss over ~3,000
token positions. That was the wrong target, and the measurement proved it:

    96.3% of a feature's causal effect lands in 1% of positions
    its effect where it fires is ~2,992x its effect where it doesn't

Averaging over 3,000 positions to one scalar discards ~99% of the signal. So this
version keeps the PER-POSITION effects. Each row is one (feature, context) pair.

The cost is nothing extra: the per-token deltas were already being computed on
every forward pass and then thrown away. This just stops throwing them away.

INTERVENTION: mean-ablation is primary - clamp the feature to its dataset mean
rather than to zero, so A's residual stream stays closer to states it has seen.
Zero-ablation is recorded alongside for a subset, as a sanity comparison.
"""
import os, sys, json, time, signal, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from nn.gpt import GPT, Config
from nn.sae import SAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = f"{ROOT}/experiments/contextual.npz"
os.makedirs(f"{ROOT}/experiments", exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--batches", type=int, default=10)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--min-act", type=float, default=1e-3, help="record positions where the feature fires above this")
ap.add_argument("--zero-check", type=int, default=100, help="how many features to also zero-ablate")
ap.add_argument("--with-resid", action="store_true",
                help="also write experiments/row_resid.npy - the per-row layer-N residual "
                     "vector, aligned row-for-row with contextual.npz. Needed by "
                     "v3_discovery.py and v2_metric2.py. ~220 MB, regenerable, so it is "
                     "gitignored rather than committed.")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

stop = False
def onsig(*_):
    global stop; stop = True; print("\n  pausing - re-run to resume")
signal.signal(signal.SIGINT, onsig)

meta = json.load(open(f"{ROOT}/data/meta.json"))
cfg = Config(vocab_size=len(meta["vocab"]))
data = np.load(f"{ROOT}/data/corpus.npy")
ack = torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location=DEV)
A = GPT(cfg).to(DEV).eval(); A.load_state_dict(ack["model"])
for p in A.parameters(): p.requires_grad_(False)
bck = torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location=DEV)
B = SAE(cfg.n_embd).to(DEV); B.load_state_dict(bck["model"])
for p in B.parameters(): p.requires_grad_(False)
NF = B.d_hidden

print(f"A frozen @ {ack['step']:,}   B @ {bck['step']:,}   layer {args.layer}   {DEV}")
print(f"pause: Ctrl-C   resume: re-run\n")

# ---------- precompute the clean run once ----------
def mk(seed):
    rs = np.random.RandomState(seed)
    ix = rs.randint(0, len(data) - cfg.block_size - 1, args.batch)
    X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    return X, Y

@torch.no_grad()
def per_token(X, Y, iv=None):
    lg, _, _ = A(X, intervene=iv)
    return F.cross_entropy(lg.view(-1, lg.size(-1)), Y.reshape(-1), reduction="none")

print(f"  precomputing clean run over {args.batches} batches...")
BAT, BASE, ACT, TOK, TGT, POSIDX, RESID = [], [], [], [], [], [], []
with torch.no_grad():
    for k in range(args.batches):
        X, Y = mk(700 + k); BAT.append((X, Y))
        BASE.append(per_token(X, Y).cpu())
        _, _, ac = A(X, record=True)
        ACT.append(B.encode(ac["resid"][args.layer].reshape(-1, cfg.n_embd) / B.scale).cpu())
        TOK.append(X.reshape(-1).cpu()); TGT.append(Y.reshape(-1).cpu())
        POSIDX.append(torch.arange(cfg.block_size).repeat(args.batch))
MU = torch.cat(ACT).mean(0).to(DEV)          # what mean-ablation clamps each feature to
NPOS = sum(b.numel() for b in BASE)
print(f"  {NPOS:,} token positions   feature mean activation {MU.mean():.5f}\n")

def hook(i, mode):
    def fn(h):
        f = B.encode(h / B.scale)
        amt = f[..., i:i+1] if mode == "zero" else (f[..., i:i+1] - MU[i])
        return h - (amt * B.W_dec[i]) * B.scale
    return fn

# ---------- harvest ----------
done = set()
if os.path.exists(OUT):
    z = np.load(OUT, allow_pickle=True)
    done = set(int(f) for f in np.unique(z["feature"]))
    prev = {k: z[k] for k in z.files}
    print(f"  resuming: {len(done)} features already harvested")
else:
    prev = None

cols = {k: [] for k in ["feature","batch","pos","act","tok","tgt","base","d_mean","d_zero"]}
resid_rows = []
summ = {k: [] for k in ["feature","n","mean","se","ci_lo","ci_hi","fire_rate","mean_act"]}
todo = [i for i in range(NF) if i not in done]
t0 = time.time()
for n, i in enumerate(todo):
    if stop: break
    dm_all, want_zero = [], (i < args.zero_check)
    dz_all = []
    for k, (X, Y) in enumerate(BAT):
        dm = (per_token(X, Y, {args.layer: hook(i, "mean")}).cpu() - BASE[k])
        dm_all.append(dm)
        if want_zero:
            dz_all.append(per_token(X, Y, {args.layer: hook(i, "zero")}).cpu() - BASE[k])
    # keep only positions where this feature actually fires - that is where the
    # causal effect lives, and it is 0.7% of positions
    kept = 0
    per_pos = []
    for k, dm in enumerate(dm_all):
        a = ACT[k][:, i]
        sel = torch.nonzero(a > args.min_act).flatten()
        if sel.numel() == 0: continue
        kept += sel.numel()
        cols["feature"].append(np.full(sel.numel(), i, np.int32))
        cols["batch"].append(np.full(sel.numel(), k, np.int16))
        cols["pos"].append(POSIDX[k][sel].numpy().astype(np.int16))
        cols["act"].append(a[sel].numpy().astype(np.float32))
        cols["tok"].append(TOK[k][sel].numpy().astype(np.int16))
        cols["tgt"].append(TGT[k][sel].numpy().astype(np.int16))
        cols["base"].append(BASE[k][sel].numpy().astype(np.float32))
        cols["d_mean"].append(dm[sel].numpy().astype(np.float32))
        cols["d_zero"].append((dz_all[k][sel].numpy().astype(np.float32) if want_zero
                               else np.full(sel.numel(), np.nan, np.float32)))
        if args.with_resid: resid_rows.append(RESID[k][sel].numpy().astype(np.float32))
        per_pos.append(dm[sel].numpy())
    # per-feature summary WITH uncertainty, so B can tell bad prediction from noisy data
    v = np.concatenate(per_pos) if per_pos else np.zeros(0, np.float32)
    if v.size:
        bs = np.array([np.random.choice(v, v.size).mean() for _ in range(200)])
        summ["feature"].append(i); summ["n"].append(v.size)
        summ["mean"].append(v.mean()); summ["se"].append(v.std(ddof=1)/np.sqrt(v.size) if v.size>1 else 0.)
        summ["ci_lo"].append(np.percentile(bs,2.5)); summ["ci_hi"].append(np.percentile(bs,97.5))
        summ["fire_rate"].append(kept/NPOS); summ["mean_act"].append(float(MU[i]))
    if (n+1) % 100 == 0:
        el = time.time()-t0
        print(f"    {n+1}/{len(todo)}   {el:.0f}s   eta {el/(n+1)*(len(todo)-n-1):.0f}s")

# ---------- save ----------
new = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
new.update({"s_"+k: np.array(v) for k, v in summ.items()})
if prev is not None:
    for k in new:
        if k in prev and prev[k].size: new[k] = np.concatenate([prev[k], new[k]])
np.savez_compressed(OUT, **new)
if args.with_resid and resid_rows:
    # aligned row-for-row with contextual.npz by construction, since both are built
    # in the same pass. Previously this array had NO generating script at all.
    RP = f"{ROOT}/experiments/row_resid.npy"
    np.save(RP, np.concatenate(resid_rows))
    print(f"  wrote experiments/row_resid.npy  {os.path.getsize(RP)/1e6:.0f} MB")

print(f"\n  {'paused' if stop else 'done'}: {len(new['feature']):,} (feature, context) rows"
      f"  from {len(new['s_feature']):,} features   {os.path.getsize(OUT)/1e6:.1f} MB")
print(f"  vs the old dataset: 1,024 rows (one scalar per feature)")
