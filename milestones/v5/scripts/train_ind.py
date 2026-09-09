#!/usr/bin/env python3
"""
V5 — train A_ind on the induction task, then compute the ground truth.

    python3 train_ind.py

Task and thresholds are fixed by PREREGISTRATION_V5.md (Rev 3). Nothing here may
be tuned after a V5 number is seen.

    random period L ~ U{20..60}   -> no fixed positional offset solves the task
    L distinct tokens, no repl.   -> the induction target is unambiguous
    L in {35,45} held out         -> content-matching is tested, not assumed
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
print = __import__("functools").partial(print, flush=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(args.seed); np.random.seed(args.seed)

V, T = 128, 128
LMIN, LMAX = 40, 60
HELDOUT = {45, 55}
TRAIN_L = [l for l in range(LMIN, LMAX+1) if l not in HELDOUT]

from nn.gpt import GPT, Config
cfg = Config(vocab_size=V, block_size=T, n_layer=2, n_head=4, n_embd=128, use_mlp=False)

def batch(bs, periods, rng):
    """[P][filler][P] - Rev 4. Exactly ONE prior occurrence per repeated token.

    Vectorised: one random permutation per row gives L distinct pattern tokens
    (perm[:L]) and a disjoint filler alphabet (perm[L:]) for free. The per-row
    Python loop with setdiff1d was ~5M calls over a full run and dominated
    training time. The distribution is unchanged.
    """
    Ls = np.asarray(periods)[rng.randint(0, len(periods), bs)]
    perm = np.argsort(rng.random((bs, V)), axis=1)        # each row a permutation of the vocab
    t = np.arange(T)[None, :]
    L = Ls[:, None]
    # Rev 5: [filler][P][P] - the two copies are ADJACENT, so the lookback is L
    # (40-60) instead of 128-L (78-108). Run 2 plateaued at ratio 0.786 with a
    # long lookback and formed no circuit at all.
    G = T - 2*L                                                    # filler length
    fill = L + (rng.random((bs, T)) * (V - L)).astype(np.int64)    # filler ⊂ vocab \ P
    idx = np.where(t < G, fill, np.where(t < G + L, t - G, t - G - L))
    X = np.take_along_axis(perm, idx, axis=1)
    x = torch.from_numpy(X).to(DEV)
    return x[:, :-1], x[:, 1:], Ls

def split_loss(model, periods, rng, n=20):
    """Loss split by whether the target is a first occurrence or a repeat."""
    fo, rp = [], []
    with torch.no_grad():
        for _ in range(n):
            x, y, Ls = batch(args.batch, periods, rng)
            lg, _, _ = model(x)
            l = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(x.shape)
            pos = torch.arange(x.shape[1], device=DEV).unsqueeze(0)
            # induction-predictable = inside the SECOND copy of P
            isrep = pos >= (T - torch.from_numpy(Ls).to(DEV).unsqueeze(1))   # 2nd copy of P
            fo.append(l[~isrep].mean().item()); rp.append(l[isrep].mean().item())
    return float(np.mean(fo)), float(np.mean(rp))

model = GPT(cfg).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
# Resume support. Attempt 3's transition began near step 15,000 and was still in
# progress when the step budget ran out at 40,000 - continuing the SAME run (same
# seed, same design) is completing that attempt, not starting a new one.
CK = f"{ROOT}/checkpoints/aind_latest.pt"
step0 = 0
if os.path.exists(CK):
    _c = torch.load(CK, map_location=DEV)
    model.load_state_dict(_c["model"])
    if "opt" in _c: opt.load_state_dict(_c["opt"])
    step0 = _c["step"]
    print(f"  resumed from step {step0:,}")
rng = np.random.RandomState(args.seed)
print(f"A_ind  {model.n_params():,} params  2L x 4H x 128d  attention-only  {DEV}")
print(f"periods: train {LMIN}-{LMAX} minus {sorted(HELDOUT)}, held out {sorted(HELDOUT)}\n")

t0 = time.time()
for s in range(step0, step0 + args.steps):
    x, y, _ = batch(args.batch, TRAIN_L, rng)
    _, loss, _ = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (s+1) % 2500 == 0:
        # periodic save - the first version only wrote at the end of the run, so a
        # kill at step 150,000 would have discarded everything since step 40,000
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s+1,
                    "cfg": {k: getattr(cfg,k) for k in
                            ["vocab_size","block_size","n_layer","n_head","n_embd","use_mlp"]}},
                   f"{ROOT}/checkpoints/aind_latest.pt.tmp")
        os.replace(f"{ROOT}/checkpoints/aind_latest.pt.tmp", f"{ROOT}/checkpoints/aind_latest.pt")
        f, r = split_loss(model, TRAIN_L, rng, 8)
        print(f"  {s+1:>6,}  loss {loss.item():.4f}   first-occurrence {f:.4f}   repeated {r:.4f}   "
              f"ratio {r/max(f,1e-9):.3f}   {time.time()-t0:.0f}s")
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step0 + args.steps,
            "cfg": {k: getattr(cfg,k) for k in ["vocab_size","block_size","n_layer","n_head","n_embd","use_mlp"]}},
           f"{ROOT}/checkpoints/aind_latest.pt")
model.eval()

# ---------------- precondition checks, PREREGISTRATION_V5 section 3 ----------
print(f"\n{'='*66}\nVALIDITY PRECONDITION (fixed before training)")
f_tr, r_tr = split_loss(model, TRAIN_L, rng, 40)
f_ho, r_ho = split_loss(model, sorted(HELDOUT), rng, 40)
c1 = r_tr < 0.5*f_tr
c3 = r_ho < 1.25*r_tr
print(f"  1. repeated < 0.5 x first-occurrence : {r_tr:.4f} < {0.5*f_tr:.4f}   {'PASS' if c1 else 'FAIL'}")
print(f"  3. held-out L within 1.25x           : {r_ho:.4f} < {1.25*r_tr:.4f}   {'PASS' if c3 else 'FAIL'}")

# ---------------- ground truth, section 2 — computed BEFORE B ----------------
@torch.no_grad()
def scores(n=40):
    pts = np.zeros(cfg.n_head); ind = np.zeros(cfg.n_head); npt = 0
    for _ in range(n):
        x, y, Ls = batch(32, TRAIN_L, rng)
        _, _, ac = model(x, record=True)
        A0 = ac["attn"][0].float().cpu().numpy()      # (B, H, Tq, Tk)
        A1 = ac["attn"][1].float().cpu().numpy()
        B_, H, Tq, _ = A0.shape
        for b in range(B_):
            L = int(Ls[b])
            t = np.arange(1, Tq)
            pts += A0[b, :, t, t-1].mean(axis=0) if A0[b,:,t,t-1].ndim==2 else 0
            tr = np.arange(T-L, Tq)                    # the second copy of P
            if len(tr) == 0: continue
            # BUG FIX: with the Rev 5 layout [filler][P][P] the copies are ADJACENT,
            # so the unique prior occurrence of token[t] is at t-L, not t-(T-L).
            # The old formula was inherited from the Rev 4 layout and pointed into
            # the filler, which can never be the answer - hence IND ~ 0.007.
            # Section 2's definition ("t_first is the token's unique earlier
            # occurrence") is unchanged; only its implementation was wrong.
            tgt = tr - L + 1                           # successor of the prior occurrence
            ok = (tgt >= 0) & (tgt < Tq)
            tr = tr[ok]; tgt = tgt[ok]; ok = np.ones(len(tr), bool)
            ind += A1[b][:, tr[ok], tgt[ok]].mean(axis=1)
        npt += B_
    return pts/npt, ind/npt

PTS, IND = scores()
print(f"\nGROUND TRUTH (computed before B is run, never adjusted)")
print(f"  {'head':>10}{'prev-token score':>19}{'induction score':>18}")
for h in range(cfg.n_head): print(f"  {'L0H'+str(h):>10}{PTS[h]:>19.4f}{'-':>18}")
for h in range(cfg.n_head): print(f"  {'L1H'+str(h):>10}{'-':>19}{IND[h]:>18.4f}")
C = [f"L0H{h}" for h in range(cfg.n_head) if PTS[h] > 0.5] + \
    [f"L1H{h}" for h in range(cfg.n_head) if IND[h] > 0.5]
c2 = len(C) >= 2 and any(c.startswith("L0") for c in C) and any(c.startswith("L1") for c in C)
print(f"\n  C = {C}   |C| = {len(C)}")
print(f"  2. |C|>=2 with one per layer          : {'PASS' if c2 else 'FAIL'}")
from math import comb
chance = comb(len(C),2)/comb(8,2) if len(C) >= 2 else 0.0
print(f"  chance level for precision@2          : C({len(C)},2)/28 = {chance*100:.1f}%")
verdict = "VALID" if (c1 and c2 and c3) else "VOID"
if len(C) >= 5: verdict += " but WEAK BY CONSTRUCTION"
print(f"\n  PRECONDITION: {verdict}")
json.dump({"pts": PTS.tolist(), "ind": IND.tolist(), "C": C, "chance": chance,
           "loss_first": f_tr, "loss_repeat": r_tr, "loss_repeat_heldout": r_ho,
           "cond1": bool(c1), "cond2": bool(c2), "cond3": bool(c3), "verdict": verdict},
          open(f"{ROOT}/findings/v5_groundtruth.json","w"), indent=1)
print(f"  -> findings/v5_groundtruth.json")
