#!/usr/bin/env python3
"""
TIER 4, STEP 1 - are B's features actually USED by A, or only correlated with it?

    python3 ablate.py                # top 40 features vs 40 random controls
    python3 ablate.py --top 100      # more features
    python3 ablate.py --layer 2      # which tap to test

THE QUESTION
B found directions that fire on interpretable things - exception names, `None`,
assignment. That is a CORRELATIONAL finding: the information is present. It does
not show A uses it. A probe can decode something at 100% accuracy that the model
ignores entirely downstream, and that is the single most common way an
interpretability result turns out to be wrong.

THE TEST
Remove a feature's contribution from A's residual stream and measure what happens
to A's loss. Ablating feature i means subtracting exactly what that feature
contributed to the reconstruction:

    x' = x - f_i(x) * W_dec[i]

THE CONTROL, WHICH IS THE ENTIRE POINT
"Ablating feature 507 hurt A" means nothing on its own - deleting ANY direction
hurts A. So we also ablate random directions, and we normalise by how much
magnitude was removed:

    damage per unit removed  =  delta_loss / ||what we deleted||

If B's features are genuinely load-bearing, they must do MORE damage per unit
removed than random directions. If they do not, the SAE found statistical
structure A does not rely on, and no amount of prediction training would fix that.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from nn.gpt import GPT, Config
from nn.sae import SAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

meta = json.load(open(f"{ROOT}/data/meta.json"))
itos = {i: c for c, i in meta["stoi"].items()}
cfg = Config(vocab_size=len(meta["vocab"]))
data = np.load(f"{ROOT}/data/corpus.npy")

ck = torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location=DEV)
A = GPT(cfg).to(DEV).eval(); A.load_state_dict(ck["model"])
bck = torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location=DEV)
B = SAE(cfg.n_embd).to(DEV); B.load_state_dict(bck["model"])
for p in A.parameters(): p.requires_grad_(False)

print(f"A frozen at step {ck['step']:,}   B at step {bck['step']:,}   tap = layer {args.layer}\n")

# one fixed batch for every measurement, so differences come from the
# intervention and not from the data
rs = np.random.RandomState(7)
ix = rs.randint(0, len(data) - cfg.block_size - 1, args.batch)
X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)

@torch.no_grad()
def per_token_loss(intervene=None):
    logits, _, _ = A(X, intervene=intervene)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), Y.reshape(-1), reduction="none")

base_tok = per_token_loss()
base = base_tok.mean().item()
print(f"baseline loss {base:.4f}\n")

# which features fire most on this batch -> the ones worth testing
with torch.no_grad():
    _, _, acts = A(X, record=True)
    R = acts["resid"][args.layer]
    Fm = B.encode(R.reshape(-1, cfg.n_embd) / B.scale)
top = Fm.max(0).values.argsort(descending=True)[:args.top].tolist()

removed_mag = {}
def feat_hook(i):
    def fn(h):
        f = B.encode(h / B.scale)
        contrib = (f[..., i:i+1] * B.W_dec[i]) * B.scale
        removed_mag[i] = contrib.norm().item()
        return h - contrib
    return fn

def rand_hook(k, target_norm):
    g = torch.Generator(device="cpu").manual_seed(1000 + k)
    d = torch.randn(cfg.n_embd, generator=g).to(DEV)
    d = d / d.norm()
    def fn(h):
        proj = (h @ d).unsqueeze(-1) * d
        cur = proj.norm().item()
        if cur > 1e-9:                      # match the magnitude we delete
            proj = proj * (target_norm / cur)
        removed_mag[f"r{k}"] = proj.norm().item()
        return h - proj
    return fn

# ---------- real features ----------
rows = []
for i in top:
    L = per_token_loss({args.layer: feat_hook(i)})
    d = L.mean().item() - base
    rows.append((i, d, removed_mag[i], L))

mean_removed = float(np.mean([r[2] for r in rows]))

# ---------- matched random control ----------
ctrl = []
for k in range(args.top):
    L = per_token_loss({args.layer: rand_hook(k, mean_removed)})
    ctrl.append((L.mean().item() - base, removed_mag[f"r{k}"]))

def eff(d, m): return d / max(m, 1e-9)
fe = np.array([eff(d, m) for _, d, m, _ in rows])
ce = np.array([eff(d, m) for d, m in ctrl])

print(f"{'':<26}{'mean d-loss':>13}{'per unit removed':>19}")
print(f"  {'B features (n=' + str(len(rows)) + ')':<24}{np.mean([r[1] for r in rows]):>13.5f}{fe.mean():>19.6f}")
print(f"  {'random controls (n=' + str(len(ctrl)) + ')':<24}{np.mean([c[0] for c in ctrl]):>13.5f}{ce.mean():>19.6f}")

ratio = fe.mean() / max(ce.mean(), 1e-12)
pooled = np.sqrt((fe.var(ddof=1) + ce.var(ddof=1)) / 2)
cohen = (fe.mean() - ce.mean()) / max(pooled, 1e-12)
print(f"\n  ratio          : {ratio:.2f}x  (>1 means B's features are more load-bearing)")
print(f"  effect size    : Cohen's d = {cohen:.2f}")
print(f"  verdict        : ", end="")
if ratio > 1.5 and cohen > 0.8:   print("B's features are CAUSALLY USED by A.")
elif ratio > 1.1 and cohen > 0.3: print("weak but real signal - features are somewhat load-bearing.")
else:                             print("NOT distinguishable from random. See notes below.")

# ---------- which characters does each feature actually serve? ----------
print(f"\n{'-'*76}\nthe 10 most load-bearing features, and WHERE removing them hurts A")
print(f"{'feat':>6}{'d-loss':>10}{'per unit':>11}   characters whose prediction degrades most")
tgt = Y.reshape(-1)
for i, d, m, L in sorted(rows, key=lambda r: -eff(r[1], r[2]))[:10]:
    delta = (L - base_tok)
    per_char = {}
    for c in delta.topk(60).indices.tolist():
        ch = itos[int(tgt[c])]
        per_char[ch] = per_char.get(ch, 0) + float(delta[c])
    worst = sorted(per_char.items(), key=lambda kv: -kv[1])[:6]
    s = "  ".join(f"{repr(c)}{v:+.2f}" for c, v in worst)
    print(f"{i:>6}{d:>10.5f}{eff(d,m):>11.5f}   {s}")

json.dump({"a_step": ck["step"], "b_step": bck["step"], "layer": args.layer,
           "baseline": base, "ratio": float(ratio), "cohen_d": float(cohen),
           "features": [{"id": int(i), "d_loss": float(d), "removed": float(m),
                         "per_unit": float(eff(d, m))} for i, d, m, _ in rows],
           "control": [{"d_loss": float(d), "removed": float(m)} for d, m in ctrl]},
          open(f"{ROOT}/findings/ablation.json", "w"), indent=1)
print(f"\n  -> findings/ablation.json")
