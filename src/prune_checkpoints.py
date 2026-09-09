#!/usr/bin/env python3
"""
Keep only the checkpoints where A actually CHANGED.

    python3 prune_checkpoints.py                 # measure (cached) + sweep + report
    python3 prune_checkpoints.py --attn 0.004    # pick a threshold
    python3 prune_checkpoints.py --attn 0.004 --apply     # delete the rest

TWO PHASES, on purpose. Measuring means loading every checkpoint and running a
forward pass - slow. Deciding what to keep is arithmetic - instant. So phase 1
caches a signature per checkpoint, and phase 2 can re-threshold for free.

WHAT "MAJOR CHANGE" MEANS HERE

I first tried KL divergence between A's next-token distributions. It failed, and
the failure is worth recording: adjacent checkpoints differ by ~0.18 nats and that
number barely moves across the whole run (0.29 early, 0.17 late). The floor is the
noise of a single SGD step, so KL cannot separate "a circuit formed" from "that
batch happened to be different". Measured, discarded.

What does work is ATTENTION DRIFT - the mean absolute change in A's attention
patterns on a fixed probe. It falls monotonically (0.0027 -> 0.0015), because
early training reorganises routing constantly and later training barely does.
Attention is where a transformer decides what to read, so a change there means A
is genuinely computing differently.

Drift is measured against the last KEPT checkpoint, not the previous one, so it
ACCUMULATES across a plateau. A quiet stretch is therefore represented by one
checkpoint, and a turbulent stretch by many - which is the behaviour we want.

Val-loss milestones are kept too, so the record always contains A's best models.
"""
import os, sys, json, glob, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn.gpt import GPT, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = f"{ROOT}/checkpoints/signatures.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--attn", type=float, default=None, help="circuit-drift threshold")
ap.add_argument("--val-delta", type=float, default=0.01, help="keep a new best val loss by this margin")
ap.add_argument("--keep-last", type=int, default=4)
ap.add_argument("--max-gap", type=int, default=5000,
                help="coverage floor: never leave a hole bigger than this many steps")
ap.add_argument("--apply", action="store_true")
ap.add_argument("--remeasure", action="store_true", help="ignore the cache")
args = ap.parse_args()

files = sorted(glob.glob(f"{ROOT}/checkpoints/a_step*.pt"))
if not files: sys.exit("no checkpoints found")
steps = [int(os.path.basename(f)[6:-3]) for f in files]

# ---------------- phase 1: measure (cached) ----------------
cache = {}
if os.path.exists(SIG) and not args.remeasure:
    z = np.load(SIG)
    cache = {int(s): (z["sig"][i], float(z["val"][i])) for i, s in enumerate(z["steps"])}

todo = [f for f, s in zip(files, steps) if s not in cache]
if todo:
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    meta = json.load(open(f"{ROOT}/data/meta.json"))
    cfg = Config(vocab_size=len(meta["vocab"]))
    data = np.load(f"{ROOT}/data/corpus.npy")
    # one fixed probe batch for every checkpoint, or we would be measuring the
    # difference between batches instead of the difference between models
    rs = np.random.RandomState(1234)
    ix = rs.randint(0, len(data) - cfg.block_size - 1, 8)
    X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    model = GPT(cfg).to(DEV).eval()
    print(f"measuring {len(todo)} new checkpoint(s) on {DEV}...")
    with torch.no_grad():
        for n, f in enumerate(todo):
            ck = torch.load(f, map_location=DEV)
            model.load_state_dict(ck["model"])
            _, loss, acts = model(X, Y, record=True)
            sig = np.concatenate([a[:, :, ::4, ::4].float().cpu().numpy().ravel()
                                  for a in acts["attn"]]).astype(np.float32)
            cache[ck["step"]] = (sig, float(loss))
            if (n+1) % 50 == 0: print(f"  {n+1}/{len(todo)}")
    ks = sorted(cache)
    np.savez_compressed(SIG, steps=np.array(ks),
                        sig=np.stack([cache[k][0] for k in ks]),
                        val=np.array([cache[k][1] for k in ks]))
    print(f"  cached -> checkpoints/signatures.npz "
          f"({os.path.getsize(SIG)/1e6:.0f} MB)\n")

order = sorted(cache)
sigs = np.stack([cache[s][0] for s in order])
vals = np.array([cache[s][1] for s in order])

# ---------------- phase 2: select (instant) ----------------
def select(tau, val_delta, keep_last, max_gap):
    """Change-driven, with a coverage floor.

    Drift alone leaves huge holes once A converges - attention stops moving, so
    nothing trips the threshold for tens of thousands of steps. That is a true
    statement about the training run but a bad development record, so max_gap
    guarantees a sample even through the quiet stretches.
    """
    keep, ref, best, last = [0], sigs[0], vals[0], order[0]
    for i in range(1, len(order)):
        drift = float(np.abs(sigs[i] - ref).mean())
        newest    = i >= len(order) - keep_last
        milestone = vals[i] < best - val_delta
        stale     = (order[i] - last) >= max_gap
        if drift > tau or newest or milestone or stale:
            keep.append(i); ref = sigs[i]; last = order[i]
        best = min(best, vals[i])
    return keep

MB = os.path.getsize(files[0]) / 1e6
if args.attn is None:
    print(f"threshold sweep  ({len(order)} checkpoints, {MB*len(order):.0f} MB total)\n")
    print(f"  {'threshold':>10} {'kept':>6} {'MB kept':>9} {'MB freed':>9}   coverage")
    for tau in [0.0015, 0.002, 0.003, 0.004, 0.006, 0.010, 0.020]:
        k = select(tau, args.val_delta, args.keep_last, args.max_gap)
        gaps = np.diff([order[i] for i in k])
        print(f"  {tau:>10.4f} {len(k):>6} {MB*len(k):>9.0f} {MB*(len(order)-len(k)):>9.0f}"
              f"   median gap {int(np.median(gaps)) if len(gaps) else 0:>6,} steps, "
              f"max {int(gaps.max()) if len(gaps) else 0:,}")
    print(f"\n  pick one:  python3 prune_checkpoints.py --attn 0.004")
    print(f"  then:      python3 prune_checkpoints.py --attn 0.004 --apply")
    sys.exit(0)

keep = select(args.attn, args.val_delta, args.keep_last, args.max_gap)
kset = {order[i] for i in keep}
print(f"threshold {args.attn}   keeping {len(keep)} of {len(order)}\n")
print(f"  {'step':>9} {'val loss':>9} {'drift':>9}  reason")
ref, best = sigs[0], vals[0]
for i in range(len(order)):
    if order[i] not in kset:
        best = min(best, vals[i]); continue
    drift = 0.0 if i == 0 else float(np.abs(sigs[i] - ref).mean())
    why = ("first" if i == 0 else
           "newest" if i >= len(order)-args.keep_last else
           "best val" if vals[i] < best - args.val_delta else
           "circuit drift" if drift > args.attn else "coverage")
    print(f"  {order[i]:>9,} {vals[i]:>9.4f} {drift:>9.5f}  {why}")
    ref = sigs[i]; best = min(best, vals[i])

drop = [f for f, s in zip(files, steps) if s not in kset]
print(f"\n  keep {len(kset)} ({MB*len(kset):.0f} MB) | drop {len(drop)} ({MB*len(drop):.0f} MB freed)")
json.dump({"threshold": args.attn, "kept": sorted(kset)},
          open(f"{ROOT}/checkpoints/prune_report.json", "w"), indent=1)
if args.apply:
    for f in drop: os.remove(f)
    print(f"  deleted {len(drop)} checkpoints.")
else:
    print(f"  DRY RUN - nothing deleted. Add --apply to remove the {len(drop)} drops.")
