#!/usr/bin/env python3
"""
NETWORK B - reads Network A, continuously, while A is still learning.

    python3 train_b.py                # follows A's latest checkpoint forever
    python3 train_b.py --layer 2      # which of A's layers to dissect
    python3 train_b.py --steps 2000   # bounded run

B is never told what to look for. Its only signal is: reconstruct A's internal
activations using as few features as possible. Whatever structure exists inside
A is what makes that possible.

Writes:
    checkpoints/b_latest.pt   B's weights
    status/b.json             live metrics
    findings/features.json    B's readings - what each discovered feature fires on
"""
import os, sys, json, time, argparse, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from nn.gpt import GPT, Config
from nn.sae import SAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/findings", exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--expansion", type=int, default=8)
ap.add_argument("--l1", type=float, default=1e-2)   # calibrated; see nn/sae.py
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

meta = json.load(open(f"{ROOT}/data/meta.json"))
itos = {i: c for c, i in meta["stoi"].items()}
data = np.load(f"{ROOT}/data/corpus.npy")
cfg = Config(vocab_size=len(meta["vocab"]))
LATEST = f"{ROOT}/checkpoints/a_latest.pt"

while not os.path.exists(LATEST):
    print("waiting for A's first checkpoint..."); time.sleep(5)

A = GPT(cfg).to(DEV).eval()
a_step = -1
def refresh_A():
    """Reload A if it has moved on. B chases a MOVING target - that is the point."""
    global a_step
    ck = torch.load(LATEST, map_location=DEV)
    if ck["step"] != a_step:
        A.load_state_dict(ck["model"]); a_step = ck["step"]
        return True
    return False
refresh_A()

B = SAE(cfg.n_embd, args.expansion, args.l1).to(DEV)
opt = torch.optim.Adam(B.parameters(), lr=args.lr)
if os.path.exists(f"{ROOT}/checkpoints/b_latest.pt"):
    # A long-running system outlives its own code. If B's architecture changed
    # since the checkpoint was written, start fresh rather than crash.
    try:
        ck = torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location=DEV)
        B.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        print(f"resumed B from step {ck['step']:,}")
    except (RuntimeError, KeyError) as e:
        print(f"checkpoint incompatible with current SAE ({str(e).splitlines()[0][:60]}...)")
        print("starting B from scratch.")
        os.replace(f"{ROOT}/checkpoints/b_latest.pt", f"{ROOT}/checkpoints/b_stale.pt")

print(f"NETWORK B  |  SAE {cfg.n_embd} -> {B.d_hidden} features  |  reading A layer {args.layer}")
print(f"           |  {sum(p.numel() for p in B.parameters()):,} params  |  device {DEV}\n")

stop = False
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))

def get_acts(bs):
    """Harvest A's residual stream. Detached: B observes A, never edits it."""
    ix = np.random.randint(0, len(data) - cfg.block_size - 1, bs)
    x = torch.from_numpy(np.stack([data[i:i + cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    with torch.no_grad():
        _, _, acts = A(x, record=True)
    return acts["resid"][args.layer].reshape(-1, cfg.n_embd), x

B.calibrate(get_acts(args.batch)[0])      # A's residual norm grows per layer; rescale first
print(f"           |  calibrated to A layer {args.layer} (scale {B.scale.item():.3f})\n")

step, hist, t0 = 0, [], time.time()
fire_count = torch.zeros(B.d_hidden, device=DEV)
while not stop and (args.steps == 0 or step < args.steps):
    if step % 200 == 0 and refresh_A():
        print(f"  >> A moved to step {a_step:,}; B is now reading the newer A")
    acts, _ = get_acts(args.batch)
    _, f, loss, mse, sp = B(acts)
    opt.zero_grad(set_to_none=True); loss.backward()
    B.W_dec.grad -= (B.W_dec.grad * B.W_dec.data).sum(-1, keepdim=True) * B.W_dec.data
    opt.step(); B.normalize_decoder()
    step += 1; hist.append(mse.item())
    fire_count += (f > 0).float().sum(0)

    if step % 25 == 0:
        l0, fvu, alive = B.stats(acts)
        dead = int((fire_count == 0).sum())
        json.dump({"step": step, "a_step": a_step, "mse": float(np.mean(hist[-25:])),
                   "fvu": fvu, "l0": l0, "features": B.d_hidden, "dead": dead,
                   "alive": True, "layer": args.layer, "updated": time.time(),
                   "history": hist[-400::4],
                   "steps_per_sec": step / max(time.time() - t0, 1e-9)},
                  open(f"{ROOT}/status/b.json", "w"))
        print(f"  step {step:>7,}  recon-err {fvu*100:5.1f}%  "
              f"active/token {l0:6.1f}/{B.d_hidden}  dead {dead:>4}  (A @ {a_step:,})")

    if step % 500 == 0:
        # ---- B reports what it found -------------------------------------
        # For each feature: the text that makes it fire hardest. No labels.
        acts_big, toks = get_acts(64)
        with torch.no_grad():
            F_ = B.encode(acts_big)
        flat = toks.reshape(-1)
        strength = F_.max(0).values
        top_feats = strength.argsort(descending=True)[:24].tolist()
        found = []
        for fi in top_feats:
            top_pos = F_[:, fi].argsort(descending=True)[:6].tolist()
            ctxs = []
            for p in top_pos:
                lo = max(0, p - 18)
                s = "".join(itos[int(t)] for t in flat[lo:p + 1].tolist())
                ctxs.append(s.replace("\n", "\\n"))
            found.append({"feature": fi, "max_act": float(strength[fi]), "contexts": ctxs})
        json.dump({"a_step": a_step, "b_step": step, "layer": args.layer,
                   "features": found}, open(f"{ROOT}/findings/features.json", "w"), indent=1)
        torch.save({"model": B.state_dict(), "opt": opt.state_dict(), "step": step},
                   f"{ROOT}/checkpoints/b_latest.pt")
        print(f"\n  --- B's reading of A @ step {a_step:,} (top 3 of 24 written to findings/) ---")
        for it in found[:3]:
            print(f"      feature {it['feature']:>4} fires on: " +
                  "  |  ".join(repr(c[-22:]) for c in it["contexts"][:3]))
        print()

json.dump({"alive": False, "step": step, "updated": time.time()}, open(f"{ROOT}/status/b.json", "w"))
print(f"\nB stopped at step {step:,}.")
