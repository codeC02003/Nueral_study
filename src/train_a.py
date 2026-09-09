#!/usr/bin/env python3
"""
NETWORK A - trains forever on Python source.

    python3 train_a.py                 # runs until you stop it (Ctrl-C)
    python3 train_a.py --steps 2000    # bounded run
    python3 train_a.py --device cpu    # force CPU

Writes continuously:
    checkpoints/a_latest.pt     newest weights (B reads this)
    checkpoints/a_step*.pt      periodic snapshots (B can study A's DEVELOPMENT)
    status/a.json               live metrics for the dashboard
    samples/a_step*.py          what A can write, at each stage
"""
import os, sys, json, time, math, argparse, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from nn.gpt import GPT, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("checkpoints", "status", "samples"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=0, help="0 = forever; N MORE steps from where we are")
ap.add_argument("--until", type=int, default=0,
                help="absolute target step - stop AT this step regardless of resume point")
ap.add_argument("--device", default=None)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--ckpt-every", type=int, default=250)
ap.add_argument("--resume", action="store_true", default=True)
ap.add_argument("--seed", type=int, default=None, help="init + data-order seed")
ap.add_argument("--tag", default="a", help="checkpoint prefix, e.g. --tag a2")
args = ap.parse_args()

DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
if args.seed is not None:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
TAG = args.tag

# ---- data -----------------------------------------------------------------
meta = json.load(open(f"{ROOT}/data/meta.json"))
stoi, vocab = meta["stoi"], meta["vocab"]
itos = {i: c for c, i in stoi.items()}
cache = f"{ROOT}/data/corpus.npy"
if os.path.exists(cache):
    data = np.load(cache)
else:
    text = open(f"{ROOT}/data/corpus.txt").read()
    data = np.array([stoi[c] for c in text], dtype=np.uint8)
    np.save(cache, data)
split = int(0.98 * len(data))
train, val = data[:split], data[split:]

cfg = Config(vocab_size=len(vocab))
def get_batch(arr, bs):
    ix = np.random.randint(0, len(arr) - cfg.block_size - 1, bs)
    x = np.stack([arr[i:i + cfg.block_size] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)

# ---- model ----------------------------------------------------------------
model = GPT(cfg).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
step0 = 0
latest = f"{ROOT}/checkpoints/{TAG}_latest.pt"
if args.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location=DEV)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    print(f"resumed from step {step0:,}")

WARMUP = 200
def lr_at(s):
    return args.lr * min(1.0, (s + 1) / WARMUP)

print(f"NETWORK A  |  {model.n_params():,} params  |  device {DEV}  |  "
      f"corpus {len(data):,} chars  |  vocab {len(vocab)}")
print(f"target: continuous. Ctrl-C to stop; --resume picks up where it left off.\n")

stop = False
def onsig(*_):
    global stop; stop = True
    print("\nstopping after this step...")
signal.signal(signal.SIGINT, onsig)

@torch.no_grad()
def evaluate(n=20):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(val, args.batch)
        tot += model(x, y)[1].item()
    model.train()
    return tot / n

@torch.no_grad()
def sample(prompt="def ", n=400):
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=DEV)
    out = model.generate(idx, n)[0].tolist()
    model.train()
    return "".join(itos[i] for i in out)

hist, t_start, step = [], time.time(), step0
best = float("inf")
# --until is absolute; --steps is relative. Conflating them once already sent a
# preregistered replication 46,250 steps past its target.
TARGET = args.until if args.until else (step0 + args.steps if args.steps else 0)
if TARGET: print(f"target: stop at step {TARGET:,}\n")
while not stop and (TARGET == 0 or step < TARGET):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train, args.batch)
    _, loss, _ = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    hist.append(loss.item())

    if step % 25 == 0:
        recent = float(np.mean(hist[-25:]))
        elapsed = time.time() - t_start
        json.dump({
            "step": step, "loss": recent, "bpc": recent / math.log(2),
            "lr": lr_at(step), "device": DEV, "params": model.n_params(),
            "steps_per_sec": (step - step0) / max(elapsed, 1e-9),
            "elapsed": elapsed, "history": hist[-400::4], "alive": True,
            "updated": time.time(),
        }, open(f"{ROOT}/status/{TAG}.json", "w"))
        print(f"  step {step:>7,}  loss {recent:.4f}  bits/char {recent/math.log(2):.3f}  "
              f"{(step-step0)/max(elapsed,1e-9):.1f} it/s")

    if step % args.ckpt_every == 0:
        vl = evaluate()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": vars(cfg), "val": vl}, latest + ".tmp")
        os.replace(latest + ".tmp", latest)          # atomic: B never reads a half-written file
        torch.save({"model": model.state_dict(), "step": step, "cfg": vars(cfg), "val": vl},
                   f"{ROOT}/checkpoints/{TAG}_step{step:07d}.pt")
        txt = sample()
        open(f"{ROOT}/samples/{TAG}_step{step:07d}.py", "w").write(txt)
        print(f"  --- checkpoint {step:,}  val {vl:.4f} ---")
        print("\n".join("      | " + l for l in txt.split("\n")[:8]))
        print()

# always persist the final state - otherwise --until 83750 with --ckpt-every 10000
# silently leaves the newest checkpoint at 80,000
vl = evaluate()
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "step": step, "cfg": vars(cfg), "val": vl}, latest + ".tmp")
os.replace(latest + ".tmp", latest)
print(f"final checkpoint saved at step {step:,} (val {vl:.4f})")
json.dump({"alive": False, "step": step, "updated": time.time()},
          open(f"{ROOT}/status/{TAG}.json", "w"))
print(f"\nstopped at step {step:,}" + (f" (target {TARGET:,})" if TARGET else "") + ".")
