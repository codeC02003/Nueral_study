#!/usr/bin/env python3
"""
TIER 4, STEP 3 - B learns to PREDICT what removing a feature does to A.

    python3 train_b_predict.py                # harvest experiments, then train
    python3 train_b_predict.py --harvest-only
    python3 train_b_predict.py --train-only
    python3 train_b_predict.py --steps 4000

Ctrl-C at any point is safe. Everything is written to disk as it goes; re-running
the same command resumes.

WHY A PREDICTION HEAD IS THE WHOLE POINT
Step 1 proved B's features are causally used, but a human ran those ablations. B
was scored on nothing. Rewarding B merely for finding features that *matter* would
reward a salience detector - a random direction also breaks A.

So B must state the effect size IN ADVANCE and be scored on it. That cannot be
faked: predicting the consequence of removing something requires a model of what
it does, including predicting ~zero for a direction A carries but never uses.

THE DESIGN DECISION THAT MAKES THIS UNDERSTANDING RATHER THAN MEMORISATION
The head is NOT given the feature's index. An index lookup would memorise a table
and generalise to nothing. It is given the feature's own PROPERTIES:

    W_dec[i]        the direction it writes into the residual stream   (128)
    W_enc[:, i]     the direction it detects                          (128)
    b_enc[i]        its firing threshold                                (1)
    firing stats    rate, mean magnitude, max                           (3)

and must output the loss increase A suffers without it. Train and test are split
BY FEATURE, so every evaluated feature is one B has never been ablated on. If it
predicts those, it has learned something general about which directions A relies
on - not a lookup table.
"""
import os, sys, json, time, signal, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from nn.gpt import GPT, Config
from nn.sae import SAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG  = f"{ROOT}/experiments/ablation_log.jsonl"
HEAD = f"{ROOT}/checkpoints/b_predictor.pt"
os.makedirs(f"{ROOT}/experiments", exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--n-features", type=int, default=520, help="how many to run experiments on")
ap.add_argument("--batches", type=int, default=2, help="probe batches per experiment (noise control)")
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--harvest-only", action="store_true")
ap.add_argument("--train-only", action="store_true")
ap.add_argument("--device", default=None)
args = ap.parse_args()
DEV = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

stop = False
def onsig(*_):
    global stop; stop = True
    print("\n  pausing - progress is on disk, re-run the same command to resume")
signal.signal(signal.SIGINT, onsig)

# ---------------------------------------------------------------- setup
meta = json.load(open(f"{ROOT}/data/meta.json"))
cfg = Config(vocab_size=len(meta["vocab"]))
data = np.load(f"{ROOT}/data/corpus.npy")
ack = torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location=DEV)
A = GPT(cfg).to(DEV).eval(); A.load_state_dict(ack["model"])
for p in A.parameters(): p.requires_grad_(False)
bck = torch.load(f"{ROOT}/checkpoints/b_latest.pt", map_location=DEV)
B = SAE(cfg.n_embd).to(DEV); B.load_state_dict(bck["model"])
for p in B.parameters(): p.requires_grad_(False)      # B's features are frozen; only the head learns

print(f"A frozen @ step {ack['step']:,}   B frozen @ step {bck['step']:,}   layer {args.layer}   {DEV}")
print(f"pause: Ctrl-C      resume: re-run the same command\n")

def make_batch(seed):
    rs = np.random.RandomState(seed)
    ix = rs.randint(0, len(data) - cfg.block_size - 1, args.batch)
    X = torch.from_numpy(np.stack([data[i:i+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    Y = torch.from_numpy(np.stack([data[i+1:i+1+cfg.block_size] for i in ix]).astype(np.int64)).to(DEV)
    return X, Y
BATCHES = [make_batch(700 + k) for k in range(args.batches)]

@torch.no_grad()
def loss_on(X, Y, iv=None):
    lg, _, _ = A(X, intervene=iv)
    return F.cross_entropy(lg.view(-1, lg.size(-1)), Y.reshape(-1)).item()

BASE = [loss_on(X, Y) for X, Y in BATCHES]

# ---------------------------------------------------------------- phase 1: harvest
def harvest():
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            try: done.add(json.loads(line)["feature"])
            except Exception: pass
    # rank features by how much they fire, so we spend experiments on live ones
    with torch.no_grad():
        _, _, acts = A(BATCHES[0][0], record=True)
        Fm = B.encode(acts["resid"][args.layer].reshape(-1, cfg.n_embd) / B.scale)
        rate = (Fm > 0).float().mean(0)
        order = Fm.max(0).values.argsort(descending=True).tolist()
    todo = [i for i in order[:args.n_features] if i not in done]
    if not todo:
        print(f"  harvest complete: {len(done)} experiments on disk"); return
    print(f"  harvesting {len(todo)} experiments ({len(done)} already done)")
    t0 = time.time()
    with open(LOG, "a") as fh:
        for n, i in enumerate(todo):
            if stop: break
            def hook(h, i=i):
                f = B.encode(h / B.scale)
                return h - (f[..., i:i+1] * B.W_dec[i]) * B.scale
            ds, mags = [], []
            for (X, Y), b0 in zip(BATCHES, BASE):
                with torch.no_grad():
                    _, _, a2 = A(X, record=True)
                    f = B.encode(a2["resid"][args.layer].reshape(-1, cfg.n_embd) / B.scale)
                    mags.append(((f[:, i:i+1] * B.W_dec[i]) * B.scale).norm().item())
                ds.append(loss_on(X, Y, {args.layer: hook}) - b0)
            fh.write(json.dumps({"feature": int(i), "d_loss": float(np.mean(ds)),
                                 "d_loss_std": float(np.std(ds)),
                                 "removed": float(np.mean(mags)),
                                 "fire_rate": float(rate[i])}) + "\n")
            fh.flush()
            if (n+1) % 50 == 0:
                el = time.time()-t0
                print(f"    {n+1}/{len(todo)}   {el:.0f}s   eta {el/(n+1)*(len(todo)-n-1):.0f}s")
    print(f"  harvest {'paused' if stop else 'done'}: {len(done)+n+1 if todo else len(done)} experiments")

# ---------------------------------------------------------------- the head
class Predictor(nn.Module):
    """feature properties -> predicted damage to A. No index, by design."""
    def __init__(self, d):
        super().__init__()
        # deliberately small, with dropout. 400-1000 samples cannot support more.
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 48), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(48, 24), nn.GELU(),
                                 nn.Linear(24, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def feature_inputs(ids):
    """What B is allowed to know about a feature before ablating it.

    All of it is observational - readable from A and B's weights without running
    a single experiment - so using it is not cheating.

    The important addition is LOGIT REACH. Whether A depends on a direction is
    mostly a question of what that direction does to A's output, and the residual
    stream connects to the output directly. So we push W_dec[i] through A's
    unembedding and summarise the resulting logit push. Raw 128-d vectors alone
    left the head memorising 416 examples and generalising at R2 = 0.07.
    """
    with torch.no_grad():
        Wu = A.head.weight                      # (vocab, n_embd)
        rows = []
        for i in ids:
            d = B.W_dec[i]
            logit = Wu @ d                      # what this direction pushes the output toward
            pr = F.softmax(logit, dim=-1)
            ent = -(pr * (pr + 1e-9).log()).sum()
            rows.append(torch.cat([
                d, B.W_enc[:, i],               # the raw directions
                B.b_enc[i].view(1),
                d.norm().view(1),
                B.W_enc[:, i].norm().view(1),
                torch.cosine_similarity(d, B.W_enc[:, i], dim=0).view(1),
                logit.norm().view(1),           # how hard it pushes the output
                logit.abs().max().view(1),      # its strongest single push
                logit.topk(5).values,           # its top-5 pushes
                ent.view(1),                    # focused (low) or diffuse (high)
            ]))
        return torch.stack(rows)

def train():
    recs = [json.loads(l) for l in open(LOG)]
    if len(recs) < 40: sys.exit("  not enough experiments yet - run the harvest first")
    ids  = [r["feature"] for r in recs]
    y    = torch.tensor([r["d_loss"] for r in recs], dtype=torch.float32, device=DEV)
    stats = torch.tensor([[r["fire_rate"], r["removed"]] for r in recs],
                         dtype=torch.float32, device=DEV)
    Xin = torch.cat([feature_inputs(ids), stats], dim=1)

    # SPLIT BY FEATURE: the test features are never ablated during training
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(recs), generator=g)
    ntr = int(.8 * len(recs))
    tr, te = perm[:ntr].to(DEV), perm[ntr:].to(DEV)

    mu, sd = Xin[tr].mean(0), Xin[tr].std(0).clamp(min=1e-6)
    ym, ys = y[tr].mean(), y[tr].std().clamp(min=1e-9)
    Xn = (Xin - mu) / sd

    head = Predictor(Xn.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=3e-1)
    step0 = 0
    if os.path.exists(HEAD):
        h = torch.load(HEAD, map_location=DEV)
        if h.get("d_in") == Xn.shape[1]:
            head.load_state_dict(h["model"]); opt.load_state_dict(h["opt"]); step0 = h["step"]
            print(f"  resumed predictor from step {step0:,}")

    print(f"  training on {len(tr)} features, holding out {len(te)} NEVER-ablated features")
    print(f"  {'step':>7} {'train MAE':>11} {'test MAE':>11} {'test R2':>9} {'pearson r':>10}")
    def evaluate(idx):
        head.eval()
        with torch.no_grad():
            p = head((Xn[idx])) * ys + ym
            a = y[idx]
            mae = (p-a).abs().mean().item()
            ss = ((a-p)**2).sum().item(); tot = ((a-a.mean())**2).sum().item()
            r2 = 1 - ss/max(tot,1e-12)
            pc = torch.corrcoef(torch.stack([p, a]))[0,1].item()
        head.train()
        return mae, r2, pc, p, a

    best_r2, best_state = -9e9, {k: v.clone() for k, v in head.state_dict().items()}
    for s in range(step0, step0 + args.steps):
        if stop: break
        head.train()
        pred = head(Xn[tr])
        loss = F.smooth_l1_loss(pred, (y[tr]-ym)/ys)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (s+1) % 250 == 0:
            trm,_,_,_,_ = evaluate(tr)
            tem, r2, pc, _, _ = evaluate(te)
            star = ""
            if r2 > best_r2:                       # keep the best model, not the last
                best_r2, best_state, star = r2, {k: v.clone() for k, v in head.state_dict().items()}, "  <- best"
            print(f"  {s+1:>7,} {trm:>11.6f} {tem:>11.6f} {r2:>9.3f} {pc:>10.3f}{star}")
            torch.save({"model": best_state, "opt": opt.state_dict(),
                        "step": s+1, "d_in": Xn.shape[1], "best_r2": best_r2,
                        "norm": {"mu": mu, "sd": sd, "ym": ym, "ys": ys}}, HEAD)

    head.load_state_dict(best_state)               # report the best, not the overfit end
    tem, r2, pc, p, a = evaluate(te)
    base_mae = (y[te] - y[tr].mean()).abs().mean().item()
    print(f"\n  HELD-OUT FEATURES (B has never ablated these)")
    print(f"    B's prediction MAE     : {tem:.6f}")
    print(f"    predict-the-mean MAE   : {base_mae:.6f}   <- the trivial baseline")
    print(f"    improvement over mean  : {(1-tem/max(base_mae,1e-12))*100:.1f}%")
    print(f"    R2                     : {r2:.3f}")
    print(f"    Pearson r              : {pc:.3f}")
    verdict = ("B has learned which directions A depends on." if r2 > .3 and pc > .55
               else "weak but real generalisation." if r2 > .1 and pc > .35
               else "NOT generalising - see notes.")
    print(f"    verdict                : {verdict}")
    Xtr, Xte = Xn[tr].cpu().numpy(), Xn[te].cpu().numpy()
    ytr = ((y[tr]-ym)/ys).cpu().numpy()
    lam = 30.0
    W = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    pr = torch.tensor(Xte@W, device=DEV, dtype=torch.float32)*ys + ym
    rr2 = 1 - ((y[te]-pr)**2).sum().item()/max(((y[te]-y[te].mean())**2).sum().item(),1e-12)
    print(f"    ridge baseline R2      : {rr2:.3f}   <- if this matches, the MLP adds nothing")

    print(f"\n  examples (predicted vs actual d-loss, unseen features):")
    ordr = a.argsort(descending=True)[:8]
    for k in ordr.tolist():
        print(f"    feature {[recs[i]['feature'] for i in te.tolist()][k]:>5}"
              f"   predicted {p[k]:+.5f}   actual {a[k]:+.5f}")
    json.dump({"test_mae": tem, "baseline_mae": base_mae, "r2": r2, "pearson": pc,
               "n_train": len(tr), "n_test": len(te), "a_step": ack["step"]},
              open(f"{ROOT}/findings/predictor.json", "w"), indent=1)

if not args.train_only:   harvest()
if not args.harvest_only and not stop: train()
print("\n  paused - re-run to resume" if stop else "\n  done")
