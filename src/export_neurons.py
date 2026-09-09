#!/usr/bin/env python3
"""
Snapshot EVERY NEURON of both networks at every input position.

The previous export gave one node per token position, which showed the sequence
but not the network. This gives the actual units:

  Network A   embedding      128 neurons
              per layer x4   128 residual + 512 MLP neurons  (+ 4 attention heads)
              output          96 vocab neurons
              -------------------------------------------------
              total         2,784 neurons

  Network B   1,024 sparse features (its "neurons")

Every value is a real activation measured at that input position.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from nn.gpt import GPT, Config
from nn.sae import SAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NTOK, LAYER = 32, 2

meta = json.load(open(f"{ROOT}/data/meta.json"))
stoi, vocab = meta["stoi"], meta["vocab"]
itos = {i: c for c, i in stoi.items()}
cfg = Config(vocab_size=len(vocab))

ck = torch.load(f"{ROOT}/checkpoints/a_latest.pt", map_location=DEV)
A = GPT(cfg).to(DEV).eval(); A.load_state_dict(ck["model"]); a_step = ck["step"]

text = open(f"{ROOT}/data/corpus.txt").read()
i = text.find("    def __init__(self")
probe = text[i:i + NTOK].replace("\n", " ")
ids = torch.tensor([[stoi[c] for c in probe]], device=DEV)

# capture the embedding layer too
with torch.no_grad():
    pos = torch.arange(len(probe), device=DEV)
    emb = (A.tok(ids) + A.pos(pos))[0].cpu().numpy()
    logits, _, acts = A(ids, record=True)
probs = F.softmax(logits[0], dim=-1).cpu().numpy()

r3 = lambda a: [round(float(v), 3) for v in a]

net = {
  "embed": [r3(emb[t]) for t in range(len(probe))],                     # (T, 128)
  "layers": [], "out": [r3(probs[t]) for t in range(len(probe))],       # (T, 96)
}
for L in range(cfg.n_layer):
    res = acts["resid"][L][0].cpu().numpy()          # (T, 128)
    mlp = acts["mlp"][L][0].cpu().numpy()            # (T, 512)
    att = acts["attn"][L][0].cpu().numpy()           # (H, T, T)
    net["layers"].append({
        "resid": [r3(res[t]) for t in range(res.shape[0])],
        "mlp":   [r3(mlp[t]) for t in range(mlp.shape[0])],
        "attn":  [[r3(row) for row in att[h]] for h in range(cfg.n_head)],
        "attn_entropy": [round(float(-(p*np.log(p+1e-9)).sum(-1).mean()), 3) for p in att],
        # per-head output magnitude at each position: how much this head CONTRIBUTED
        "head_out": [[round(float(np.abs(att[h, t]).max()), 3) for t in range(att.shape[1])]
                     for h in range(cfg.n_head)],
    })

# ---- Network B's neurons -------------------------------------------------
b = {"info": None, "act": None, "features": [], "layer": LAYER}
bp = f"{ROOT}/checkpoints/b_latest.pt"
if os.path.exists(bp):
    bck = torch.load(bp, map_location=DEV)
    B = SAE(cfg.n_embd).to(DEV); B.load_state_dict(bck["model"])
    r = acts["resid"][LAYER].reshape(-1, cfg.n_embd)
    with torch.no_grad():
        fmat = B.encode(r / B.scale).cpu().numpy()                       # (T, 1024)
    # sparse by nature - store only what fires
    b["act"] = [[[int(k), round(float(fmat[t, k]), 3)]
                 for k in np.nonzero(fmat[t] > 1e-4)[0]] for t in range(fmat.shape[0])]
    b["info"] = {"step": bck["step"], "d_hidden": int(B.d_hidden),
                 "n_params": int(sum(p.numel() for p in B.parameters()))}
    fj = f"{ROOT}/findings/features.json"
    if os.path.exists(fj):
        b["features"] = json.load(open(fj))["features"][:24]

def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
sa, sb = rd(f"{ROOT}/status/a.json"), rd(f"{ROOT}/status/b.json")

n_a = (cfg.n_embd + cfg.n_layer*(cfg.n_embd + 4*cfg.n_embd) + cfg.vocab_size)
out = {
  "captured": time.time(),
  "cfg": {**{k: getattr(cfg, k) for k in
             ["n_layer","n_head","n_embd","vocab_size","block_size"]},
          "d_mlp": 4*cfg.n_embd, "params": A.n_params(), "neurons": n_a},
  "status": {"a_step": a_step, "a_loss": sa.get("loss"), "a_bpc": sa.get("bpc"),
             "a_ips": sa.get("steps_per_sec"), "a_alive": sa.get("alive", False),
             "a_history": [round(v,4) for v in (sa.get("history") or [])][-60:],
             "b_step": sb.get("step"), "b_fvu": sb.get("fvu"), "b_l0": sb.get("l0"),
             "b_dead": sb.get("dead"), "b_alive": sb.get("alive", False),
             "b_history": [round(v,5) for v in (sb.get("history") or [])][-60:],
             "device": DEV, "corpus": meta["n_chars"]},
  "tokens": list(probe), "vocab": vocab,
  "a": net, "b": b,
}
path = f"{ROOT}/viz/neurons.json"
json.dump(out, open(path, "w"), separators=(",", ":"))
print(f"  wrote viz/neurons.json  ({os.path.getsize(path)/1024:.0f} KB)")
print(f"  A: {n_a:,} neurons across {cfg.n_layer} layers   @ step {a_step:,}")
print(f"  B: {b['info']['d_hidden']:,} feature neurons      @ step {b['info']['step']:,}" if b["info"] else "  B: none")
print(f"  probe: {probe!r}")
