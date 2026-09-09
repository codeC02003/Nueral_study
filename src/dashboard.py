#!/usr/bin/env python3
"""
Live view of both networks. Read-only - it never touches training.

    python3 dashboard.py           # live, refreshes every 2s
    python3 dashboard.py --once    # print one frame and exit
"""
import os, sys, json, time, glob
ONCE = "--once" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = dict(g="\033[32m", r="\033[31m", d="\033[2m", b="\033[1m", c="\033[36m",
         y="\033[33m", m="\033[35m", w="\033[37m", x="\033[0m")
SPARK = "▁▂▃▄▅▆▇█"

def spark(v, w=52):
    if not v or len(v) < 2: return ""
    v = v[-w:]
    lo, hi = min(v), max(v); rng = (hi - lo) or 1e-9
    return "".join(SPARK[min(7, int((x - lo) / rng * 7.99))] for x in v)

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def age(d):
    if not d: return "never started"
    dt = time.time() - d.get("updated", 0)
    if not d.get("alive"): return f"{C['d']}stopped{C['x']}"
    if dt > 30: return f"{C['r']}stalled {dt:.0f}s{C['x']}"
    return f"{C['g']}live{C['x']}"

def pick_a():
    """Show whichever A is LIVE; fall back to the most recently updated."""
    cands = []
    for p in sorted(glob.glob(f"{ROOT}/status/a*.json")):
        d = load(p)
        if d: cands.append((os.path.basename(p)[:-5], d))
    if not cands: return None, None
    live = [c for c in cands if c[1].get("alive") and time.time()-c[1].get("updated",0) < 60]
    tag, d = (live or sorted(cands, key=lambda c: -c[1].get("updated",0)))[0]
    return tag, d

def eval_progress():
    """The A2 four-layer evaluation writes result files, not a status file."""
    done = sorted(glob.glob(f"{ROOT}/findings/layers_a2/L*.json"))
    if not done and not os.path.exists(f"{ROOT}/logs/a2_eval.log"): return None
    line = ""
    try:
        for l in open(f"{ROOT}/logs/a2_eval.log"):
            l = l.rstrip()
            if l.strip(): line = l.strip()
    except Exception: pass
    return [os.path.basename(f)[:-5] for f in done], line[:96]

while True:
    tag, a = pick_a()
    b = load(f"{ROOT}/status/b.json")
    L = [f"{C['b']}NEURAL ANATOMY{C['x']}  {C['d']}A learns Python · B learns A · "
         f"{time.strftime('%H:%M:%S')}{C['x']}", ""]

    # ---- A --------------------------------------------------------------
    L.append(f"{C['b']}{C['c']}NETWORK {tag.upper() if tag else 'A'}{C['x']}  "
             f"{C['d']}the subject — a 4-layer char transformer on Python source{C['x']}   {age(a)}")
    if a and "loss" in a:
        L += ["",
              f"   step {C['b']}{a['step']:>9,}{C['x']}    loss {C['y']}{a['loss']:.4f}{C['x']}"
              f"    bits/char {C['y']}{a['bpc']:.3f}{C['x']}    {a['steps_per_sec']:.1f} it/s"
              f"    {C['d']}{a['params']:,} params · {a['device']}{C['x']}",
              f"   {C['d']}loss{C['x']} {C['y']}{spark(a.get('history', []))}{C['x']}"]
        smp = sorted(glob.glob(f"{ROOT}/samples/{tag}_step*.py"))
        if smp:
            txt = open(smp[-1]).read().split("\n")
            L += ["", f"   {C['d']}what A can write right now ({os.path.basename(smp[-1])}){C['x']}"]
            L += [f"   {C['d']}│{C['x']} {l[:74]}" for l in txt[:7]]
    else:
        L += ["", f"   {C['d']}no A training — A1 is frozen at step 83,750 (this is intended){C['x']}"]

    # ---- B --------------------------------------------------------------
    L += ["", f"{C['b']}{C['m']}NETWORK B{C['x']}  {C['d']}the anatomist — a sparse autoencoder reading A's residual stream{C['x']}   {age(b)}"]
    if b and "fvu" in b:
        lag = (a['step'] - b['a_step']) if a and 'step' in a else 0
        L += ["",
              f"   step {C['b']}{b['step']:>9,}{C['x']}    recon err {C['m']}{b['fvu']*100:5.1f}%{C['x']}"
              f"    active/token {C['m']}{b['l0']:.1f}{C['x']}/{b['features']}"
              f"    dead {b['dead']}    {C['d']}A layer {b['layer']}{C['x']}",
              f"   {C['d']}reading A @ step {b['a_step']:,}  (A is {lag:,} steps ahead){C['x']}",
              f"   {C['d']}err {C['x']}{C['m']}{spark(b.get('history', []))}{C['x']}"]
        fnd = load(f"{ROOT}/findings/features.json")
        if fnd:
            L += ["", f"   {C['d']}features B discovered — nobody told it what to look for{C['x']}"]
            for it in fnd["features"][:5]:
                ctx = "  ".join(repr(c[-20:]) for c in it["contexts"][:3])
                L.append(f"   {C['d']}│{C['x']} f{it['feature']:<5}{ctx[:68]}")
    else:
        L += ["", f"   {C['d']}SAE training stopped — B's frozen SAEs are in checkpoints/b_*.pt{C['x']}"]

    ep = eval_progress()
    if ep:
        done, line = ep
        L += ["", f"{C['b']}{C['y']}A2 REPLICATION{C['x']}  {C['d']}frozen four-layer evaluation{C['x']}"]
        L += ["", f"   layers complete: {C['g'] if done else C['d']}{', '.join(done) if done else 'none yet'}{C['x']}  of L0, L1, L2, L3"]
        if line: L.append(f"   {C['d']}{line}{C['x']}")
    L += ["", f"{C['d']}   Ctrl-C to close the dashboard. Training keeps running.{C['x']}"]
    if ONCE:
        print("\n".join(L)); break
    sys.stdout.write("\033[H\033[J" + "\n".join(L) + "\n")
    sys.stdout.flush()
    time.sleep(2)
