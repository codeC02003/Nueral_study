"""Export a Value graph + a frame-by-frame recording of its backward sweep."""
import json


def export(root, path):
    from nn.engine import trace
    nodes, edges = trace(root)

    topo, seen = [], set()
    def build(v):
        if v in seen: return
        seen.add(v)
        for c in v._prev: build(c)
        topo.append(v)
    build(root)
    order = list(reversed(topo))                     # the backward sweep order

    ids = {id(v): i for i, v in enumerate(topo)}

    # depth = longest path from a leaf, so the graph lays out left(inputs)->right(loss)
    depth = {}
    for v in topo:
        depth[id(v)] = 0 if not v._prev else 1 + max(depth[id(c)] for c in v._prev)

    node_json = [{
        "id": ids[id(v)],
        "label": v.label or v._op or f"{v.data:g}",
        "op": v._op,
        "data": v.data,
        "isLeaf": not v._prev,
        "isRoot": v is root,
        "depth": depth[id(v)],
    } for v in topo]

    # Record grads after each backward step -> the animation frames.
    for v in topo: v.grad = 0.0
    frames = [{"active": None, "grads": [v.grad for v in topo],
               "note": "All gradients start at 0. Nothing is known to affect L yet."}]
    root.grad = 1.0
    frames.append({"active": ids[id(root)], "grads": [v.grad for v in topo],
                   "note": "SEED: dL/dL = 1. The only fact the sweep needs to begin."})
    for v in order:
        v._backward()
        kids = ", ".join(c.label or c._op or f"{c.data:g}" for c in v._prev) or "nobody"
        frames.append({
            "active": ids[id(v)],
            "grads": [n.grad for n in topo],
            "note": f"'{v.label or v._op}' pushes its gradient down to: {kids}",
        })

    out = {"nodes": node_json,
           "edges": [{"from": ids[id(a)], "to": ids[id(b)]} for a, b in edges],
           "frames": frames}
    with open(path, "w") as f:
        json.dump(out, f)
    return out
