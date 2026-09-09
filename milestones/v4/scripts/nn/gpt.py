"""
NETWORK A - a small character-level transformer, built to be READ.

Every design choice here trades capability for legibility, because A's whole
purpose is to be dissected by B:

  - character level     -> 96 embedding rows, each one a nameable symbol
  - 4 layers, 4 heads   -> 16 attention heads total, few enough to enumerate
  - 128 dims            -> small enough that B can model the whole residual stream
  - returns activations -> B needs the insides, not just the output

The architecture is otherwise a standard pre-LN GPT: exactly what GPT-2 is,
scaled down ~150x.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, vocab_size=96, block_size=128, n_layer=4,
                 n_head=4, n_embd=128, dropout=0.0):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.n_embd, self.dropout = n_layer, n_head, n_embd, dropout


class CausalSelfAttention(nn.Module):
    """Each position looks BACKWARD at earlier positions and copies information."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.hd = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x, keep=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                             # (B, nh, T, T)
        if keep is not None:
            keep["attn"] = att.detach()      # <- what B reads to find circuits

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Attention MOVES information between positions; the MLP PROCESSES it in place."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, keep=None):
        # Residual stream: each block ADDS to a running total rather than
        # replacing it. That is why "the residual stream" is the object B
        # should model - it is the network's shared memory bus.
        x = x + self.attn(self.ln1(x), keep)
        h = self.mlp[1](self.mlp[0](self.ln2(x)))
        if keep is not None:
            keep["mlp"] = h.detach()
        return x + self.mlp[2](h)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight              # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, record=False, intervene=None):
        """
        intervene: {layer_index: fn(x) -> x'} applied to the residual stream
        immediately after that block. This is the WRITE path that makes Tier 4
        possible - `record` reads A's internals, `intervene` edits them, and the
        difference in loss between the two runs is the causal evidence.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)

        acts = {"resid": [], "attn": [], "mlp": []} if record else None
        for li, blk in enumerate(self.blocks):
            keep = {} if record else None
            x = blk(x, keep)
            if intervene is not None and li in intervene:
                x = intervene[li](x)
            if record:
                acts["resid"].append(x.detach())
                acts["attn"].append(keep["attn"])
                acts["mlp"].append(keep["mlp"])

        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, acts

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        for _ in range(n):
            crop = idx[:, -self.cfg.block_size:]
            logits, _, _ = self(crop)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
