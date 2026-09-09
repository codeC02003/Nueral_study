r"""
NETWORK B (Tier 2) - a sparse autoencoder over A's residual stream.

WHY THIS ARCHITECTURE, specifically.

demo_superposition.py proved A packs more features than it has neurons, as
overlapping directions. So B cannot read neurons. It has to find the DIRECTIONS.

The trick: give B far MORE hidden units than A has dimensions (128 -> 1024),
but force only a handful to fire at once. An overcomplete-but-sparse basis is
exactly the shape superposition compresses FROM, so B is being asked to run
the compression backwards.

B's loss has no labels in it anywhere:

    L = || x - x_hat ||^2   +   lambda * |f|_1
        [ reconstruct A's ]       [ use as few  ]
        [ activation      ]       [ features as ]
                                  [ possible    ]

Calibrating lambda is not optional. Measured on A layer 2:

    lambda    features active per token    reconstruction error
    0.003            113                          2.6%     too dense - no sparsity
    0.01              34                          6.4%     <- chosen
    0.03              10                         17.2%     sparser, lossier
    0.1                1                        106.0%     collapsed

Too low and B just memorises densely, defeating the point. Too high and B
gives up and outputs the mean. The window is narrower than people expect.

That is the whole supervision. Everything B learns about A, it learns from
those two terms fighting each other.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, d_in, expansion=8, l1=3e-3):
        super().__init__()
        self.d_in, self.d_hidden, self.l1 = d_in, d_in * expansion, l1
        # A's residual stream GROWS across layers (norm 1.3 at layer 0, 3.1 at
        # layer 3). Without normalising, one l1 value means a different level of
        # sparsity at every layer, and results stop being comparable.
        self.register_buffer("scale", torch.tensor(1.0))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, self.d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = nn.Parameter(torch.empty(self.d_hidden, d_in))
        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.normalize_decoder()

    @torch.no_grad()
    def calibrate(self, x):
        self.scale.fill_((x.norm(dim=-1).mean() / math.sqrt(self.d_in)).item())

    @torch.no_grad()
    def normalize_decoder(self):
        # Unit-norm decoder columns. Without this the model cheats the L1
        # penalty by shrinking f and growing W_dec to compensate.
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        x = x / self.scale
        f = self.encode(x)
        xh = self.decode(f)
        mse = (xh - x).pow(2).mean(-1).mean()
        sparsity = f.abs().sum(-1).mean()
        return xh, f, mse + self.l1 * sparsity, mse, sparsity

    @torch.no_grad()
    def stats(self, x):
        x = x / self.scale
        f = self.encode(x)
        xh = self.decode(f)
        l0 = (f > 0).float().sum(-1).mean().item()          # features active per token
        var = x.var(0).sum()
        fvu = ((x - xh).pow(2).sum(-1).mean() / var.clamp(min=1e-9)).item()
        return l0, fvu, (f > 0).any(0)                       # which features fired at all
