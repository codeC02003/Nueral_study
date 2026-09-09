"""
DEMO (not a lesson - you don't need to understand this code yet, just the result)

Claim: a network will cram MORE features into a layer than that layer has
neurons, by storing them as overlapping directions. If true, "read neuron #3
and say what it means" is a broken plan, and Network B needs a smarter strategy.

Setup (Elhage et al., "Toy Models of Superposition", Anthropic 2022):
  - 20 independent features in the world
  - a bottleneck with only 5 neurons
  - features are SPARSE: any given input has ~1 of the 20 active
  - task: squeeze 20 -> 5 -> reconstruct the 20

Information theory says you cannot store 20 independent numbers in 5.
But sparsity changes the game: if features rarely co-occur, collisions
rarely happen, so it's worth packing them anyway.
"""
import torch

torch.manual_seed(0)
N_FEATURES, N_HIDDEN, SPARSITY = 20, 5, 0.05

W = torch.nn.Parameter(torch.randn(N_HIDDEN, N_FEATURES) * 0.1)
b = torch.nn.Parameter(torch.zeros(N_FEATURES))
opt = torch.optim.Adam([W, b], lr=1e-2)

for step in range(8000):
    x = torch.rand(2048, N_FEATURES)                       # feature magnitudes
    x = x * (torch.rand(2048, N_FEATURES) < SPARSITY)      # ...but mostly zero
    h = x @ W.T                                            # 20 -> 5  (squeeze)
    xh = torch.relu(h @ W + b)                             # 5 -> 20  (recover)
    loss = ((x - xh) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

print(f"final reconstruction loss: {loss.item():.6f}\n")

norms = W.norm(dim=0)                       # how strongly each feature is stored
represented = (norms > 0.5).sum().item()
print("HOW MANY OF THE 20 FEATURES DID IT ACTUALLY LEARN TO STORE?")
print("  per-feature weight norms:", " ".join(f"{n:.2f}" for n in norms))
print(f"  features stored : {represented}")
print(f"  neurons available: {N_HIDDEN}")
print(f"  -> {represented} features living in {N_HIDDEN} dimensions.\n")

print("ARE THE STORED DIRECTIONS ORTHOGONAL (i.e. non-interfering)?")
G = (W.T @ W).detach()
off = G - torch.diag(torch.diag(G))
print(f"  largest off-diagonal overlap: {off.abs().max():.3f}  (0.00 = clean, >0 = interference)")
print("  -> they overlap. The model accepts noise to fit more in.\n")

print("IS ANY SINGLE NEURON INTERPRETABLE ON ITS OWN?")
for h_i in range(N_HIDDEN):
    fires_for = (W[h_i].abs() > 0.3).nonzero().flatten().tolist()
    print(f"  neuron {h_i}: responds to features {fires_for}")
print("""
  -> Each neuron answers to SEVERAL unrelated features. That is polysemanticity.
     Asking 'what does neuron 3 mean?' has no clean answer. The meaningful
     objects are DIRECTIONS in the 5-d space, and there are more of them
     than there are axes.""")
