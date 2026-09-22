# graphtransport

Python port of [GraphTransportation.jl](../GraphTransportation.jl), aimed at being
usable directly in machine learning pipelines (e.g. as a differentiable layer or
graph/dataset utility), not just a 1:1 translation of the Julia API.

## Status

Phases 1-5 of the [porting plan](PORTING_PLAN.md) are implemented: admissible means, `MarkovGraph`, the
predefined graphs, the Sinkhorn/entropic-OT core with its hand-derived
gradient and PyTorch and JAX autograd backends, the SOCP geodesic,
barycenter and analysis (cvxpy + Clarabel), geodesics and barycenters by
shooting on the Hamiltonian flow, and the unified
`geodesic`/`transport_cost`/`barycenter`/`analysis` API over all three
methods. Phase 6 (Chambolle-Pock) is not ported; the plan recommends skipping
it. Each step landed as its own
reviewed, tested PR -- nothing was ported in bulk -- and the core numerics
are cross-checked against values produced by the Julia package itself
(`tests/julia_reference/`).

### Choosing a method

- **`method="shooting"` (default)** -- Newton shooting on the Hamiltonian
  flow. Exact in time, needs no optional dependency, and works with every
  admissible mean, including the exact logarithmic mean. It requires
  **strictly positive** densities. Given data that is zero somewhere, or
  close enough to zero that shooting fails, it warns
  (`ShootingFallbackWarning`) and falls back to the SOCP at its default
  N=10 -- if cvxpy is installed; otherwise it raises. Pass `fallback=False`
  to always get the error instead.
- **`method="socp"`** -- a second-order-cone program. Handles densities
  supported on part of the graph, and its barycenter is a global optimum,
  which makes it the certificate for the others. Time-discretisation error
  O(1/N). Needs `graphtransport[socp]`.
- **`method="sinkhorn"`** -- entropic regularisation for a ground cost. A
  different object from the other two: it needs a `cost` and an `epsilon`,
  and it blurs.

The default diverges from the Julia package, which defaults to `:socp`.
Shooting wins at matched accuracy -- 150 RK4 steps agree with the SOCP at
N=40 to 4-5 digits, at about a third of the cost -- but it is not faster
than the SOCP at its default N=10 on graphs above roughly 64 nodes, since
each Newton step integrates n trajectories. For large graphs, or data near
the boundary, pass `method="socp"`.

### Differentiable geodesics

`geodesic` and `transport_cost` accept torch tensors and return tensors. With
the default `method="shooting"`, gradients flow back to both endpoints
through W2, the density path, the momenta and the potentials:

```python
import torch
import graphtransport as gt

G = gt.MarkovGraph(*gt.grid_markov_chain(5))
pi = torch.tensor(G.pi)
logits = torch.zeros(G.n, requires_grad=True)
rhoA = torch.softmax(logits, 0) / pi          # a density w.r.t. pi
rhoB = torch.linspace(0.5, 1.5, G.n, dtype=torch.float64)
rhoB = rhoB / (rhoB @ pi)
gt.transport_cost(G, rhoA, rhoB).backward()   # logits.grad is d W / d logits
```

The gradients are exact for the discrete problem the solver solves, obtained
by the implicit function theorem rather than by differentiating through
Newton's iterations, so they agree with finite differences of the returned
values. The SOCP and Sinkhorn methods are not differentiable here: with inputs
that require grad they raise, and so does shooting where it would otherwise
fall back to the SOCP. `barycenter` and `analysis` do not take tensors yet.

### Installing

PyTorch is a required dependency: the shooting method differentiates the
Hamiltonian flow with it, so its Newton Jacobian is exact (as Julia's, which
uses ForwardDiff, is). A plain `pip install torch` on Linux fetches the CUDA
build, several GB; for the CPU build, install torch first from PyTorch's CPU
index:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install graphtransport
```

Optional extras: `pip install "graphtransport[socp]"` for cvxpy and Clarabel,
and `"[jax]"` for the JAX Sinkhorn backend. The default method needs neither.
(`"[torch]"` still resolves, and installs nothing extra.)

The exact Jacobian costs time: on a 5x5 grid, a shooting geodesic went from
0.8 s to 1.5 s and a three-reference barycenter from 17 s to 33 s. (The speed
comparison under "Choosing a method" predates it.) torch's default thread
pool adds CPU time but no speed at these sizes, and the solver warns once
(`TorchThreadsWarning`) when torch uses more than one thread;
`torch.set_num_threads(1)` avoids the cost.

There is a `graphtransport-py-draft` sibling directory containing an earlier,
unreviewed first attempt at a full port. It is kept only as reference material
and is not part of this package's history or design.

## Development

```
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest              # the slow Julia cross-checks are deselected; pytest -m "" runs them too
```
