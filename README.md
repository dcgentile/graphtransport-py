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
  flow. Exact in time, needs no conic solver, works with every admissible
  mean, including the exact logarithmic mean, and is differentiable (see
  below). It requires **strictly positive** densities, and gets stiff as they
  approach zero: on a 5x5 grid, a row of nodes at 1e-4 solves and one at 1e-5
  does not. A long transport through thin densities can also need more than
  the default 50 Newton iterations (`maxiters=`): corner bumps over a floor
  of 1e-3 on a 10x10 grid take 112, in the Julia package as here. Given data
  that is zero somewhere, or that shooting fails on, it warns
  (`ShootingFallbackWarning`) and falls back to the SOCP at its default N=10
  -- if cvxpy is installed; otherwise it raises. Pass `fallback=False` to
  always get the error instead.
- **`method="socp"`** -- a second-order-cone program. Handles densities
  supported on part of the graph, and its barycenter is a global optimum,
  which makes it the certificate for the others. Time-discretisation error
  O(1/N). Needs `graphtransport[socp]`.
- **`method="sinkhorn"`** -- entropic regularisation for a ground cost. A
  different object from the other two: it needs a `cost` and an `epsilon`,
  and it blurs.

The default diverges from the Julia package, which defaults to `:socp`.
What shooting offers is exactness in time, no conic solver, and gradients --
not speed. Seconds per geodesic on n x n grids (one torch thread, best of
two runs, after cvxpy's first-call overhead), for near-uniform densities and
for Gaussian bumps in opposite corners:

| nodes | near-uniform: shooting | SOCP N=10 | SOCP N=40 | corner bumps: shooting | SOCP N=10 | SOCP N=40 |
|---|---|---|---|---|---|---|
| 9 | 0.34 | 0.01 | 0.03 | 1.1 | 0.01 | 0.03 |
| 25 | 0.37 | 0.04 | 0.49 | 1.5 | 0.04 | 0.51 |
| 64 | 0.56 | 0.30 | 1.5 | 2.9 | 0.38 | 2.4 |
| 100 | 0.77 | 0.51 | 3.0 | 5.2 | 0.72 | 3.4 |
| 144 | 1.5 | 0.84 | 4.6 | 12 | 1.3 | 6.4 |
| 256 | 6.4 | 1.6 | 8.6 | 71 | 1.9 | 9.7 |

- The SOCP at its default N=10 was faster than shooting on every problem.
  Its W2 agreed with shooting's to within 2.2e-4 relative on the
  near-uniform densities, and to 3e-4 to 4e-3 on the bumps.
- Against the SOCP at N=40, shooting was faster on near-uniform densities
  from 25 nodes up, and slower on the bumps at every size.
- More N is not always more accurate: from 64 nodes the SOCP's
  near-uniform W2 was further from shooting's at N=40 than at N=10 (at 256
  nodes, 3.8e-4 against 8.5e-6), the conic solver's tolerance rather than
  the time step setting the error.
- Shooting's cost depends on the data, not just the graph: on 256 nodes,
  6.4 s for near-uniform densities and 71 s for the corner bumps.

For large graphs, long transports, or data near the boundary, pass
`method="socp"`. Multiple shooting, which splits a long transport into
short segments, is planned to cut shooting's cost there.

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
values. They are first order only: differentiating a gradient again
(`create_graph=True`, for a gradient penalty or a Hessian-vector product)
raises. Where `rhoA == rhoB`, `transport_cost`'s gradient is 0, the
subgradient at its minimum (as for `torch.linalg.norm`). The SOCP and Sinkhorn
methods are not differentiable here: with inputs that require grad they
raise, and so does shooting where it would otherwise fall back to the SOCP.
`barycenter` and `analysis` do not take tensors yet.

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
0.8 s to 1.5 s and a three-reference barycenter from 17 s to 33 s. torch's
default thread pool adds CPU time but no speed at these sizes, and the
solver warns once
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
