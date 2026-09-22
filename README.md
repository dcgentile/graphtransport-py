# graphtransport

Python port of [GraphTransportation.jl](../GraphTransportation.jl), aimed at being
usable directly in machine learning pipelines (e.g. as a differentiable layer or
graph/dataset utility), not just a 1:1 translation of the Julia API.

## Status

Phases 1-5 of the [porting plan](PORTING_PLAN.md) are implemented: admissible means, `MarkovGraph`, the
predefined graphs, the Sinkhorn/entropic-OT core with its hand-derived
gradient and optional PyTorch and JAX autograd backends, the SOCP geodesic,
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

Optional extras: `pip install "graphtransport[socp]"` for cvxpy and Clarabel;
`"[torch]"` / `"[jax]"` for the autograd-native Sinkhorn backends. The
default method needs none of them.

There is a `graphtransport-py-draft` sibling directory containing an earlier,
unreviewed first attempt at a full port. It is kept only as reference material
and is not part of this package's history or design.

## Development

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
