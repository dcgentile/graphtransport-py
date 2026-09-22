# graphtransport

Python port of [GraphTransportation.jl](../GraphTransportation.jl), aimed at being
usable directly in machine learning pipelines (e.g. as a differentiable layer or
graph/dataset utility), not just a 1:1 translation of the Julia API.

## Status

Phases 1-4 of the [porting plan](PORTING_PLAN.md) are implemented: admissible means, `MarkovGraph`, the
Sinkhorn/entropic-OT core with its hand-derived gradient, optional PyTorch
and JAX autograd backends, the predefined graphs, the SOCP geodesic,
barycenter and analysis (cvxpy + Clarabel), and the unified
`geodesic`/`barycenter`/`analysis` API dispatching over
`method="socp"` (the default, as in Julia) and `method="sinkhorn"`.
Shooting (Phase 5) is next. Each step lands as its own reviewed, tested PR —
nothing is ported in bulk — and the core numerics are cross-checked against
values produced by the Julia package itself (`tests/julia_reference/`).

Optional extras: `pip install "graphtransport[socp]"` for cvxpy and Clarabel,
which the default `method="socp"` needs; `"[torch]"` / `"[jax]"` for the
autograd-native Sinkhorn backends. `method="sinkhorn"` needs neither.

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
