# graphtransport

[![CI](https://github.com/dcgentile/graphtransport-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dcgentile/graphtransport-py/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-dcgentile.github.io-blue)](https://dcgentile.github.io/graphtransport-py/)

**Optimal transport on graphs.** Geodesics, distances, barycenters and
barycentric coordinates for densities on the nodes of a graph, under the
discrete transport metric of Maas and of Erbar, Rumpf, Schmitzer & Simon.
Differentiable with PyTorch.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.gif">
    <img src="docs/assets/hero-light.gif" width="300" alt="A geodesic from a blob to a ring on a 16x16 grid graph">
  </picture>
  <br>
  <em>A geodesic from a blob to a ring on the 16×16 grid graph (display smoothed between nodes).</em>
</p>

**Documentation:** <https://dcgentile.github.io/graphtransport-py/>

## What it does

- **Geodesics and distances** between two densities: the whole transport path,
  its momenta and potentials, and the distance $W$.
- **Barycenters**: the density minimising $\sum_i \lambda_i W^2(\rho_i, \nu)$ for
  reference densities $\rho_i$ and weights $\lambda_i$.
- **Barycentric analysis**: the weights that make a given density a barycenter
  of the references.
- **Three methods behind one API**: Newton shooting on the geodesic equations
  (exact in time, with an adaptive multiple-shooting mode), a second-order cone
  program (handles densities that are zero on part of the graph), and entropic
  Sinkhorn.
- **Gradients**: `geodesic` and `transport_cost` take torch tensors, with exact
  gradients through the solver.
- **Any reversible Markov chain**: grids, hypercubes and other built-in graphs,
  or your own from an edge list, adjacency or weight matrix, with a choice of
  mean for the metric.

## Install

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "graphtransport[socp] @ git+https://github.com/dcgentile/graphtransport-py"
```

PyTorch is required. A plain `pip install torch` on Linux fetches the CUDA
build, several GB, so install the CPU build first as above. The `[socp]` extra
adds cvxpy and Clarabel for the SOCP method; the default method doesn't need it.

## Quick start

```python
import numpy as np
import graphtransport as gt

G = gt.MarkovGraph(*gt.grid_markov_chain(5))       # a 5x5 grid and its random walk
rng = np.random.default_rng(0)
a, b = rng.uniform(0.5, 1.5, G.n), rng.uniform(0.5, 1.5, G.n)
a, b = a / (a @ G.pi), b / (b @ G.pi)                # densities with respect to G.pi

sol = gt.geodesic(G, a, b)                          # sol.W2, and sol.rho: the (n, 151) path
W = gt.transport_cost(G, a, b)                      # the distance itself
nu, J, info = gt.barycenter(G, [a, b], [0.3, 0.7])  # the weighted barycenter
lam = gt.analysis(G, nu, [a, b])                    # recovers [0.3, 0.7]
```

Densities are taken **with respect to `G.pi`**, the chain's stationary
distribution: `rho @ G.pi == 1`. The probability vector a density stands for is
`rho * G.pi`.

## Pictures

<p align="center">
  <img src="docs/assets/digits.png" width="340" alt="Barycenters of four digit images">
</p>

**Barycenters of digits.** The corners are four 16×16 digit images, each a
density on the pixel grid. Every other panel is their barycenter, for weights
that vary bilinearly across the square. Colour is only a label.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/methods-dark.png">
  <img src="docs/assets/methods-light.png" alt="The same geodesic at time 1/2 by shooting, the SOCP and Sinkhorn">
</picture>

**Three methods, one transport.** The blob-to-ring geodesic at time ½, one
square per node. Shooting and the SOCP compute the same geodesic; Sinkhorn's
entropic interpolation blurs.

More in the [gallery](https://dcgentile.github.io/graphtransport-py/gallery/),
and a walkthrough of the whole API in
[`notebooks/demo.ipynb`](notebooks/demo.ipynb).

## Choosing a method

Every function takes `method=`:

| method | use it for | notes |
|---|---|---|
| `"shooting"` (default) | exact geodesics; gradients | densities must be strictly positive; with data that touches zero it warns and falls back to the SOCP |
| `"socp"` | data that is zero on part of the graph; a certified barycenter | error $O(1/N)$ in time; fastest at its default `N=10`; needs `[socp]` |
| `"sinkhorn"` | a fast, smooth approximation for a ground cost | a different object: needs `cost=` and `epsilon=`, and blurs |

On long transports the default shooting mode switches to multiple shooting by
itself, about twice as fast there. Timings and details are in
[choosing a method](https://dcgentile.github.io/graphtransport-py/methods/).

## Gradients

```python
import torch

pi = torch.tensor(G.pi)
logits = torch.zeros(G.n, dtype=torch.float64, requires_grad=True)
W = gt.transport_cost(G, torch.softmax(logits, 0) / pi, torch.tensor(b))
W.backward()                                        # logits.grad: dW/dlogits
```

The gradients are exact for the discrete problem the solver solves, obtained by
the implicit function theorem rather than by unrolling Newton's iterations. See
[differentiable geodesics](https://dcgentile.github.io/graphtransport-py/differentiable/).

## Relation to GraphTransportation.jl

graphtransport began as a Python port of the Julia package
[GraphTransportation.jl](https://github.com/dcgentile/GraphTransportation.jl),
and its core numerics are cross-checked against values produced by it: the
test suite's `tests/julia_reference/` scripts regenerate them. It is a
standalone package, and it differs where that made it more useful in Python:

- the default method is shooting, where the Julia package defaults to the SOCP;
- shooting has an adaptive multiple-shooting mode;
- `geodesic` and `transport_cost` are differentiable with PyTorch;
- conic solves that end at reduced accuracy are rejected rather than accepted.

Julia's Chambolle–Pock solver is not included.

## Development

```
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,socp]"
pytest                   # about 4 minutes; pytest -m "" adds the slow Julia cross-checks
```

The documentation site builds with `pip install -e ".[docs]" && mkdocs serve`.
Its figures are regenerated by `python docs/figures/make_figures.py`.

## License

MIT; see [LICENSE](LICENSE).
