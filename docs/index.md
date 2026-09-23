# graphtransport

Optimal transport on graphs, in Python: geodesics, distances, barycenters and
barycentric coordinates for densities on the nodes of a graph, under the
discrete transport metric of Maas and of Erbar, Rumpf, Schmitzer & Simon. A
port of [GraphTransportation.jl](https://github.com/dcgentile/GraphTransportation.jl),
cross-checked against it.

<figure markdown>
![A geodesic from a blob to a ring on a 16×16 grid](assets/hero-light.gif#only-light){ width="300" }
![A geodesic from a blob to a ring on a 16×16 grid](assets/hero-dark.gif#only-dark){ width="300" }
<figcaption>A geodesic on the 16×16 grid graph, from a blob to a ring, computed by
shooting (display smoothed between nodes).</figcaption>
</figure>

## Install

```
pip install torch --index-url https://download.pytorch.org/whl/cpu   # the CPU build; plain pip fetches CUDA
pip install "graphtransport[socp]"                                    # [socp] adds the SOCP method
```

## Quick start

```python
import numpy as np
import graphtransport as gt

G = gt.MarkovGraph(*gt.grid_markov_chain(5))       # a 5x5 grid and its random walk
rng = np.random.default_rng(0)
a, b = rng.uniform(0.5, 1.5, G.n), rng.uniform(0.5, 1.5, G.n)
a, b = a / (a @ G.pi), b / (b @ G.pi)                # densities with respect to G.pi

sol = gt.geodesic(G, a, b)                          # sol.W2, sol.rho: (n, 151) path
W = gt.transport_cost(G, a, b)                      # the distance itself
nu, J, info = gt.barycenter(G, [a, b], [0.3, 0.7])  # the weighted barycenter
lam = gt.analysis(G, nu, [a, b])                    # recovers [0.3, 0.7]
```

Densities are taken **with respect to `G.pi`**, the chain's stationary
distribution: `rho @ G.pi == 1`, and the probability vector a density stands for
is `rho * G.pi`.

## The API

| function | returns |
|---|---|
| `geodesic(G, rhoA, rhoB)` | the transport path: `W2`, the densities `rho`, momenta `m`, endpoint potentials |
| `transport_cost(G, rhoA, rhoB)` | the distance \(W(\rho_A, \rho_B)\) |
| `barycenter(G, refs, lam)` | the minimiser of \(\sum_i \lambda_i W^2(\rho_i, \nu)\) |
| `analysis(G, target, refs)` | the weights that make `target` a barycenter of `refs` |

Each takes `method=`: `"shooting"` (the default), `"socp"` or `"sinkhorn"`; see
[choosing a method](methods.md). `geodesic` and `transport_cost` also take torch
tensors and are [differentiable](differentiable.md).
