"""Numeric cross-checks against GraphTransportation.jl.

Reference values were produced by tests/julia_reference/sinkhorn_reference.jl
run against the Julia package (commit c4f28a2, Julia 1.12.6):

    julia --project=/path/to/GraphTransportation.jl tests/julia_reference/sinkhorn_reference.jl

on the 3x3 grid with three probability vectors peaked at nodes 0, 2, 7
(Julia: 1, 3, 8). Agreement is expected to floating-point round-off, since
the Python code performs the same operations in the same order.
"""

import numpy as np

from graphtransport import MarkovGraph, markov_chain_from_edge_list
from graphtransport.sinkhorn import bfs_hops, ground_cost, sinkhorn_barycenter


def _grid3():
    edges = []
    for i in range(9):
        if (i + 1) % 3 != 0:
            edges.append((i, i + 1))
        if i + 3 < 9:
            edges.append((i, i + 3))
    return MarkovGraph(*markov_chain_from_edge_list(edges))


def _mu(G):
    hops = bfs_hops(G)
    return np.column_stack([np.exp(-2 * hops[k]) / np.exp(-2 * hops[k]).sum() for k in (0, 2, 7)])


# fmt: off
JULIA_MU = np.array([
    [0.75136535287504713, 0.10168624284552195, 0.013761736476765693, 0.10168624284552195, 0.013761736476765693, 0.0018624485039107097, 0.013761736476765693, 0.0018624485039107097, 0.00025205499579036145],
    [0.013761736476765698, 0.101686242845522, 0.75136535287504747, 0.0018624485039107103, 0.013761736476765698, 0.101686242845522, 0.00025205499579036156, 0.0018624485039107103, 0.013761736476765698],
    [0.0016909303564897184, 0.012494379263487331, 0.0016909303564897184, 0.012494379263487331, 0.092321669299223724, 0.012494379263487331, 0.092321669299223724, 0.68216999359888753, 0.092321669299223724],
]).T

JULIA_DIFFUSION_COST_ROW0 = [0, 0.99999999999999978, 0.035928143712574828, 1, 0.017964071856287425, 1, 0.035928143712574828, 1, 0.07185628742514967]

# sinkhorn_barycenter([0.5, 0.3, 0.2], mu, nothing, cost, 0.1; iters=256)
JULIA_BARYCENTER_EPS01_ITERS256 = [0.16577498872441102, 0.32317635185395649, 0.11654446071004262, 0.10867800661110708, 0.14833796820437148, 0.060153300709033894, 0.031385420987137563, 0.029973076648858912, 0.015976425551081152]

# sinkhorn_barycenter([0.25, 0.75], mu[:, 1:2], nothing, cost, 0.05; iters=64)
JULIA_BARYCENTER2_EPS005_ITERS64 = [0.067265174704165875, 0.3724671819419369, 0.36798469030184033, 0.014795654493803391, 0.061165282218208147, 0.093067643163897887, 0.0020509646972534695, 0.0083996969016206721, 0.012803711577273613]
# fmt: on


def test_reference_measures_match_julia():
    np.testing.assert_allclose(_mu(_grid3()), JULIA_MU, rtol=1e-14)


def test_diffusion_ground_cost_matches_julia():
    C = ground_cost(_grid3(), "diffusion")
    np.testing.assert_allclose(C[0], JULIA_DIFFUSION_COST_ROW0, rtol=1e-12, atol=1e-15)


def test_sinkhorn_barycenter_matches_julia():
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    mu = _mu(G)
    p = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=256)
    np.testing.assert_allclose(p, JULIA_BARYCENTER_EPS01_ITERS256, rtol=1e-12)
    p2 = sinkhorn_barycenter([0.25, 0.75], mu[:, :2], cost, 0.05, iters=64)
    np.testing.assert_allclose(p2, JULIA_BARYCENTER2_EPS005_ITERS64, rtol=1e-12)
