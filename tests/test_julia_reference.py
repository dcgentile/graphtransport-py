"""Numeric cross-checks against GraphTransportation.jl.

Reference values were produced by tests/julia_reference/sinkhorn_reference.jl
run against the Julia package (commit c4f28a2, Julia 1.12.6):

    julia --project=/path/to/GraphTransportation.jl tests/julia_reference/sinkhorn_reference.jl

on the 3x3 grid with three probability vectors peaked at nodes 0, 2, 7
(Julia: 1, 3, 8). Agreement is expected to floating-point round-off, since
the Python code performs the same operations in the same order.
"""

import numpy as np
import pytest
from julia_values import (
    JULIA_MU,
    JULIA_DIFFUSION_COST_ROW0,
    JULIA_BARYCENTER_EPS01_ITERS256,
    JULIA_BARYCENTER2_EPS005_ITERS64,
    JULIA_LOSS_GRADIENT_ALPHA_ITERS40,
    JULIA_BARYCENTRIC_LOSS_ALPHA_ITERS40,
    JULIA_W_LAMBDA_ITERS40,
    JULIA_SIMPLEX_REGRESSION_LAMBDA,
    JULIA_API_BARY_NU,
    JULIA_API_BARY_J,
    JULIA_API_GEO_W2,
    JULIA_API_GEO_RHO_MID,
    JULIA_API_TRANSPORT_COST,
    JULIA_API_ANALYSIS_LAMBDA,
)

from graphtransport import MarkovGraph, markov_chain_from_edge_list
from graphtransport.sinkhorn import (
    barycentric_loss,
    bfs_hops,
    ground_cost,
    loss_gradient,
    simplex_regression,
    sinkhorn_barycenter,
    sinkhorn_differentiate,
)


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


def test_reference_measures_match_julia():
    np.testing.assert_allclose(_mu(_grid3()), JULIA_MU, rtol=1e-14)


def test_diffusion_ground_cost_matches_julia():
    # Julia uses the plain walk; this package defaults to the lazy walk (see ground_cost).
    C = ground_cost(_grid3(), "diffusion", laziness=0.0)
    np.testing.assert_allclose(C[0], JULIA_DIFFUSION_COST_ROW0, rtol=1e-12, atol=1e-15)


def test_sinkhorn_barycenter_matches_julia():
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    mu = _mu(G)
    p = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=256)
    np.testing.assert_allclose(p, JULIA_BARYCENTER_EPS01_ITERS256, rtol=1e-12)
    p2 = sinkhorn_barycenter([0.25, 0.75], mu[:, :2], cost, 0.05, iters=64)
    np.testing.assert_allclose(p2, JULIA_BARYCENTER2_EPS005_ITERS64, rtol=1e-12)


def test_backward_pass_matches_julia():
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    mu = _mu(G)
    q = np.array(JULIA_BARYCENTER_EPS01_ITERS256)
    alpha = np.array([0.2, -0.1, 0.3])
    assert barycentric_loss(alpha, mu, q, cost, 0.1, iters=40) == pytest.approx(
        JULIA_BARYCENTRIC_LOSS_ALPHA_ITERS40, rel=1e-12
    )
    np.testing.assert_allclose(loss_gradient(alpha, mu, q, cost, 0.1, iters=40), JULIA_LOSS_GRADIENT_ALPHA_ITERS40, rtol=1e-10)
    _, w = sinkhorn_differentiate([0.5, 0.3, 0.2], mu, q, cost, 0.1, 40)
    np.testing.assert_allclose(w, JULIA_W_LAMBDA_ITERS40, atol=1e-14)


def test_unified_api_matches_julia():
    from graphtransport import analysis, barycenter, geodesic, transport_cost

    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    mu = _mu(G)
    refs = [mu[:, s] / G.pi for s in range(3)]

    nu, J, _ = barycenter(G, refs, [0.5, 0.3, 0.2], method="sinkhorn", cost=cost, epsilon=0.1, iters=256)
    np.testing.assert_allclose(nu, JULIA_API_BARY_NU, rtol=1e-12)
    assert J == pytest.approx(JULIA_API_BARY_J, rel=1e-12)

    sol = geodesic(G, refs[0], refs[1], method="sinkhorn", cost=cost, epsilon=0.1, N=4, iters=256)
    assert sol.rho.shape == (9, 5)
    assert sol.W2 == pytest.approx(JULIA_API_GEO_W2, rel=1e-12)
    np.testing.assert_allclose(sol.rho[:, 2], JULIA_API_GEO_RHO_MID, rtol=1e-12)
    assert transport_cost(G, refs[0], refs[1], cost=cost, epsilon=0.1, N=4, iters=256) == pytest.approx(
        JULIA_API_TRANSPORT_COST, rel=1e-12
    )

    lam_hat = analysis(G, nu, refs, method="sinkhorn", cost=cost, epsilon=0.1, iters=256)
    np.testing.assert_allclose(lam_hat, JULIA_API_ANALYSIS_LAMBDA, atol=1e-5)


def test_simplex_regression_matches_julia_to_optimizer_tolerance():
    # Optim.jl and scipy take different L-BFGS paths, so agreement is to the
    # optimizers' tolerance, not round-off.
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    mu = _mu(G)
    lam_hat = simplex_regression(mu, np.array(JULIA_BARYCENTER_EPS01_ITERS256), cost, 0.1, iters=256)
    np.testing.assert_allclose(lam_hat, JULIA_SIMPLEX_REGRESSION_LAMBDA, atol=1e-5)
