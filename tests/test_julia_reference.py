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
    JULIA_SOCP,
    JULIA_BSOCP,
    JULIA_LONG_TRANSPORT_LOG_MAP,
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
    assert transport_cost(G, refs[0], refs[1], method="sinkhorn", cost=cost, epsilon=0.1, N=4, iters=256) == pytest.approx(
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


def test_geodesic_socp_matches_julia():
    # tests/julia_reference/socp_reference.jl: geodesic_socp(G, refs[1], refs[2]; N=4)
    # for each conic-representable mean. Agreement is to the solvers' tolerance
    # (Clarabel through JuMP there, through cvxpy here), not round-off; the
    # potentials agree absolutely, not merely up to an additive constant.
    # This is also what pins the edge order to Julia's findnz: m0 is compared
    # entry by entry, so a permuted E would fail here.
    pytest.importorskip("cvxpy")
    from graphtransport import ArithmeticMean, GeometricMean, HarmonicMean, QuadLogMean
    from graphtransport.socp import geodesic_socp

    G = _grid3()
    mu = _mu(G)
    refs = [mu[:, s] / G.pi for s in range(3)]
    means = {
        "GeometricMean": GeometricMean(),
        "ArithmeticMean": ArithmeticMean(),
        "HarmonicMean": HarmonicMean(),
        "QuadLogMean(6)": QuadLogMean(6),
    }
    for name, mean in means.items():
        ref = JULIA_SOCP[name]
        sol = geodesic_socp(G.with_mean(mean), refs[0], refs[1], N=4)
        assert sol.W2 == pytest.approx(ref["W2"], rel=1e-5), name

        def close(got, want, tol, what):
            # tolerances relative to each quantity's own scale: m0 is O(30) and
            # the potentials O(10), so a fixed atol would mean different things
            want = np.asarray(want, dtype=float)
            np.testing.assert_allclose(
                got, want, atol=tol * max(1.0, np.abs(want).max()), err_msg=f"{name} {what}"
            )

        close(sol.rho[:, 2], ref["rho_mid"], 1e-4, "rho_mid")
        close(sol.m0, ref["m0"], 1e-4, "m0")
        close(sol.phi0, ref["phi0"], 1e-3, "phi0")
        close(sol.phi1, ref["phi1"], 1e-3, "phi1")


def test_barycenter_socp_matches_julia():
    # tests/julia_reference/socp_barycenter_reference.jl: barycenter_socp and
    # analyze_socp on the triangle, refs = [2,.5,.5] and its rotations,
    # lam = [0.5, 0.3, 0.2], N = 6, per conic mean. Agreement is to the two
    # solvers' tolerance. lam_hat_momentum has no other test pinning it.
    pytest.importorskip("cvxpy")
    from graphtransport import (
        ArithmeticMean, GeometricMean, HarmonicMean, QuadLogMean, triangle_markov_chain,
    )
    from graphtransport.socp import analyze_socp, barycenter_socp

    G = MarkovGraph(*triangle_markov_chain())
    refs = [np.array([2.0, 0.5, 0.5]), np.array([0.5, 2.0, 0.5]), np.array([0.5, 0.5, 2.0])]
    lam = [0.5, 0.3, 0.2]
    means = {
        "GeometricMean": GeometricMean(),
        "ArithmeticMean": ArithmeticMean(),
        "HarmonicMean": HarmonicMean(),
        "QuadLogMean(8)": QuadLogMean(8),
    }
    for name, mean in means.items():
        ref = JULIA_BSOCP[name]
        Gm = G.with_mean(mean)
        nu, J, geodesics = barycenter_socp(Gm, refs, lam, N=6)
        np.testing.assert_allclose(nu, ref["nu"], atol=1e-3, err_msg=f"{name} nu")
        assert J == pytest.approx(ref["J"], rel=1e-4), name
        np.testing.assert_allclose([g.W2 for g in geodesics], ref["W2s"], atol=1e-3, err_msg=f"{name} W2s")
        np.testing.assert_allclose(
            analyze_socp(Gm, nu, refs, N=6), ref["lam_hat"], atol=1e-3, err_msg=f"{name} lam_hat"
        )
        np.testing.assert_allclose(
            analyze_socp(Gm, nu, refs, N=6, convention="momentum"), ref["lam_hat_momentum"],
            atol=1e-3, err_msg=f"{name} lam_hat_momentum",
        )


# The slow cases take 14-46 s each; they run in the all-extras CI job (-m "").
@pytest.mark.parametrize(
    "case",
    [(10, 9), (12, 8), pytest.param((12, 11), marks=pytest.mark.slow), pytest.param((16, 15), marks=pytest.mark.slow)],
    ids=lambda c: f"{c[0]}x{c[0]}-shift{c[1]}",
)
def test_log_map_matches_julia_on_long_transports(case):
    # Julia's Jacobian is ForwardDiff's. The port's forward-difference Jacobian
    # stalled on every one of these (line search failed at iteration 6 on the
    # 10x10); with an exact Jacobian, Newton takes Julia's steps: the same
    # iteration count, and W2 to rounding.
    from graphtransport import grid_markov_chain
    from graphtransport.shooting import log_map

    k, shift = case
    G = MarkovGraph(*grid_markov_chain(k))
    xy = np.array([(i % k, i // k) for i in range(G.n)], dtype=float)

    def bump(center):
        rho = np.exp(-((xy - center) ** 2).sum(axis=1) / 8) + 0.05
        return rho / (rho @ G.pi)

    W2, iters = JULIA_LONG_TRANSPORT_LOG_MAP[case]
    r = log_map(G, bump((0, 0)), bump((shift, shift)))
    assert r.W2 == pytest.approx(W2, rel=1e-12)
    # Julia's count, give or take the last iteration: the final residual can
    # land either side of tol (CI once stopped the 16x16 at 27 with residual
    # 6.4e-10 < 1e-9, where Julia and local runs take 28), from last-bit
    # differences between machines. A forward-difference Jacobian is not
    # within one: it failed outright.
    assert abs(r.iters - iters) <= 1
