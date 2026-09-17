import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from graphtransport import ArithmeticMean, HarmonicMean, MarkovGraph, QuadLogMean, grid_markov_chain, triangle_markov_chain  # noqa: E402
from graphtransport.graph import graph_gradient  # noqa: E402
from graphtransport.socp import analyze_socp, barycenter_socp, geodesic_socp  # noqa: E402

REFS = [np.array([2.0, 0.5, 0.5]), np.array([0.5, 2.0, 0.5]), np.array([0.5, 0.5, 2.0])]
LAM = np.array([0.5, 0.3, 0.2])


def _triangle(mean=None):
    Q, pi = triangle_markov_chain()
    return MarkovGraph(Q, pi, mean=mean)


def _grad_scale(G, phi):
    return np.abs(graph_gradient(G, phi)).max()


def test_two_reference_barycenter_lies_on_the_geodesic():
    # Bary({nu0, nu1}, (1-t, t)) equals the geodesic point at t = k/N: the two
    # SOCPs solve the same joint problem, so they agree to solver tolerance.
    Q, pi = grid_markov_chain(3)
    G = MarkovGraph(Q, pi)
    rng = np.random.default_rng(1)
    nu0 = rng.random(G.n) + 0.1
    nu0 /= nu0 @ pi
    nu1 = rng.random(G.n) + 0.1
    nu1 /= nu1 @ pi
    N = 10
    sol = geodesic_socp(G, nu0, nu1, N=N)
    for k in (2, 5, 8):
        t = k / N
        nu_bary, _, _ = barycenter_socp(G, [nu0, nu1], [1 - t, t], N=N)
        assert np.abs(nu_bary - sol.rho[:, k]).max() < 3e-3


def test_symmetric_references_give_uniform_barycenter():
    G = _triangle()
    nu, J, geos = barycenter_socp(G, REFS, np.full(3, 1 / 3), N=10)
    np.testing.assert_allclose(nu, 1.0, atol=1e-5)
    assert all(g.status == cp.OPTIMAL for g in geos)
    W2s = [g.W2 for g in geos]
    assert W2s[0] == pytest.approx(W2s[1], abs=1e-5) and W2s[1] == pytest.approx(W2s[2], abs=1e-5)
    assert J == pytest.approx(sum(W2s) / 3, abs=1e-6)


def test_zero_weight_drops_that_reference():
    _, _, geos = barycenter_socp(_triangle(), REFS, [0.5, 0.5, 0.0], N=10)
    assert len(geos) == 2


def test_invalid_weights_raise():
    G = _triangle()
    with pytest.raises(ValueError):
        barycenter_socp(G, REFS, [0.5, 0.5], N=4)
    with pytest.raises(ValueError):
        barycenter_socp(G, REFS, [0.6, 0.6, -0.2], N=4)
    with pytest.raises(ValueError):
        barycenter_socp(G, REFS, [0.0, 0.0, 0.0], N=4)


def test_endpoint_potentials_and_kkt_stationarity():
    # Per-block potentials (lam_i divided out) must match an independent
    # geodesic_socp solve, and stationarity in nu is sum_i lam_i phi1_i = const.
    G = _triangle()
    nu, _, geos = barycenter_socp(G, REFS, LAM, N=3)
    assert np.all(nu > 1e-3)  # fully supported: no slack term in stationarity
    for ref, geo in zip(REFS, geos):
        indep = geodesic_socp(G, ref, nu, N=3)
        for a, b in ((geo.phi1, indep.phi1), (geo.phi0, indep.phi0)):
            np.testing.assert_allclose(graph_gradient(G, a), graph_gradient(G, b), atol=1e-3 * _grad_scale(G, b))
    stationarity = sum(l * geo.phi1 for l, geo in zip(LAM, geos))
    assert _grad_scale(G, stationarity) < 1e-5 * _grad_scale(G, geos[0].phi1)


def test_analyze_socp_recovers_barycentric_coordinates():
    G = _triangle()
    nu, _, _ = barycenter_socp(G, REFS, LAM, N=10)
    lam_pot = analyze_socp(G, nu, REFS, N=10)
    lam_mom = analyze_socp(G, nu, REFS, N=10, convention="momentum")
    np.testing.assert_allclose(lam_pot, LAM, atol=1e-3)  # exact stationarity, solver tolerance
    np.testing.assert_allclose(lam_mom, LAM, atol=1e-2)  # O(h) proxy

    # recovery does not depend on N being fine: at N=2 the true lam is already
    # a numerical zero of the Gram form
    nu2, _, _ = barycenter_socp(G, REFS, LAM, N=2)
    lam2, A2 = analyze_socp(G, nu2, REFS, N=2, return_system=True)
    np.testing.assert_allclose(lam2, LAM, atol=1e-3)
    assert LAM @ A2 @ LAM <= 1e-6 * np.diag(A2).max()
    assert lam2 @ A2 @ lam2 <= 1e-6 * np.diag(A2).max()


def test_analyze_socp_rejects_unknown_convention():
    with pytest.raises(ValueError, match="convention"):
        analyze_socp(_triangle(), REFS[0], REFS, N=2, convention="dual")


@pytest.mark.parametrize("theta", [ArithmeticMean(), HarmonicMean(), QuadLogMean(8)], ids=repr)
def test_barycenter_and_analysis_round_trip_per_mean(theta):
    G = _triangle(theta)
    nu, J, geos = barycenter_socp(G, REFS, LAM, N=6)
    assert abs(nu @ G.pi - 1) < 1e-8 and nu.min() >= -1e-8
    assert J == pytest.approx(sum(l * geodesic_socp(G, r, nu, N=6).W2 for l, r in zip(LAM, REFS)), rel=1e-4)
    np.testing.assert_allclose(analyze_socp(G, nu, REFS, N=6), LAM, atol=2e-3)
    stationarity = sum(l * geo.phi1 for l, geo in zip(LAM, geos))
    assert _grad_scale(G, stationarity) < 1e-5 * _grad_scale(G, geos[0].phi1)


def test_analysing_with_a_different_mean_is_a_convention_mismatch():
    nu_h, _, _ = barycenter_socp(_triangle(HarmonicMean()), REFS, LAM, N=6)
    lam = analyze_socp(_triangle(ArithmeticMean()), nu_h, REFS, N=6)
    assert np.linalg.norm(lam - LAM) > 1e-3
