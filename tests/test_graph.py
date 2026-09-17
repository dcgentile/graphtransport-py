import numpy as np
import pytest

from graphtransport.chains import markov_chain_from_edge_list
from graphtransport.graph import MarkovGraph, graph_divergence, graph_gradient, metric_tensor
from graphtransport.means import ArithmeticMean, GeometricMean, HarmonicMean

geomean = GeometricMean()


def _triangle():
    Q, pi = markov_chain_from_edge_list([(0, 1), (1, 2), (0, 2)])
    return MarkovGraph(Q, pi)


def test_construction_orders_edges_and_computes_kappa():
    G = _triangle()
    assert G.n == 3
    assert G.E.shape == (3, 2)
    # every edge oriented with x < y
    assert np.all(G.E[:, 0] < G.E[:, 1])
    # kappa[e] = Q[x,y] * pi[x] should match reversibility: also == Q[y,x]*pi[y]
    for e, (x, y) in enumerate(G.E):
        assert G.kappa[e] == pytest.approx(G.Q[x, y] * G.pi[x])
        assert G.kappa[e] == pytest.approx(G.Q[y, x] * G.pi[y])


def test_rejects_non_reversible_chain():
    Q = np.array([[0.0, 1.0], [0.5, 0.5]])
    pi = np.array([0.5, 0.5])
    # Q[0,1]*pi[0] = 0.5, Q[1,0]*pi[1] = 0.25 -- not reversible
    with pytest.raises(ValueError):
        MarkovGraph(Q, pi)


def test_graph_gradient_antisymmetric_convention():
    G = _triangle()
    phi = np.array([1.0, 3.0, 7.0])
    grad = graph_gradient(G, phi)
    for e, (x, y) in enumerate(G.E):
        assert grad[e] == pytest.approx(phi[x] - phi[y])


def test_divergence_is_adjoint_of_gradient():
    # <phi, div m>_pi == -<grad phi, m>_Q, with <m, w>_Q := sum_e kappa[e] m[e] w[e]
    G = _triangle()
    rng = np.random.default_rng(0)
    phi = rng.uniform(-1, 1, size=G.n)
    m = rng.uniform(-1, 1, size=G.E.shape[0])

    lhs = np.dot(phi * G.pi, graph_divergence(G, m))
    rhs = -np.dot(G.kappa * graph_gradient(G, phi), m)
    assert lhs == pytest.approx(rhs)


def test_metric_tensor_matches_pointwise_geomean():
    G = _triangle()
    rho = np.array([1.0, 4.0, 9.0])
    theta = metric_tensor(G, rho)
    for e, (x, y) in enumerate(G.E):
        assert theta[e] == pytest.approx(geomean(rho[x], rho[y]))


def test_metric_tensor_accepts_custom_mean():
    G = _triangle()
    rho = np.array([1.0, 4.0, 9.0])
    theta = metric_tensor(G, rho, mean=lambda a, b: (a + b) / 2)
    for e, (x, y) in enumerate(G.E):
        assert theta[e] == pytest.approx((rho[x] + rho[y]) / 2)


def test_default_mean_is_geometric():
    G = _triangle()
    assert G.mean == GeometricMean()
    assert "GeometricMean" in repr(G)


def test_mean_keyword_is_stored_and_used_by_metric_tensor():
    Q, pi = markov_chain_from_edge_list([(0, 1), (1, 2), (0, 2)])
    G = MarkovGraph(Q, pi, mean=HarmonicMean())
    rho = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(metric_tensor(G, rho), HarmonicMean()(rho[G.E[:, 0]], rho[G.E[:, 1]]))
    # explicit override still wins
    np.testing.assert_allclose(metric_tensor(G, rho, mean=geomean), geomean(rho[G.E[:, 0]], rho[G.E[:, 1]]))


def test_with_mean_shares_cached_matrices():
    G = _triangle()
    H = G.with_mean(ArithmeticMean())
    assert H.mean == ArithmeticMean()
    assert G.mean == GeometricMean()  # original untouched
    assert H.Q is G.Q and H.D is G.D and H.kappa is G.kappa and H.E is G.E and H.pi is G.pi
    rho = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(metric_tensor(H, rho), (rho[H.E[:, 0]] + rho[H.E[:, 1]]) / 2)
