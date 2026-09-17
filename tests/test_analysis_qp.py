import numpy as np
import pytest

pytest.importorskip("cvxpy")

from graphtransport import ArithmeticMean, HarmonicMean, MarkovGraph, QuadLogMean, grid_markov_chain  # noqa: E402
from graphtransport.gram import gram_matrix, potential_gram_qp, simplex_qp, solve_barycentric_coordinates_qp  # noqa: E402
from graphtransport.graph import graph_gradient, metric_tensor  # noqa: E402


def test_simplex_qp_diagonal_has_closed_form():
    a = np.array([1.0, 2.0, 4.0])
    lam = simplex_qp(np.diag(a))
    expected = (1 / a) / (1 / a).sum()
    np.testing.assert_allclose(lam, expected, atol=1e-6)
    assert lam.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(lam >= -1e-9)


def test_simplex_qp_vertex_solution():
    # one reference dominates: A = B^T B with B making column 0 tiny
    B = np.array([[0.01, 1.0, 1.0], [0.0, 1.0, -1.0]])
    lam = simplex_qp(B.T @ B)
    np.testing.assert_allclose(lam, [1.0, 0.0, 0.0], atol=1e-5)


def test_gram_matrix_matches_direct_sum():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(size=(4, 4)) for _ in range(3)]
    g = rng.uniform(0.5, 2.0, size=(4, 4))
    A = gram_matrix(vecs, g)
    for i in range(3):
        for j in range(3):
            assert A[i, j] == pytest.approx(np.sum(vecs[i] * vecs[j] * g))
    np.testing.assert_allclose(A, A.T)


def test_solve_barycentric_coordinates_qp_return_system(capsys):
    rng = np.random.default_rng(1)
    vecs = [rng.normal(size=12) for _ in range(3)]
    g = rng.uniform(0.5, 2.0, size=12)
    lam, A = solve_barycentric_coordinates_qp(vecs, g, return_system=True, compute_condition=True)
    assert "condition number" in capsys.readouterr().out
    np.testing.assert_allclose(A, gram_matrix(vecs, g))
    assert lam.shape == (3,)
    assert lam.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("mean", [None, ArithmeticMean(), HarmonicMean(), QuadLogMean(6)], ids=str)
def test_potential_gram_is_riemannian_inner_product_at_target(mean):
    # Mirrors the Julia "potential_gram_qp: Gram matrix is the Riemannian inner
    # product at the target" testset: A_ij == sum_e kappa_e theta(target)_e grad(phi_i)_e grad(phi_j)_e.
    Q, pi = grid_markov_chain(3)
    G = MarkovGraph(Q, pi, mean=mean)
    rng = np.random.default_rng(2)
    target = rng.uniform(0.5, 1.5, size=9)
    target /= target @ G.pi
    potentials = [rng.normal(size=9) for _ in range(3)]
    lam, A = potential_gram_qp(G, target, potentials, return_system=True)
    theta = metric_tensor(G, target)
    for i in range(3):
        for j in range(3):
            gi, gj = graph_gradient(G, potentials[i]), graph_gradient(G, potentials[j])
            assert A[i, j] == pytest.approx(np.sum(G.kappa * theta * gi * gj))
    assert lam.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(lam >= -1e-9)


def test_potential_gram_qp_recovers_weights_of_a_stationary_point():
    # If sum_i lam_i phi_i is constant (KKT stationarity of a barycenter), the QP
    # objective lam^T A lam is zero at lam and the QP recovers it.
    Q, pi = grid_markov_chain(3)
    G = MarkovGraph(Q, pi)
    rng = np.random.default_rng(3)
    target = rng.uniform(0.5, 1.5, size=9)
    target /= target @ G.pi
    lam_true = np.array([0.5, 0.3, 0.2])
    phi1, phi2 = rng.normal(size=9), rng.normal(size=9)
    phi3 = -(lam_true[0] * phi1 + lam_true[1] * phi2) / lam_true[2] + 4.0  # makes sum lam_i phi_i constant
    lam = potential_gram_qp(G, target, [phi1, phi2, phi3])
    np.testing.assert_allclose(lam, lam_true, atol=1e-5)
