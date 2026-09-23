import numpy as np
import pytest

from graphtransport import ArithmeticMean, HarmonicMean, MarkovGraph, QuadLogMean, grid_markov_chain
from graphtransport.gram import gram_matrix, potential_gram_qp, simplex_qp, solve_barycentric_coordinates_qp
from graphtransport.graph import graph_gradient, metric_tensor
from graphtransport.solvers import cvxpy_available

# The scipy method needs nothing optional, so these tests run everywhere; the
# cvxpy method runs when the socp extra is installed.
METHODS = [
    pytest.param("cvxpy", marks=pytest.mark.skipif(not cvxpy_available(), reason="needs graphtransport[socp]")),
    "scipy",
]


@pytest.mark.parametrize("method", METHODS)
def test_simplex_qp_diagonal_has_closed_form(method):
    a = np.array([1.0, 2.0, 4.0])
    lam = simplex_qp(np.diag(a), method=method)
    expected = (1 / a) / (1 / a).sum()
    np.testing.assert_allclose(lam, expected, atol=1e-6)
    assert lam.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(lam >= -1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_simplex_qp_vertex_solution(method):
    # one reference dominates: A = B^T B with B making column 0 tiny
    B = np.array([[0.01, 1.0, 1.0], [0.0, 1.0, -1.0]])
    lam = simplex_qp(B.T @ B, method=method)
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


@pytest.mark.parametrize("method", METHODS)
def test_potential_gram_qp_recovers_weights_of_a_stationary_point(method):
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
    lam = potential_gram_qp(G, target, [phi1, phi2, phi3], method=method)
    np.testing.assert_allclose(lam, lam_true, atol=1e-5)


def _planted(seed=0):
    """A Gram matrix whose simplex minimizer is exactly (0.5, 0.3, 0.2), value 0."""
    rng = np.random.default_rng(seed)
    lam = np.array([0.5, 0.3, 0.2])
    V = rng.normal(size=(3, 12))
    V[2] = -(lam[0] * V[0] + lam[1] * V[1]) / lam[2]
    return V @ V.T, lam


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [1e-16, 1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8])
def test_simplex_qp_is_scale_invariant(method, scale):
    # Before normalizing A, cvxpy returned an error of 1e-2 at scale 1e-8 and
    # uniform weights at 1e-12, reporting "optimal" both times.
    A, lam_true = _planted()
    np.testing.assert_allclose(simplex_qp(scale * A, method=method), lam_true, atol=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_simplex_qp_output_is_exactly_on_the_simplex(method):
    rng = np.random.default_rng(4)
    B = rng.normal(size=(6, 4))
    lam = simplex_qp(B @ B.T, method=method)  # rank 4 of 6: some weights are exactly 0
    assert np.all(lam >= 0)
    assert lam.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.skipif(not cvxpy_available(), reason="needs graphtransport[socp]")
def test_methods_agree_with_active_constraints():
    rng = np.random.default_rng(5)
    for _ in range(5):
        B = rng.normal(size=(6, 4))
        A = B @ B.T
        np.testing.assert_allclose(simplex_qp(A, method="scipy"), simplex_qp(A, method="cvxpy"), atol=1e-6)


@pytest.mark.parametrize(
    "A, message",
    [
        ([[1.0, 3.0], [3.0, 1.0]], "not positive semidefinite"),  # used to return [0.5, 0.5]
        ([[1.0, np.nan], [np.nan, 1.0]], "non-finite"),
        (np.ones((2, 3)), "square"),
        (np.zeros((0, 0)), "square"),
    ],
)
def test_simplex_qp_rejects_invalid_matrices(A, message):
    with pytest.raises(ValueError, match=message):
        simplex_qp(A, method="scipy")


def test_simplex_qp_trivial_cases():
    np.testing.assert_array_equal(simplex_qp(np.zeros((3, 3))), np.full(3, 1 / 3))
    np.testing.assert_array_equal(simplex_qp([[2.0]]), [1.0])
    with pytest.raises(ValueError, match="method"):
        simplex_qp(np.eye(2), method="gurobi")


@pytest.mark.filterwarnings("error")
def test_compute_condition_on_a_singular_matrix(capsys):
    # singular: the target is exactly a barycenter of the references
    vecs = np.random.default_rng(0).normal(size=(3, 12))
    vecs[2] = -(0.5 * vecs[0] + 0.3 * vecs[1]) / 0.2
    solve_barycentric_coordinates_qp(list(vecs), np.ones(12), compute_condition=True, method="scipy")
    assert "numerically singular: rank 2 of 3" in capsys.readouterr().out
    solve_barycentric_coordinates_qp([np.zeros(4)] * 2, np.ones(4), compute_condition=True, method="scipy")
    assert "zero" in capsys.readouterr().out


def test_missing_cvxpy_gives_the_install_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cvxpy", None)  # makes `import cvxpy` raise ImportError
    with pytest.raises(ImportError, match=r"graphtransport\[socp\]"):
        simplex_qp(np.eye(2), method="cvxpy")
    np.testing.assert_allclose(simplex_qp(np.diag([1.0, 3.0])), [0.75, 0.25], atol=1e-9)  # auto falls back to scipy
