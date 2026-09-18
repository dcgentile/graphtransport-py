import numpy as np
import pytest

from graphtransport import MarkovGraph, markov_chain_from_edge_list
from graphtransport.sinkhorn import (
    barycentric_loss,
    bfs_hops,
    ground_cost,
    loss_gradient,
    regularize_cost,
    simplex_regression,
    sinkhorn_barycenter,
    sinkhorn_differentiate,
    sqeuc_loss,
)
from graphtransport.sinkhorn.core import _sinkhorn_forward


def _grid(n):
    edges = []
    for i in range(n * n):
        if (i + 1) % n != 0:
            edges.append((i, i + 1))
        if i + n < n * n:
            edges.append((i, i + n))
    return MarkovGraph(*markov_chain_from_edge_list(edges))


@pytest.fixture(scope="module")
def grid3():
    G = _grid(3)
    cost = ground_cost(G, "shortest_path")
    hops = bfs_hops(G)
    mu = np.column_stack([np.exp(-2 * hops[k]) / np.exp(-2 * hops[k]).sum() for k in (0, 2, 7)])
    return G, cost, mu


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    M = rng.uniform(0.2, 1.0, size=(9, 3))
    M /= M.sum(axis=0)
    q = rng.uniform(0.2, 1.0, size=9)
    q /= q.sum()
    return M, q


def test_differentiate_without_target_matches_barycenter(grid3):
    _, cost, mu = grid3
    p, w = sinkhorn_differentiate([0.5, 0.3, 0.2], mu, None, cost, 0.1, 64)
    assert w is None
    np.testing.assert_allclose(p, sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=64))


@pytest.mark.parametrize("L", [6, 60])
def test_w_matches_finite_differences_in_lambda(grid3, L):
    # Direct check of Algorithm 1's w = grad_lambda E_L against central finite
    # differences of the same finite-L objective (the Julia suite's test).
    _, cost, _ = grid3
    M, q = _random_problem(3)
    K = regularize_cost(cost, 0.1)
    lam = np.array([0.5, 0.3, 0.2])
    h = 1e-6

    def E(lv):
        p, _, _ = _sinkhorn_forward(lv, M, K, L)
        return sqeuc_loss(p, q)

    fd = np.array([(E(lam + h * e) - E(lam - h * e)) / (2 * h) for e in np.eye(3)])
    _, w = sinkhorn_differentiate(lam, M, q, cost, 0.1, L)
    np.testing.assert_allclose(w, fd, rtol=1e-4, atol=1e-6)


def test_loss_gradient_matches_finite_differences_through_softmax(grid3):
    _, cost, _ = grid3
    M, q = _random_problem(2)
    alpha = np.array([0.2, -0.1, 0.3])
    h = 1e-6

    def E(a):
        return barycentric_loss(a, M, q, cost, 0.1, iters=40)

    fd = np.array([(E(alpha + h * e) - E(alpha - h * e)) / (2 * h) for e in np.eye(3)])
    analytic = loss_gradient(alpha, M, q, cost, 0.1, iters=40)
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)
    # tangent to the simplex after the softmax Jacobian
    assert abs(analytic.sum()) < 1e-10


def test_simplex_regression_recovers_synthesis_weights(grid3):
    _, cost, mu = grid3
    lam_true = np.array([0.5, 0.3, 0.2])
    target = sinkhorn_barycenter(lam_true, mu, cost, 0.1, iters=256)
    lam_hat = simplex_regression(mu, target, cost, 0.1, iters=256)
    assert lam_hat.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(lam_hat > 0)
    np.testing.assert_allclose(lam_hat, lam_true, atol=1e-5)


def test_simplex_regression_random_problem_recovers_weights(grid3):
    _, cost, _ = grid3
    rng = np.random.default_rng(1)
    M = rng.uniform(0.2, 1.0, size=(9, 3))
    M /= M.sum(axis=0)
    lam_true = np.array([0.6, 0.1, 0.3])
    target = sinkhorn_barycenter(lam_true, M, cost, 0.1, iters=256)
    lam_hat = simplex_regression(M, target, cost, 0.1, iters=256)
    np.testing.assert_allclose(lam_hat, lam_true, atol=2e-2)


def test_regression_runs_one_forward_pass_per_evaluation(grid3, monkeypatch):
    from graphtransport.sinkhorn import core

    _, cost, mu = grid3
    target = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=64)
    calls = []
    forward = core._sinkhorn_forward
    monkeypatch.setattr(core, "_sinkhorn_forward", lambda *a: calls.append(1) or forward(*a))
    evaluations = []
    real_minimize = core.minimize

    def counting_minimize(fun, *args, **kwargs):
        result = real_minimize(fun, *args, **kwargs)
        evaluations.append(result.nfev)
        return result

    monkeypatch.setattr(core, "minimize", counting_minimize)
    simplex_regression(mu, target, cost, 0.1, iters=64)
    assert len(calls) == evaluations[0]


@pytest.mark.filterwarnings("error")
def test_regression_warns_only_when_the_budget_runs_out(grid3):
    _, cost, mu = grid3
    target = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=64)
    simplex_regression(mu, target, cost, 0.1, iters=64)  # converges: no warning
    with pytest.warns(UserWarning, match="iteration/evaluation limit"):
        simplex_regression(mu, target, cost, 0.1, iters=64, maxiter=2)


def test_target_shape_is_checked(grid3):
    _, cost, mu = grid3
    for bad in (0.1, np.ones(1), np.ones((9, 1))):
        with pytest.raises(ValueError, match="target must be"):
            sinkhorn_differentiate([0.5, 0.3, 0.2], mu, bad, cost, 0.1, 16)
    with pytest.raises(ValueError, match="needs a target"):
        loss_gradient(np.zeros(3), mu, None, cost, 0.1)
