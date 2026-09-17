import numpy as np
import pytest

from graphtransport import MarkovGraph, markov_chain_from_edge_list
from graphtransport.sinkhorn import (
    bfs_hops,
    build_geodesic,
    ground_cost,
    logarithmic_change_of_variable,
    regularize_cost,
    sinkhorn_barycenter,
    sinkhorn_plan,
)


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


def test_regularize_cost_is_gibbs_kernel(grid3):
    _, cost, _ = grid3
    K = regularize_cost(cost, 0.1)
    np.testing.assert_allclose(K, np.exp(-cost / 0.1))
    assert np.all(K > 0)


def test_logarithmic_change_of_variable_is_softmax():
    lam = logarithmic_change_of_variable([0.0, 0.0, 0.0])
    np.testing.assert_allclose(lam, [1 / 3] * 3)
    lam = logarithmic_change_of_variable([2.0, -1.0, 0.5])
    assert lam.sum() == pytest.approx(1.0)
    assert np.all(lam > 0)
    assert np.argmax(lam) == 0


def test_barycenter_is_probability_vector(grid3):
    _, cost, mu = grid3
    p = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=256)
    assert p.shape == (9,)
    assert np.all(p > 0)
    assert p.sum() == pytest.approx(1.0, abs=1e-8)


def test_barycenter_is_permutation_equivariant(grid3):
    _, cost, mu = grid3
    lam = [0.5, 0.3, 0.2]
    perm = np.random.default_rng(0).permutation(9)
    p = sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256)
    p_perm = sinkhorn_barycenter(lam, mu[perm], cost[np.ix_(perm, perm)], 0.1, iters=256)
    np.testing.assert_allclose(p_perm, p[perm], rtol=1e-10)


def test_barycenter_with_pure_weight_ignores_other_measures(grid3):
    _, cost, mu = grid3
    p_a = sinkhorn_barycenter([1.0, 0.0], mu[:, [0, 1]], cost, 0.1, iters=256)
    p_b = sinkhorn_barycenter([1.0, 0.0], mu[:, [0, 2]], cost, 0.1, iters=256)
    np.testing.assert_allclose(p_a, p_b, rtol=1e-10)


def test_barycenter_of_identical_measures_converges_to_that_measure(grid3):
    # The entropic barycenter of a measure with itself is that measure blurred
    # by the kernel; the blur vanishes as epsilon -> 0.
    _, cost, mu = grid3
    v = mu[:, 0]
    errors = [
        np.abs(sinkhorn_barycenter([0.5, 0.5], np.column_stack([v, v]), cost, eps, iters=256) - v).sum()
        for eps in (0.1, 0.03, 0.01, 0.005)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3


def test_marginal_errors_of_barycenter_plans_are_small(grid3):
    _, cost, mu = grid3
    eps, iters = 0.1, 256
    p = sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, eps, iters=iters)
    K = regularize_cost(cost, eps)
    for s in range(3):
        P = sinkhorn_plan(K, mu[:, s], p, iters=iters)
        assert np.all(P >= 0)
        assert np.abs(P.sum(axis=1) - mu[:, s]).sum() < 1e-6
        assert np.abs(P.sum(axis=0) - p).sum() < 1e-6


def test_sinkhorn_plan_structure(grid3):
    _, cost, mu = grid3
    K = regularize_cost(cost, 0.1)
    P = sinkhorn_plan(K, mu[:, 0], mu[:, 1], iters=256)
    # P = diag(u) K diag(v): P / K has rank one
    ratio = P / K
    assert np.linalg.matrix_rank(ratio, tol=1e-8) == 1
    assert P.sum() == pytest.approx(1.0, abs=1e-8)


def test_build_geodesic_endpoints_and_shape(grid3):
    _, cost, mu = grid3
    path = build_geodesic(mu[:, :2], cost, epsilon=0.1, steps=4, iters=256)
    assert path.shape == (9, 5)
    np.testing.assert_allclose(path[:, 0], sinkhorn_barycenter([1.0, 0.0], mu[:, :2], cost, 0.1, iters=256))
    np.testing.assert_allclose(path[:, -1], sinkhorn_barycenter([0.0, 1.0], mu[:, :2], cost, 0.1, iters=256))
    np.testing.assert_allclose(path.sum(axis=0), 1.0, atol=1e-8)


def test_build_geodesic_rejects_wrong_measure_count(grid3):
    _, cost, mu = grid3
    with pytest.raises(ValueError):
        build_geodesic(mu, cost)
