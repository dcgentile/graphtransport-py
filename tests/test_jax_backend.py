import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from graphtransport import MarkovGraph, markov_chain_from_edge_list  # noqa: E402
from graphtransport.sinkhorn import bfs_hops, ground_cost, sinkhorn_barycenter, sinkhorn_differentiate  # noqa: E402
from graphtransport.sinkhorn.backends import jax_backend  # noqa: E402
from test_julia_reference import JULIA_BARYCENTER_EPS01_ITERS256  # noqa: E402


def _grid3():
    edges = []
    for i in range(9):
        if (i + 1) % 3 != 0:
            edges.append((i, i + 1))
        if i + 3 < 9:
            edges.append((i, i + 3))
    return MarkovGraph(*markov_chain_from_edge_list(edges))


@pytest.fixture(scope="module")
def problem():
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    hops = bfs_hops(G)
    mu = np.column_stack([np.exp(-2 * hops[k]) / np.exp(-2 * hops[k]).sum() for k in (0, 2, 7)])
    return cost, mu


def test_forward_matches_numpy_and_julia(problem):
    cost, mu = problem
    lam = np.array([0.5, 0.3, 0.2])
    p = jax_backend.sinkhorn_barycenter(lam, jnp.asarray(mu), cost, 0.1, iters=256)
    assert p.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(p), sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(p), JULIA_BARYCENTER_EPS01_ITERS256, rtol=1e-12)


def test_grad_matches_hand_derived_backward_pass(problem):
    cost, mu = problem
    rng = np.random.default_rng(5)
    q = rng.uniform(0.2, 1.0, size=9)
    q /= q.sum()
    lam = np.array([0.5, 0.3, 0.2])
    for L in (6, 60):

        def loss(c):
            p = jax_backend.sinkhorn_barycenter(c, jnp.asarray(mu), cost, 0.1, iters=L)
            return 0.5 * jnp.sum((p - jnp.asarray(q)) ** 2)

        g = jax.grad(loss)(jnp.asarray(lam))
        _, w = sinkhorn_differentiate(lam, mu, q, cost, 0.1, L)
        np.testing.assert_allclose(np.asarray(g), w, rtol=1e-10, atol=1e-14)


def test_check_grads_in_coords_and_measures(problem):
    from jax.test_util import check_grads

    cost, mu = problem
    f = lambda c, m: jax_backend.sinkhorn_barycenter(c, m, cost, 0.1, iters=12)  # noqa: E731
    check_grads(f, (jnp.array([0.5, 0.3, 0.2]), jnp.asarray(mu)), order=1, modes=("rev",))


def test_jit_matches_eager(problem):
    cost, mu = problem
    f = jax.jit(lambda c, m: jax_backend.sinkhorn_barycenter(c, m, cost, 0.1, iters=64))
    lam = jnp.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(
        np.asarray(f(lam, jnp.asarray(mu))),
        np.asarray(jax_backend.sinkhorn_barycenter(lam, jnp.asarray(mu), cost, 0.1, iters=64)),
        rtol=1e-12,
    )


def test_agrees_with_torch_backend(problem):
    torch = pytest.importorskip("torch")
    from graphtransport.sinkhorn.backends import torch_backend

    cost, mu = problem
    lam = [0.5, 0.3, 0.2]
    p_jax = np.asarray(jax_backend.sinkhorn_barycenter(lam, jnp.asarray(mu), cost, 0.1, iters=128))
    p_torch = torch_backend.sinkhorn_barycenter(lam, torch.tensor(mu), cost, 0.1, iters=128).numpy()
    np.testing.assert_allclose(p_jax, p_torch, rtol=1e-12)
