import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from julia_values import JULIA_BARYCENTER_EPS01_ITERS256  # noqa: E402

from graphtransport import MarkovGraph, markov_chain_from_edge_list  # noqa: E402
from graphtransport.sinkhorn import bfs_hops, ground_cost, sinkhorn_barycenter, sinkhorn_differentiate  # noqa: E402
from graphtransport.sinkhorn.backends import jax_backend  # noqa: E402


@pytest.fixture(autouse=True)
def x64():
    """Run each test in float64 so results compare with numpy to rounding
    error. Scoped to the test, not set globally at import: the flag is
    process-wide and would otherwise leak into every jax test collected after
    this file. `test_float32_default_mode` covers JAX's default."""
    with jax.enable_x64(True):
        yield


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

        def loss(c, L=L):
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


# The backend rejects what the numpy implementation rejects, with the same messages.


def _underflow_problem():
    n = 9
    cost = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) ** 2 / 64.0
    return cost, np.eye(n)[:, [0, 8]]  # two point masses at opposite ends


def test_kernel_underflow_raises_when_eager_and_passes_nan_through_jit():
    cost, mu = _underflow_problem()
    lam = jnp.array([0.5, 0.5])
    assert jnp.isfinite(jax_backend.sinkhorn_barycenter(lam, mu, cost, 0.01)).all()
    with pytest.raises(FloatingPointError, match="larger epsilon"):
        jax_backend.sinkhorn_barycenter(lam, mu, cost, 0.001)
    # a traced value cannot be inspected, so under jit the nan is returned (documented)
    jitted = jax.jit(lambda c: jax_backend.sinkhorn_barycenter(c, mu, cost, 0.001))
    assert jnp.isnan(jitted(lam)).all()


@pytest.mark.parametrize("epsilon", [0.0, -0.5, float("inf"), float("nan")])
def test_rejects_bad_epsilon(problem, epsilon):
    cost, mu = problem
    with pytest.raises(ValueError, match="epsilon"):
        jax_backend.sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, epsilon)


def test_epsilon_may_be_traced(problem):
    cost, mu = problem
    lam = jnp.array([0.5, 0.3, 0.2])
    f = lambda e: jax_backend.sinkhorn_barycenter(lam, mu, cost, e, iters=16)[0]  # noqa: E731
    assert jnp.isfinite(jax.grad(f)(0.1))
    np.testing.assert_allclose(jax.jit(f)(0.1), f(0.1), rtol=1e-12)


@pytest.mark.parametrize("iters", [0, 1])
def test_rejects_an_iteration_budget_that_runs_no_iterations(problem, iters):
    cost, mu = problem
    with pytest.raises(ValueError, match="iters"):
        jax_backend.sinkhorn_barycenter([0.5, 0.3, 0.2], mu, cost, 0.1, iters=iters)


def test_rejects_traced_iters(problem):
    cost, mu = problem
    f = jax.jit(lambda n: jax_backend.sinkhorn_barycenter(jnp.array([0.5, 0.3, 0.2]), mu, cost, 0.1, iters=n))
    with pytest.raises(TypeError, match="Python int"):
        f(8)


def test_rejects_misshapen_inputs(problem):
    cost, mu = problem
    with pytest.raises(ValueError, match=r"shape \(n, S\)"):
        jax_backend.sinkhorn_barycenter([0.5, 0.3, 0.2], mu.T, cost, 0.1)
    with pytest.raises(ValueError, match="one weight per measure"):
        jax_backend.sinkhorn_barycenter([0.5, 0.5], mu, cost, 0.1)


def test_integer_measures_are_promoted(problem):
    cost, _ = problem
    lam = [0.5, 0.3, 0.2]
    diracs = np.eye(9, dtype=int)[:, [0, 4, 8]]
    p = jax_backend.sinkhorn_barycenter(lam, diracs, cost, 0.5, iters=64)
    np.testing.assert_allclose(np.asarray(p), sinkhorn_barycenter(lam, diracs, cost, 0.5, iters=64), rtol=1e-12)


def test_float32_default_mode(problem):
    # What a user gets without opting in to x64: float64 inputs are computed in float32.
    cost, mu = problem
    lam = np.array([0.5, 0.3, 0.2])
    with jax.enable_x64(False):
        p = jax_backend.sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256)
        g = jax.grad(lambda c: jax_backend.sinkhorn_barycenter(c, mu, cost, 0.1, iters=60)[0])(jnp.asarray(lam))
    assert p.dtype == jnp.float32 and g.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(p), sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256), rtol=1e-4)
    assert np.all(np.isfinite(np.asarray(g)))
