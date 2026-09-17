import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graphtransport import MarkovGraph, markov_chain_from_edge_list  # noqa: E402
from graphtransport.sinkhorn import bfs_hops, ground_cost, sinkhorn_barycenter, sinkhorn_differentiate  # noqa: E402
from graphtransport.sinkhorn.backends import torch_backend  # noqa: E402
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
    p_torch = torch_backend.sinkhorn_barycenter(lam, torch.tensor(mu), cost, 0.1, iters=256)
    assert p_torch.dtype == torch.float64
    np.testing.assert_allclose(p_torch.numpy(), sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256), rtol=1e-12)
    np.testing.assert_allclose(p_torch.numpy(), JULIA_BARYCENTER_EPS01_ITERS256, rtol=1e-12)


def test_autograd_matches_hand_derived_backward_pass(problem):
    # Independent check of Step 5's reverse-mode gradient: autograd through the
    # same finite-L forward must reproduce Algorithm 1's w = grad_lambda E_L.
    cost, mu = problem
    rng = np.random.default_rng(5)
    q = rng.uniform(0.2, 1.0, size=9)
    q /= q.sum()
    lam = np.array([0.5, 0.3, 0.2])
    for L in (6, 60):
        coords = torch.tensor(lam, requires_grad=True)
        p = torch_backend.sinkhorn_barycenter(coords, torch.tensor(mu), cost, 0.1, iters=L)
        loss = 0.5 * torch.sum((p - torch.tensor(q)) ** 2)
        loss.backward()
        _, w = sinkhorn_differentiate(lam, mu, q, cost, 0.1, L)
        np.testing.assert_allclose(coords.grad.numpy(), w, rtol=1e-10, atol=1e-14)


def test_gradcheck_in_coords_and_measures(problem):
    cost, mu = problem
    coords = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64, requires_grad=True)
    measures = torch.tensor(mu, requires_grad=True)
    cost_t = torch.tensor(cost)
    assert torch.autograd.gradcheck(
        lambda c, m: torch_backend.sinkhorn_barycenter(c, m, cost_t, 0.1, iters=12),
        (coords, measures),
    )


def test_float32_inputs_run_in_float32(problem):
    cost, mu = problem
    p = torch_backend.sinkhorn_barycenter([0.5, 0.3, 0.2], torch.tensor(mu, dtype=torch.float32), cost, 0.1, iters=64)
    assert p.dtype == torch.float32
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-4)
