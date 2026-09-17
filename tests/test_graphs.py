import numpy as np
import pytest

from graphtransport import MarkovGraph
from graphtransport.graphs import (
    cube_markov_chain,
    double_t_markov_chain,
    grid_markov_chain,
    hypercube_markov_chain,
    square_markov_chain,
    t_markov_chain,
    triangle_markov_chain,
    triangle_with_tail_markov_chain,
    triangular_prism_markov_chain,
    weighted_hypercube_markov_chain,
    wheel_markov_chain,
)

CASES = [
    (triangle_markov_chain, 3, 3),
    (triangle_with_tail_markov_chain, 4, 4),
    (square_markov_chain, 4, 4),
    (t_markov_chain, 4, 3),
    (double_t_markov_chain, 8, 10),
    (triangular_prism_markov_chain, 6, 9),
    (cube_markov_chain, 8, 12),
    (hypercube_markov_chain, 16, 32),
    (wheel_markov_chain, 9, 12),
    (lambda: grid_markov_chain(3), 9, 12),
    (lambda: grid_markov_chain(4), 16, 24),
]


@pytest.mark.parametrize("make, n, n_edges", CASES, ids=[c[0].__name__ if hasattr(c[0], "__name__") else "grid" for c in CASES])
def test_node_and_edge_counts_and_valid_chain(make, n, n_edges):
    Q, pi = make()
    assert Q.shape == (n, n)
    assert (Q > 0).sum() == 2 * n_edges
    np.testing.assert_allclose(Q.sum(axis=1), 1.0)
    np.testing.assert_allclose(Q.T @ pi, pi, atol=1e-12)
    G = MarkovGraph(Q, pi)  # reversibility check inside
    assert G.E.shape == (n_edges, 2)


def test_regular_graphs_have_uniform_pi():
    for make, degree in ((triangle_markov_chain, 2), (square_markov_chain, 2), (cube_markov_chain, 3), (hypercube_markov_chain, 4)):
        Q, pi = make()
        np.testing.assert_allclose(pi, 1 / len(pi))
        np.testing.assert_allclose((Q > 0).sum(axis=1), degree)


def test_wheel_hub_degree():
    Q, _ = wheel_markov_chain()
    assert (Q[8] > 0).sum() == 4
    assert set(np.flatnonzero(Q[8])) == {1, 3, 5, 7}


def test_grid3_adjacency_hand_checked():
    Q, pi = grid_markov_chain(3)
    neighbours = {i: set(np.flatnonzero(Q[i])) for i in range(9)}
    assert neighbours[0] == {1, 3}
    assert neighbours[4] == {1, 3, 5, 7}
    assert neighbours[8] == {5, 7}
    # pi proportional to degree: corners 2, edges 3, centre 4, total 24
    np.testing.assert_allclose(pi[[0, 2, 6, 8]], 2 / 24)
    np.testing.assert_allclose(pi[[1, 3, 5, 7]], 3 / 24)
    assert pi[4] == pytest.approx(4 / 24)


def test_weighted_hypercube_is_nonuniform_and_seedable():
    Q1, pi1 = weighted_hypercube_markov_chain(rng=0)
    Q2, pi2 = weighted_hypercube_markov_chain(rng=0)
    np.testing.assert_allclose(Q1, Q2)
    assert not np.allclose(pi1, 1 / 16)
    assert (Q1 > 0).sum() == 64
    MarkovGraph(Q1, pi1)
    _, pi3 = weighted_hypercube_markov_chain(rng=1)
    assert not np.allclose(pi1, pi3)
