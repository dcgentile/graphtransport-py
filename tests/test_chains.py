import numpy as np
import pytest

from graphtransport.chains import (
    markov_chain_from_adjacency_matrix,
    markov_chain_from_edge_list,
    markov_chain_from_weight_matrix,
    stationary_from_transition,
)


def _assert_valid_chain(Q, pi, n):
    assert Q.shape == (n, n)
    assert pi.shape == (n,)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0)
    np.testing.assert_allclose(Q.T @ pi, pi, atol=1e-10)
    assert pi.sum() == pytest.approx(1.0)


def test_weight_matrix_triangle():
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    Q, pi = markov_chain_from_weight_matrix(W)
    _assert_valid_chain(Q, pi, 3)
    np.testing.assert_allclose(pi, [1 / 3, 1 / 3, 1 / 3])


def test_weight_matrix_respects_weights():
    # node 0 has much larger total weight than nodes 1, 2
    W = np.array([[0, 10, 10], [10, 0, 1], [10, 1, 0]], dtype=float)
    Q, pi = markov_chain_from_weight_matrix(W)
    _assert_valid_chain(Q, pi, 3)
    assert pi[0] > pi[1]
    assert pi[1] == pytest.approx(pi[2])


def test_adjacency_matrix_matches_weight_matrix_on_binary_input():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    Q1, pi1 = markov_chain_from_adjacency_matrix(A)
    Q2, pi2 = markov_chain_from_weight_matrix(A)
    np.testing.assert_allclose(Q1, Q2)
    np.testing.assert_allclose(pi1, pi2)


def test_edge_list_star_graph_degree_weighted_stationary():
    # 0-indexed star: center 0 connected to 1, 2, 3
    E = [(0, 1), (0, 2), (0, 3)]
    Q, pi = markov_chain_from_edge_list(E)
    _assert_valid_chain(Q, pi, 4)
    # center has degree 3, leaves have degree 1 -> pi ∝ degree
    assert pi[0] == pytest.approx(0.5)
    np.testing.assert_allclose(pi[1:], [1 / 6, 1 / 6, 1 / 6])


def test_edge_list_rejects_empty():
    with pytest.raises(ValueError):
        markov_chain_from_edge_list([])


def test_stationary_from_transition_recovers_known_pi():
    Q, pi_expected = markov_chain_from_weight_matrix(
        np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=float)
    )
    pi_recovered = stationary_from_transition(Q)
    np.testing.assert_allclose(pi_recovered, pi_expected, atol=1e-8)


@pytest.mark.parametrize(
    "W, message",
    [
        ([[0.0, 1.0, 0.0], [3.0, 0.0, 1.0], [0.0, 1.0, 0.0]], "symmetric"),
        ([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], r"node\(s\) \[2\] have no edges"),
        ([[0.0, -1.0], [-1.0, 0.0]], "nonnegative"),
        ([[0.0, np.nan], [np.nan, 0.0]], "finite"),
        (np.ones((2, 3)), "square"),
    ],
)
def test_weight_matrix_is_validated_without_assert(W, message):
    # These were a bare assert (stripped by python -O, which then returned a
    # non-stationary pi), a row of nan, and a silently accepted negative weight.
    with pytest.raises(ValueError, match=message):
        markov_chain_from_weight_matrix(W)
