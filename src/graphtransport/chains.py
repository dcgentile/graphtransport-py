"""Markov chain constructors, ported from GraphTransportation.jl's core/MarkovChains.jl.

Node indices are 0-indexed throughout this package, unlike the Julia
original's 1-indexed convention -- callers passing edge lists or matrices
should use 0-indexed node ids.
"""

from __future__ import annotations

import numpy as np


def markov_chain_from_weight_matrix(W) -> tuple[np.ndarray, np.ndarray]:
    """Weighted random-walk Markov chain from a non-negative symmetric weight matrix.

    Q[x, y] = W[x, y] / sum(W[x, :]), pi[x] = sum(W[x, :]) / sum(W).
    Returns (Q, pi).
    """
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    Q = W / d[:, np.newaxis]
    pi = d / d.sum()
    assert np.allclose(Q.T @ pi, pi)
    return Q, pi


def markov_chain_from_adjacency_matrix(A) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random-walk Markov chain from a binary adjacency matrix.

    Equivalent to markov_chain_from_weight_matrix(A): for a symmetric 0/1
    matrix the two formulas coincide (both normalize by node degree).
    Returns (Q, pi).
    """
    return markov_chain_from_weight_matrix(A)


def markov_chain_from_edge_list(E) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random-walk Markov chain on the graph defined by an edge list.

    E is a sequence of 0-indexed (i, j) pairs; all edges are unweighted.
    Returns (Q, pi).
    """
    edges = list(E)
    if not edges:
        raise ValueError("edge list must be nonempty")
    n = max(max(i, j) for i, j in edges) + 1
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return markov_chain_from_weight_matrix(A)


def stationary_from_transition(Q) -> np.ndarray:
    """Stationary distribution of a row-stochastic transition matrix Q.

    Solves the overdetermined linear system (Q^T - I) pi = 0, sum(pi) = 1
    by least squares, mirroring the Julia original's `\\` solve.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    A = np.vstack([Q.T - np.eye(n), np.ones((1, n))])
    b = np.concatenate([np.zeros(n), [1.0]])
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    return v / v.sum()
