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
    pi is stationary for Q (in fact Q is reversible) because W is symmetric,
    so W must be square, finite, nonnegative and symmetric, and every node
    needs positive total weight. Returns (Q, pi).
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] == 0:
        raise ValueError(f"W must be a nonempty square matrix, got shape {W.shape}")
    if not np.all(np.isfinite(W)) or W.min() < 0:
        raise ValueError("W must be finite and nonnegative")
    if not np.allclose(W, W.T, rtol=1e-12, atol=1e-12 * np.abs(W).max()):
        raise ValueError("W must be symmetric; otherwise pi = row sums / total is not stationary for Q")
    d = W.sum(axis=1)
    if np.any(d == 0):
        raise ValueError(f"node(s) {np.flatnonzero(d == 0).tolist()} have no edges; Q is undefined there")
    Q = W / d[:, np.newaxis]
    pi = d / d.sum()
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
