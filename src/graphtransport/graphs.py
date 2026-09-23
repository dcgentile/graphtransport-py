"""Predefined graphs, ported from GraphTransportation.jl's core/CommonGraphs.jl.

Each constructor returns ``(Q, pi)`` for the uniform random walk on the named
graph (weighted, for ``weighted_hypercube_markov_chain``), ready for
``MarkovGraph(Q, pi)``. Node ids are 0-indexed; the Julia originals' edge
lists are reproduced with every index shifted down by one, so node k here
is node k+1 in Julia.

Not ported: ``ma_house_markov_chain`` (needs the bundled Massachusetts
shapefile and a geometry library).
"""

from __future__ import annotations

import numpy as np

from graphtransport.chains import markov_chain_from_edge_list, markov_chain_from_weight_matrix

_HYPERCUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 8), (1, 9), (2, 10), (3, 11),
    (4, 12), (5, 13), (6, 14), (7, 15),
    (8, 9), (9, 10), (10, 11), (11, 8),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (12, 13), (13, 14), (14, 15), (15, 12),
]  # fmt: skip


def triangle_markov_chain():
    """The 3-cycle."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (2, 0)])


def triangle_with_tail_markov_chain():
    """A 3-cycle with one pendant edge (node 3 attached to node 2)."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (2, 0), (2, 3)])


def square_markov_chain():
    """The 4-cycle."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (2, 3), (3, 0)])


def t_markov_chain():
    """The T-shaped graph: path 0-1-2 with a branch 1-3."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (1, 3)])


def double_t_markov_chain():
    """Two T-shaped graphs (nodes 0-3 and 4-7) joined at corresponding
    vertices: 0-4, 1-5, 2-6, 3-7."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (1, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (5, 7)])


def triangular_prism_markov_chain():
    """Two triangular faces (0-1-2 and 3-4-5) joined by three edges."""
    return markov_chain_from_edge_list([(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (5, 3)])


def cube_markov_chain():
    """The 3-cube: 8 nodes, 12 edges."""
    return markov_chain_from_edge_list(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
    )


def hypercube_markov_chain():
    """The 4-cube: 16 nodes, 32 edges, uniform stationary distribution."""
    return markov_chain_from_edge_list(_HYPERCUBE_EDGES)


def weighted_hypercube_markov_chain(rng=None):
    """The 4-cube with random integer edge weights in {2, ..., 20} (a
    symmetrized draw of two uniform integers in 1..10 per ordered pair), so
    the stationary distribution is non-uniform. Julia draws fresh weights
    on every call; here ``rng`` (a seed or numpy Generator) makes the draw
    reproducible."""
    rng = np.random.default_rng(rng)
    A = np.zeros((16, 16))
    for i, j in _HYPERCUBE_EDGES:
        A[i, j] = A[j, i] = 1.0
    M = rng.integers(1, 11, size=(16, 16)).astype(float)
    W = A * (M + M.T)
    return markov_chain_from_weight_matrix(W)


def wheel_markov_chain():
    """An 8-cycle (nodes 0-7) with a hub (node 8) joined to the odd nodes
    1, 3, 5, 7 (Julia's even nodes 2, 4, 6, 8): 12 edges, hub degree 4. Not
    the graph-theoretic wheel, whose hub joins every rim node. This is what
    the Julia package calls ``grid_markov_chain()`` with no argument."""
    return markov_chain_from_edge_list(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (8, 1), (8, 3), (8, 5), (8, 7)]
    )


def grid_markov_chain(n: int):
    """The n x n grid graph (n^2 nodes, row-major, nearest-neighbor edges), n >= 2."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"grid_markov_chain needs an integer n >= 2, got {n!r}")
    edges = []
    for i in range(n * n):
        if (i + 1) % n != 0:
            edges.append((i, i + 1))
        if i + n < n * n:
            edges.append((i, i + n))
    return markov_chain_from_edge_list(edges)
