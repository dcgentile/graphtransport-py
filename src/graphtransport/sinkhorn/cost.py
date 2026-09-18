"""Ground costs on the nodes of a MarkovGraph for the entropic (Sinkhorn)
barycenter. Ported from the graph_diameter/ground_cost section of
GraphTransportation.jl's sinkhorn/Sinkhorn.jl.
"""

from __future__ import annotations

import warnings
from collections import deque

import numpy as np

from graphtransport.graph import MarkovGraph

COST_RULES = ("shortest_path", "diffusion")


def bfs_hops(G: MarkovGraph) -> np.ndarray:
    """All-pairs BFS hop counts, edges of G unweighted; inf where unreachable."""
    n = G.n
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for x, y in G.E:
        nbrs[x].append(int(y))
        nbrs[y].append(int(x))
    D = np.full((n, n), np.inf)
    for s in range(n):
        D[s, s] = 0.0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in nbrs[u]:
                if D[s, v] == np.inf:
                    D[s, v] = D[s, u] + 1
                    queue.append(v)
    return D


def graph_diameter(G: MarkovGraph) -> int:
    """Largest BFS hop count between any two nodes. Raises if G is disconnected."""
    D = bfs_hops(G)
    if not np.all(np.isfinite(D)):
        raise ValueError("graph_diameter: graph is disconnected")
    return int(round(D.max()))


def ground_cost(
    G: MarkovGraph, rule: str, *, t: int | None = None, laziness: float = 0.5, normalize: bool = True
) -> np.ndarray:
    """A ground cost matrix on the nodes of G for the Sinkhorn barycenter.

    - "shortest_path": squared BFS hop count (edges of G unweighted).
    - "diffusion": squared diffusion distance at time t,
      D_t(x, y)^2 = sum_z (P^t[x, z] - P^t[y, z])^2 / pi(z), under the lazy
      chain P = laziness * I + (1 - laziness) * G.Q (same stationary pi).
      t defaults to the graph diameter. Uses the chain G.Q itself, which
      coincides with the random walk on the adjacency only for unweighted
      chains. laziness=0 is the plain walk, as in GraphTransportation.jl.

    With normalize=True (default) the matrix is divided by its maximum so
    entries lie in [0, 1], which keeps the kernel exp(-cost/epsilon) from
    underflowing at the usual epsilon. Raises if the graph is disconnected.

    Degeneracy of the diffusion distance. Spectrally D_t^2 = sum_j
    lambda_j^(2t) (psi_j(x) - psi_j(y))^2 over the eigenpairs of P, so D_t is
    a metric only when P has no zero eigenvalue; otherwise those coordinates
    drop out and D_t is a pseudo-metric, D_t(x, y) = 0 exactly when
    (e_x - e_y) P^t = 0. For the plain walk this is common: two nodes with
    the same neighbourhood (the ends of a 3-path, opposite corners of a
    4-cycle, the leaves of a star) have identical rows of Q, hence zero cost
    for every t >= 1, and a Sinkhorn barycenter moves mass between them for
    free. The default laziness 1/2 maps each eigenvalue lambda of Q to
    (1 + lambda) / 2 >= 0, which sends those pairs to 1/2 and also removes
    the period-2 oscillation on bipartite graphs; the only zero left is
    lambda = -1, and it collapses a pair only on the two-node graph (any
    laziness > 1/2 avoids even that). Because a custom chain or laziness can
    reintroduce the problem, a cost with a zero between distinct nodes warns,
    and an identically zero cost raises.
    """
    # The diffusion cost is finite even across components, so check connectivity up front.
    hops = bfs_hops(G)
    if not np.all(np.isfinite(hops)):
        raise ValueError("ground_cost: graph is disconnected")
    if rule == "shortest_path":
        C = hops**2
    elif rule == "diffusion":
        if t is None:
            t = int(round(hops.max()))
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 1:
            raise ValueError(f"ground_cost: t must be an integer >= 1, got {t!r}")
        if not 0 <= laziness < 1:
            raise ValueError(f"ground_cost: laziness must lie in [0, 1), got {laziness!r}")
        P = laziness * np.eye(G.n) + (1 - laziness) * G.Q.toarray()
        Pt = np.linalg.matrix_power(P, int(t))
        C = (((Pt[:, np.newaxis, :] - Pt[np.newaxis, :, :]) ** 2) / G.pi).sum(axis=-1)
        _check_separates_nodes(C)
    else:
        raise ValueError(f"ground_cost: rule must be one of {COST_RULES}, got {rule!r}")
    scale = C.max()  # zero only for a single-node graph
    return C / scale if normalize and scale > 0 else C


def _check_separates_nodes(C: np.ndarray) -> None:
    """Flag a diffusion cost that is only a pseudo-metric (see ground_cost)."""
    n = C.shape[0]
    if n < 2:
        return
    collapsed = np.argwhere(np.triu(C <= 1e-14 * C.max(), k=1))
    if len(collapsed) == n * (n - 1) // 2:
        raise ValueError(
            "ground_cost: the diffusion cost is identically zero (every row of P^t coincides); "
            'use a different laziness or rule="shortest_path"'
        )
    if len(collapsed):
        x, y = collapsed[0]
        warnings.warn(
            f"ground_cost: the diffusion cost is zero between {len(collapsed)} pair(s) of distinct nodes "
            f"(e.g. {x} and {y}), so transport between them is free; the chain has a zero eigenvalue. "
            'Use laziness > 0 or rule="shortest_path".',
            stacklevel=3,
        )
