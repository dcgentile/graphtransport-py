"""Ground costs on the nodes of a MarkovGraph for the entropic (Sinkhorn)
barycenter. Ported from the graph_diameter/ground_cost section of
GraphTransportation.jl's sinkhorn/Sinkhorn.jl.
"""

from __future__ import annotations

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


def ground_cost(G: MarkovGraph, rule: str, *, t: int | None = None, normalize: bool = True) -> np.ndarray:
    """A ground cost matrix on the nodes of G for the Sinkhorn barycenter.

    - "shortest_path": squared BFS hop count (edges of G unweighted).
    - "diffusion": squared diffusion distance at time t under the chain G.Q,
      D_t(x, y)^2 = sum_z (Q^t[x, z] - Q^t[y, z])^2 / pi(z). t defaults to
      the graph diameter so no pair has zero distance. Uses the chain G.Q
      itself, which coincides with the random walk on the adjacency only
      for unweighted chains.

    With normalize=True (default) the matrix is divided by its maximum so
    entries lie in [0, 1], which keeps the kernel exp(-cost/epsilon) from
    underflowing at the usual epsilon. Raises if the graph is disconnected.
    """
    if rule == "shortest_path":
        C = bfs_hops(G) ** 2
    elif rule == "diffusion":
        if t is None:
            t = graph_diameter(G)
        Qt = np.linalg.matrix_power(G.Q.toarray(), t)
        C = (((Qt[:, np.newaxis, :] - Qt[np.newaxis, :, :]) ** 2) / G.pi).sum(axis=-1)
    else:
        raise ValueError(f"ground_cost: rule must be one of {COST_RULES}, got {rule!r}")
    if not np.all(np.isfinite(C)):
        raise ValueError("ground_cost: graph is disconnected")
    return C / C.max() if normalize else C
