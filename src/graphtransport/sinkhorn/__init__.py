"""Entropic optimal transport on graphs: ground costs, Sinkhorn barycenters,
and barycentric-coordinate recovery."""

from graphtransport.sinkhorn.cost import bfs_hops, graph_diameter, ground_cost

__all__ = ["bfs_hops", "graph_diameter", "ground_cost"]
