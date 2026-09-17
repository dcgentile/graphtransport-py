"""Entropic optimal transport on graphs: ground costs, Sinkhorn barycenters,
and barycentric-coordinate recovery."""

from graphtransport.sinkhorn.core import (
    build_geodesic,
    logarithmic_change_of_variable,
    regularize_cost,
    sinkhorn_barycenter,
    sinkhorn_plan,
)
from graphtransport.sinkhorn.cost import bfs_hops, graph_diameter, ground_cost

__all__ = [
    "bfs_hops",
    "build_geodesic",
    "graph_diameter",
    "ground_cost",
    "logarithmic_change_of_variable",
    "regularize_cost",
    "sinkhorn_barycenter",
    "sinkhorn_plan",
]
