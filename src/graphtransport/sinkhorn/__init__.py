"""Entropic optimal transport on graphs: ground costs, Sinkhorn barycenters,
and barycentric-coordinate recovery."""

from graphtransport.sinkhorn.core import (
    barycentric_loss,
    build_geodesic,
    logarithmic_change_of_variable,
    loss_gradient,
    regularize_cost,
    simplex_regression,
    sinkhorn_barycenter,
    sinkhorn_differentiate,
    sinkhorn_plan,
    sqeuc_loss,
)
from graphtransport.sinkhorn.cost import bfs_hops, graph_diameter, ground_cost

__all__ = [
    "barycentric_loss",
    "bfs_hops",
    "build_geodesic",
    "graph_diameter",
    "ground_cost",
    "logarithmic_change_of_variable",
    "loss_gradient",
    "regularize_cost",
    "simplex_regression",
    "sinkhorn_barycenter",
    "sinkhorn_differentiate",
    "sinkhorn_plan",
    "sqeuc_loss",
]
