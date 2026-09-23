"""graphtransport: optimal transport and Wasserstein barycenters on graphs.

Geodesics, distances, barycenters and barycentric analysis for densities on
the nodes of a graph, under the discrete transport metric. The unified entry
points are geodesic, transport_cost, barycenter and analysis (see api.py).
Began as a port of the Julia package GraphTransportation.jl, against which its
numerics are cross-checked.
"""

from graphtransport.api import GeodesicSolution, ShootingFallbackWarning, analysis, barycenter, geodesic, transport_cost
from graphtransport.chains import (
    markov_chain_from_adjacency_matrix,
    markov_chain_from_edge_list,
    markov_chain_from_weight_matrix,
    stationary_from_transition,
)
from graphtransport.graph import MarkovGraph, graph_divergence, graph_gradient, metric_tensor
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
from graphtransport.means import (
    AdmissibleMean,
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    QuadLogMean,
)
from graphtransport.sinkhorn import graph_diameter, ground_cost

__version__ = "0.1.0"

__all__ = [
    "AdmissibleMean",
    "ArithmeticMean",
    "GeodesicSolution",
    "GeometricMean",
    "HarmonicMean",
    "LogarithmicMean",
    "MarkovGraph",
    "QuadLogMean",
    "ShootingFallbackWarning",
    "analysis",
    "barycenter",
    "cube_markov_chain",
    "double_t_markov_chain",
    "geodesic",
    "grid_markov_chain",
    "hypercube_markov_chain",
    "square_markov_chain",
    "t_markov_chain",
    "transport_cost",
    "triangle_markov_chain",
    "triangle_with_tail_markov_chain",
    "triangular_prism_markov_chain",
    "weighted_hypercube_markov_chain",
    "wheel_markov_chain",
    "graph_diameter",
    "graph_divergence",
    "graph_gradient",
    "ground_cost",
    "markov_chain_from_adjacency_matrix",
    "markov_chain_from_edge_list",
    "markov_chain_from_weight_matrix",
    "metric_tensor",
    "stationary_from_transition",
]
