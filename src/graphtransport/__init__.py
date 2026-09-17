"""graphtransport: optimal transport and Wasserstein barycenters on graphs.

Python port of GraphTransportation.jl. Modules are added incrementally; see
README.md for the porting plan and current status.
"""

from graphtransport.chains import (
    markov_chain_from_adjacency_matrix,
    markov_chain_from_edge_list,
    markov_chain_from_weight_matrix,
    stationary_from_transition,
)
from graphtransport.graph import MarkovGraph, graph_divergence, graph_gradient, metric_tensor
from graphtransport.means import geomean, logmean, logmean_partial_s, logmean_partial_t

__version__ = "0.1.0"

__all__ = [
    "MarkovGraph",
    "geomean",
    "graph_divergence",
    "graph_gradient",
    "logmean",
    "logmean_partial_s",
    "logmean_partial_t",
    "markov_chain_from_adjacency_matrix",
    "markov_chain_from_edge_list",
    "markov_chain_from_weight_matrix",
    "metric_tensor",
    "stationary_from_transition",
]
