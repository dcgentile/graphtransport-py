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
from graphtransport.means import (
    AdmissibleMean,
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    QuadLogMean,
)

__version__ = "0.1.0"

__all__ = [
    "AdmissibleMean",
    "ArithmeticMean",
    "GeometricMean",
    "HarmonicMean",
    "LogarithmicMean",
    "MarkovGraph",
    "QuadLogMean",
    "graph_divergence",
    "graph_gradient",
    "markov_chain_from_adjacency_matrix",
    "markov_chain_from_edge_list",
    "markov_chain_from_weight_matrix",
    "metric_tensor",
    "stationary_from_transition",
]
