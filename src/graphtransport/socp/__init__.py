"""Discrete transport geodesics, barycenters and analysis as second-order
cone programs (cvxpy + Clarabel/SCS). Ported from GraphTransportation.jl's
socp/. Optional dependency: ``pip install "graphtransport[socp]"``."""

from graphtransport.socp.analysis import analyze_socp
from graphtransport.socp.barycenter import barycenter_socp
from graphtransport.socp.geodesic import endpoint_potentials, geodesic_block, geodesic_socp

__all__ = ["analyze_socp", "barycenter_socp", "endpoint_potentials", "geodesic_block", "geodesic_socp"]
