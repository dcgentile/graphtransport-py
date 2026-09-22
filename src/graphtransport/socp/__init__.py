"""Discrete transport geodesics, barycenters and analysis as second-order
cone programs (cvxpy + Clarabel/SCS). Ported from GraphTransportation.jl's
socp/. Optional dependency: ``pip install "graphtransport[socp]"``."""

from graphtransport.socp.geodesic import geodesic_block, geodesic_socp

__all__ = ["geodesic_block", "geodesic_socp"]
