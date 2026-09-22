"""Geodesics by shooting: the Hamiltonian flow and the exp/log maps.

Ported from GraphTransportation.jl's shooting/. Self-contained numpy -- no
conic solver -- and it accepts every AdmissibleMean, including the exact
LogarithmicMean that the SOCP can only approximate with QuadLogMean.

Everything here needs **strictly positive** densities. For data supported on
part of the graph, use the SOCP (exact) instead.
"""

from graphtransport.shooting.hamiltonian import (
    PositivityFloorError,
    hamiltonian,
    hamiltonian_flow,
    integrate_hamiltonian,
    rho_floor,
)

__all__ = [
    "PositivityFloorError",
    "hamiltonian",
    "hamiltonian_flow",
    "integrate_hamiltonian",
    "rho_floor",
]
