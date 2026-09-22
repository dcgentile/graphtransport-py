"""Geodesics by shooting: the Hamiltonian flow and the exp/log maps.

Ported from GraphTransportation.jl's shooting/. Self-contained numpy -- no
conic solver -- and it accepts every AdmissibleMean, including the exact
LogarithmicMean that the SOCP can only approximate with QuadLogMean.

Everything here needs **strictly positive** densities. For data supported on
part of the graph, use the SOCP (exact) instead.
"""

from graphtransport.shooting.barycenter import barycenter_shooting
from graphtransport.shooting.explog import (
    LogMapResult,
    MollifiedLogMapResult,
    ShootingError,
    analyze_shooting,
    exp_map,
    log_map,
    log_map_mollified,
    momentum_to_potential,
    solve_weighted_laplacian,
    weighted_laplacian,
)
from graphtransport.shooting.geodesic import geodesic_shooting
from graphtransport.shooting.hamiltonian import (
    PositivityFloorError,
    hamiltonian,
    hamiltonian_flow,
    integrate_hamiltonian,
    rho_floor,
)

__all__ = [
    "LogMapResult",
    "MollifiedLogMapResult",
    "PositivityFloorError",
    "ShootingError",
    "analyze_shooting",
    "barycenter_shooting",
    "exp_map",
    "geodesic_shooting",
    "log_map",
    "log_map_mollified",
    "momentum_to_potential",
    "solve_weighted_laplacian",
    "weighted_laplacian",
    "hamiltonian",
    "hamiltonian_flow",
    "integrate_hamiltonian",
    "rho_floor",
]
