"""The discrete transport geodesic by shooting, ported from the `:shooting`
branch of GraphTransportation.jl's API.jl (`_geodesic_shooting`)."""

from __future__ import annotations

import time

import numpy as np

from graphtransport.api import GeodesicSolution
from graphtransport.graph import MarkovGraph
from graphtransport.shooting.explog import log_map
from graphtransport.shooting.hamiltonian import integrate_hamiltonian


def geodesic_shooting(G: MarkovGraph, rhoA, rhoB, *, nsteps: int = 150, tol: float = 1e-9, maxiters: int = 50,
                      phi0_init=None, floor_rtol: float = 1e-6, verbose: bool = False) -> GeodesicSolution:
    """The geodesic from rhoA to rhoB: Newton shooting on the Hamiltonian flow
    (log_map) for the initial potential, then the flow integrated to produce
    the path. Exact in time up to RK4's truncation error, and requires both
    endpoints strictly positive.

    rho has nsteps + 1 columns and m has nsteps, m[:, t] being the momentum
    theta(rho_t) * grad(phi_t) at the start of step t. The endpoint potentials
    use the SOCP's W2-gradient convention, so the two methods' phi0/phi1 are
    comparable: phi0 = -2 phi(0) and phi1 = 2 phi(1), where phi is the flow's
    velocity potential.
    """  # fmt: skip
    t0 = time.perf_counter()
    r = log_map(G, rhoA, rhoB, nsteps=nsteps, tol=tol, maxiters=maxiters, phi0_init=phi0_init,
                floor_rtol=floor_rtol, verbose=verbose)
    rhoA = np.asarray(rhoA, dtype=float)
    rho_path, phi_path = integrate_hamiltonian(G, rhoA / (rhoA @ G.pi), r.phi0, nsteps=nsteps, floor_rtol=floor_rtol)
    x, y = G.E[:, 0], G.E[:, 1]
    rho_t, phi_t = rho_path[:, :-1], phi_path[:, :-1]
    m = G.mean(rho_t[x], rho_t[y]) * (phi_t[x] - phi_t[y])
    return GeodesicSolution(
        r.W2, rho_path, m, m[:, 0].copy(), -2 * r.phi0, 2 * phi_path[:, -1], "converged",
        time.perf_counter() - t0,
    )  # fmt: skip
