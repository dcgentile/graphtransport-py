"""The discrete transport geodesic by shooting, ported from the `:shooting`
branch of GraphTransportation.jl's API.jl (`_geodesic_shooting`)."""

from __future__ import annotations

import time

import numpy as np

from graphtransport.api import GeodesicSolution
from graphtransport.graph import MarkovGraph
from graphtransport.shooting.explog import log_map
from graphtransport.shooting.hamiltonian import integrate_hamiltonian


def geodesic_shooting(
    G: MarkovGraph,
    rhoA,
    rhoB,
    *,
    nsteps: int = 150,
    tol: float = 1e-9,
    maxiters: int = 50,
    phi0_init=None,
    floor_rtol: float = 1e-6,
    segments="auto",
    verbose: bool = False,
) -> GeodesicSolution:
    """The geodesic from rhoA to rhoB: Newton shooting on the Hamiltonian flow
    (log_map) for the initial potential, then the flow integrated to produce
    the path. Fourth-order in time (RK4's truncation error), and requires both
    endpoints strictly positive.

    rho has nsteps + 1 columns and m has nsteps, m[:, t] being the momentum
    theta(rho_t) * grad(phi_t) at the start of step t. The endpoint potentials
    use the SOCP's W2-gradient convention, so the two methods' phi0/phi1 are
    comparable: phi0 = -2 phi(0) and phi1 = 2 phi(1), where phi is the flow's
    velocity potential.

    When the log map is solved by multiple shooting (``segments`` > 1, or the
    default "auto" on a long transport; see log_map), the
    path is integrated segment by segment from the solved junction states
    rather than in one sweep from phi0: over a long transport that sweep
    amplifies the error in phi0 the same way single shooting's Newton system
    does. The potential's constant drift along each segment is carried across
    the junctions, so phi is continuous and phi1 matches single shooting's.
    """
    t0 = time.perf_counter()
    r = log_map(
        G,
        rhoA,
        rhoB,
        nsteps=nsteps,
        tol=tol,
        maxiters=maxiters,
        phi0_init=phi0_init,
        floor_rtol=floor_rtol,
        segments=segments,
        verbose=verbose,
    )
    if r.starts is None:
        rhoA = np.asarray(rhoA, dtype=float)
        rho_path, phi_path = integrate_hamiltonian(
            G, rhoA / (rhoA @ G.pi), r.phi0, nsteps=nsteps, floor_rtol=floor_rtol
        )
    else:
        rho_path, phi_path = _stitch_segments(G, r.starts, nsteps, floor_rtol)
    x, y = G.E[:, 0], G.E[:, 1]
    rho_t, phi_t = rho_path[:, :-1], phi_path[:, :-1]
    m = G.mean(rho_t[x], rho_t[y]) * (phi_t[x] - phi_t[y])
    return GeodesicSolution(
        W2=r.W2,
        rho=rho_path,
        m=m,
        m0=m[:, 0].copy(),
        phi0=-2 * r.phi0,
        phi1=2 * phi_path[:, -1],
        status="converged",
        solvetime=time.perf_counter() - t0,
    )


def _stitch_segments(G: MarkovGraph, starts, nsteps: int, floor_rtol: float):
    """The (n, nsteps + 1) path from multiple shooting's segment start states,
    each segment integrated on its own. Each segment's gauge-fixed start
    potential is shifted by the constant the previous segment's potential
    drifted by, so phi is continuous across the junctions."""
    from graphtransport.shooting.multiple import segment_steps

    rho_s, phi_s = starts
    rho_path, phi_path = [rho_s[:, [0]]], [phi_s[:, [0]]]
    shift = 0.0
    for k, steps in enumerate(segment_steps(nsteps, rho_s.shape[1])):
        rho_k, phi_k = integrate_hamiltonian(
            G, rho_s[:, k], phi_s[:, k] + shift, nsteps=steps, T=steps / nsteps, floor_rtol=floor_rtol
        )
        rho_path.append(rho_k[:, 1:])
        phi_path.append(phi_k[:, 1:])
        if k + 1 < rho_s.shape[1]:
            shift = float(phi_k[:, -1] @ G.pi) / G.pi.sum()
    return np.hstack(rho_path), np.hstack(phi_path)
