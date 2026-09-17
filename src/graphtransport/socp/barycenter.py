"""The barycenter SOCP, ported from GraphTransportation.jl's socp/Barycenter.jl."""

from __future__ import annotations

import numpy as np

from graphtransport.api import GeodesicSolution
from graphtransport.graph import MarkovGraph
from graphtransport.socp.geodesic import endpoint_potentials, geodesic_block
from graphtransport.solvers import solve_conic


def barycenter_socp(G: MarkovGraph, refs, lam, *, N: int = 10, solver=None, check: bool = True,
                    verbose: bool = False, **solver_kwargs):
    """The discrete transport barycenter of the reference densities ``refs``
    with weights ``lam`` as a single joint second-order-cone program: one
    geodesic block per reference with lam_i > 0, all sharing a free right
    endpoint nu.

    Returns (nu, J, geodesics): the barycenter, the optimal objective
    J = sum_i lam_i W_h^2(refs_i, nu), and one GeodesicSolution per active
    reference (in the order of ``refs``) from that reference to nu, each
    carrying its endpoint potentials for the *unweighted* geodesic (the
    lam_i factor is divided out of the duals). KKT stationarity of the joint
    program in nu is exactly sum_i lam_i phi1_i = const on the support of
    nu, which is what the potential-based analysis checks.

    References with lam_i == 0 are dropped rather than solved with zero
    weight. ``lam`` must be a probability vector.
    """
    import cvxpy as cp

    lam = np.asarray(lam, dtype=float)
    if len(refs) != len(lam):
        raise ValueError(f"got {len(refs)} references but {len(lam)} weights")
    if np.any(lam < 0) or abs(lam.sum() - 1.0) > 1e-8:
        raise ValueError("lam must be a probability vector (lam >= 0, sum(lam) == 1)")
    active = np.flatnonzero(lam > 0)
    if active.size == 0:
        raise ValueError("at least one lam_i must be > 0")

    h = 1.0 / N
    nu = cp.Variable(G.n, nonneg=True)
    constraints = [nu @ G.pi == 1]
    blocks = []
    for i in active:
        blk = geodesic_block(G, N, h, np.asarray(refs[i], dtype=float), nu)
        blocks.append((int(i), blk))
        constraints += blk["constraints"]

    objective = h * sum(float(lam[i]) * blk["action"] for i, blk in blocks)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    status = solve_conic(problem, solver, "barycenter_socp", check=check, verbose=verbose, **solver_kwargs)
    solvetime = float(problem.solver_stats.solve_time or 0.0)

    geodesics = []
    for i, blk in blocks:
        phi0, phi1 = endpoint_potentials(G, blk, weight=float(lam[i]))
        m = np.asarray(blk["m"].value, dtype=float)
        geodesics.append(
            GeodesicSolution(
                float(h * blk["action"].value), np.asarray(blk["rho"].value, dtype=float),
                m, m[:, 0].copy(), phi0, phi1, status, solvetime,
            )  # fmt: skip
        )
    return np.asarray(nu.value, dtype=float), float(problem.value), geodesics
