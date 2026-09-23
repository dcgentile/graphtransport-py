"""The barycenter SOCP, ported from GraphTransportation.jl's socp/Barycenter.jl."""

from __future__ import annotations

import numpy as np

from graphtransport.api import GeodesicSolution
from graphtransport.graph import MarkovGraph
from graphtransport.socp.geodesic import (
    _check_steps,
    _objective_value,
    _value,
    endpoint_potentials,
    geodesic_block,
)
from graphtransport.solvers import import_cvxpy, solve_conic


def barycenter_socp(
    G: MarkovGraph, refs, lam, *, N: int = 10, solver=None, check: bool = True, verbose: bool = False, **solver_kwargs
):
    """The discrete transport barycenter of the reference densities ``refs``
    with weights ``lam`` as a single joint second-order-cone program: one
    geodesic block per reference with lam_i > 0, all sharing a free right
    endpoint nu.

    Returns (nu, J, geodesics): the barycenter, the optimal objective
    J = sum_i lam_i W_h^2(refs_i, nu), and one GeodesicSolution per active
    reference (in the order of ``refs``) from that reference to nu, each
    carrying its endpoint potentials for the *unweighted* geodesic (the
    lam_i factor is divided out of the duals) and its ``ref_index``, the
    position of its reference in ``refs``. Since zero-weight references are
    dropped, ``geodesics[k]`` need not be the geodesic of ``refs[k]``;
    ``ref_index`` is what relates the two. KKT stationarity of the joint
    program in nu is exactly sum_i lam_i phi1_i = const on the support of
    nu, which is what the potential-based analysis checks.

    References with lam_i == 0 are dropped rather than solved with zero
    weight. ``lam`` must be a probability vector.
    """
    cp = import_cvxpy()

    lam = np.asarray(lam, dtype=float)
    if len(refs) != len(lam):
        raise ValueError(f"got {len(refs)} references but {len(lam)} weights")
    if np.any(lam < 0) or abs(lam.sum() - 1.0) > 1e-8:
        raise ValueError("lam must be a probability vector (lam >= 0, sum(lam) == 1)")
    active = np.flatnonzero(lam > 0)
    if active.size == 0:
        raise ValueError("at least one lam_i must be > 0")

    _check_steps(N)
    h = 1.0 / N
    nu = cp.Variable(G.n, nonneg=True)
    constraints: list = [nu @ G.pi == 1]
    blocks = []
    for i in active:
        blk = geodesic_block(G, N, h, np.asarray(refs[i], dtype=float), nu)
        blocks.append((int(i), blk))
        constraints += blk["constraints"]

    objective = h * sum(float(lam[i]) * blk["action"] for i, blk in blocks)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    status = solve_conic(
        problem,
        solver,
        "barycenter_socp",
        check=check,
        verbose=verbose,
        hint="Try a smaller N, fewer QuadLogMean nodes, or check=False to inspect the iterate. The joint "
        "program is len(lam > 0) times the size of one geodesic.",
        **solver_kwargs,
    )
    solvetime = float(problem.solver_stats.solve_time or 0.0)

    # A failed solve leaves every .value at None; report NaN of the right shape
    # rather than raising on the arithmetic (see _value).
    geodesics = []
    for i, blk in blocks:
        phi0, phi1 = endpoint_potentials(G, blk, weight=float(lam[i]))
        action = blk["action"].value
        m = _value(blk["m"], (G.E.shape[0], N))
        geodesics.append(
            GeodesicSolution(
                float(h * action) if action is not None else np.nan,
                _value(blk["rho"], (G.n, N + 1)),
                m,
                m[:, 0].copy(),
                phi0,
                phi1,
                status,
                solvetime,
                i,
            )
        )
    nu_value = np.full(G.n, np.nan) if nu.value is None else np.asarray(nu.value, dtype=float)
    J = _objective_value(problem)
    return nu_value, J, geodesics
