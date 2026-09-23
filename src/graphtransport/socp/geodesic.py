"""The geodesic SOCP, ported from GraphTransportation.jl's socp/Geodesic.jl.

One geodesic's worth of variables and constraints (``geodesic_block``) --
density/momentum/mean/action variables, the discrete continuity equation,
the mean's hypograph in conic form and the action epigraph -- and the
single-geodesic program built from one block (``geodesic_socp``). The
barycenter program (a later step) shares several blocks' right endpoint.

Conic forms of the means (the constraint theta_e <= mean(rho_x, rho_y)),
all expressed through cvxpy's 3-d power cone x^a y^(1-a) >= |z|:

- GeometricMean:  theta^2 <= rx ry            -> PowCone3D(rx, ry, theta, 1/2)
- ArithmeticMean: theta <= (rx + ry)/2        -> a linear row
- HarmonicMean:   theta <= 2 rx ry/(rx + ry). Multiply by rx + ry > 0 and use
  4 rx ry = (rx+ry)^2 - (rx-ry)^2: (rx-ry)^2 <= (rx+ry)(rx+ry-2 theta)
                                              -> PowCone3D(rx+ry, rx+ry-2theta, rx-ry, 1/2)
- QuadLogMean(K): theta <= sum_k w_k theta_k with rx^a_k ry^(1-a_k) >= theta_k
                                              -> K power cones + a linear row
- LogarithmicMean has no finite conic representation (use QuadLogMean).

The action epigraph m^2 <= theta w is PowCone3D(theta, w, m, 1/2). These are
the same cones as the Julia rotated-SOC forms (2xy >= z^2 with z scaled by
sqrt 2), written without the sqrt-2 bookkeeping.
"""

from __future__ import annotations

import numpy as np

from graphtransport.api import GeodesicSolution
from graphtransport.graph import MarkovGraph
from graphtransport.means import (
    AdmissibleMean,
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    QuadLogMean,
)
from graphtransport.solvers import import_cvxpy, solve_conic


def _mean_cone(cp, mean: AdmissibleMean, rx, ry, theta):
    """Constraints for theta <= mean(rx, ry), elementwise over vectors."""
    if isinstance(mean, GeometricMean):
        return [cp.PowCone3D(rx, ry, theta, 0.5)]
    if isinstance(mean, ArithmeticMean):
        return [theta <= (rx + ry) / 2]
    if isinstance(mean, HarmonicMean):
        return [cp.PowCone3D(rx + ry, rx + ry - 2 * theta, rx - ry, 0.5)]
    if isinstance(mean, QuadLogMean):
        thetas = [cp.Variable(theta.shape, nonneg=True) for _ in range(mean.K)]
        cones = [cp.PowCone3D(rx, ry, tk, float(a)) for tk, a in zip(thetas, mean.alpha, strict=True)]
        return cones + [theta <= sum(float(w) * tk for w, tk in zip(mean.w, thetas, strict=True))]
    if isinstance(mean, LogarithmicMean):
        raise ValueError(
            "LogarithmicMean has no conic representation; use QuadLogMean(K) in the SOCP (K=8 is accurate to 1e-10)"
        )
    raise TypeError(f"unsupported mean for the SOCP: {mean!r}")


def _check_steps(N) -> None:
    """N must be a positive integer (bool is an int, hence the first clause)."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError(f"N must be an integer >= 1 (the number of time intervals), got {N!r}")


def _objective_value(problem) -> float:
    """A solved problem's objective, or NaN if the solve produced none."""
    return np.nan if problem.value is None else float(np.asarray(problem.value))


def _value(variable, shape: tuple[int, int]) -> np.ndarray:
    """A solved variable's value, or NaN of the right shape if the solve
    produced none. np.asarray(None, dtype=float) is the 0-d array nan rather
    than an error, so without this an infeasible solve returns a solution
    object whose fields are 0-d."""
    if variable.value is None:
        return np.full(shape, np.nan)
    return np.asarray(variable.value, dtype=float)


def geodesic_block(G: MarkovGraph, N: int, h: float, left, right) -> dict:
    """Variables and constraints for one geodesic on G with N time intervals
    of length h, from density ``left`` to density ``right`` (each a vector or
    a cvxpy expression, e.g. the shared barycenter variable).

    Returns a dict with the variables ``rho`` (n, N+1), ``m`` (|E|, N),
    ``theta`` (|E|, N), ``w`` (|E|, N); the endpoint constraints ``c_left``,
    ``c_right`` (whose duals carry the potentials, see ``endpoint_potentials``)
    and the continuity constraint ``c_cont`` ((n, N): column t is time step
    t); ``constraints``, the full list to hand to the problem; and
    ``action``, the expression sum_e kappa_e w_{e,t} summed over t (the
    block's contribution to the objective before the factor h).
    """
    cp = import_cvxpy()

    n = G.n
    nE = G.E.shape[0]
    rho = cp.Variable((n, N + 1), nonneg=True)
    m = cp.Variable((nE, N))
    theta = cp.Variable((nE, N), nonneg=True)
    w = cp.Variable((nE, N), nonneg=True)

    c_left = rho[:, 0] == left
    c_right = rho[:, N] == right
    # discrete continuity equation: (rho_{t+1} - rho_t)/h + div(m_t) = 0
    c_cont = (rho[:, 1:] - rho[:, :-1]) / h + G.D @ m == 0

    rbar = (rho[:, :-1] + rho[:, 1:]) / 2
    rx = cp.vec(rbar[G.E[:, 0], :], order="F")
    ry = cp.vec(rbar[G.E[:, 1], :], order="F")
    theta_v = cp.vec(theta, order="F")
    constraints = [c_left, c_right, c_cont]
    constraints += _mean_cone(cp, G.mean, rx, ry, theta_v)
    # action epigraph: m^2 <= theta w
    constraints.append(cp.PowCone3D(theta_v, cp.vec(w, order="F"), cp.vec(m, order="F"), 0.5))

    action = cp.sum(G.kappa @ w)
    return {
        "rho": rho, "m": m, "theta": theta, "w": w,
        "c_left": c_left, "c_right": c_right, "c_cont": c_cont,
        "constraints": constraints, "action": action,
    }  # fmt: skip


def endpoint_potentials(G: MarkovGraph, block: dict, *, weight: float = 1.0):
    """The potentials phi0, phi1 at a solved block's endpoints: the gradient
    of the block's action with respect to each endpoint density in the
    pi-weighted pairing, dW2 = <phi0, d left>_pi + <phi1, d right>_pi.
    ``weight`` is the factor multiplying this block's action in the
    objective (lam_i in the barycenter program), divided back out so the
    result is always the potential of the unweighted geodesic.

    cvxpy canonicalizes ``lhs == rhs`` as ``lhs - rhs == 0`` and reports the
    multiplier y with d(optimum)/d(rhs) = -y, so the sign is flipped here
    (calibrated against finite differences and the two-node closed form in
    the tests, as the Julia package does for JuMP's convention).
    """

    def phi(constraint):
        dual = constraint.dual_value
        if dual is None:  # a failed solve with check=False
            return np.full(G.n, np.nan)
        return -np.asarray(dual, dtype=float).reshape(G.n) / G.pi / weight

    return phi(block["c_left"]), phi(block["c_right"])


def geodesic_socp(
    G: MarkovGraph, rhoA, rhoB, *, N: int = 10, solver=None, check: bool = True, verbose: bool = False, **solver_kwargs
) -> GeodesicSolution:
    """The discrete transport geodesic between densities rhoA and rhoB on G as
    a single second-order-cone program. W2 is the squared distance.

    The mobility is G.mean: GeometricMean (default), ArithmeticMean,
    HarmonicMean or QuadLogMean(K) (K power cones per edge and time step); a
    graph built with LogarithmicMean has no conic form and raises. N is the
    number of time intervals (h = 1/N); rho has N+1 columns, m has N. With
    check=True a solve that does not reach an optimal status raises rather
    than returning the solver's last iterate. Extra keyword arguments are
    passed to the cvxpy solver.
    """
    cp = import_cvxpy()

    _check_steps(N)
    h = 1.0 / N
    rhoA = np.asarray(rhoA, dtype=float)
    rhoB = np.asarray(rhoB, dtype=float)
    blk = geodesic_block(G, N, h, rhoA, rhoB)
    problem = cp.Problem(cp.Minimize(h * blk["action"]), blk["constraints"])
    status = solve_conic(
        problem, solver, "geodesic_socp", check=check, verbose=verbose,
        hint="Try a smaller N, fewer QuadLogMean nodes, or check=False to inspect the iterate.",
        **solver_kwargs,
    )  # fmt: skip
    rho = _value(blk["rho"], (G.n, N + 1))
    m = _value(blk["m"], (G.E.shape[0], N))
    return GeodesicSolution(
        _objective_value(problem),
        rho, m, m[:, 0].copy(),
        *endpoint_potentials(G, blk), status, float(problem.solver_stats.solve_time or 0.0),
    )  # fmt: skip
