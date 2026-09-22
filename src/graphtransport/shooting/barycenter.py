"""The discrete transport barycenter by Riemannian gradient descent with
exact-in-time geodesics. Ported from the `:shooting` branch of
GraphTransportation.jl's API.jl (`_barycenter_shooting`)."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from graphtransport.graph import MarkovGraph
from graphtransport.shooting.explog import ShootingError, exp_map, log_map
from graphtransport.shooting.hamiltonian import PositivityFloorError, _check_interior, hamiltonian, rho_floor

logger = logging.getLogger(__name__)


def _positive_float(value, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {bound}, got {value!r}")
    return value


def barycenter_shooting(G: MarkovGraph, refs, lam, *, h: float = 1.0, maxiters: int = 200, tol: float = 1e-5,
                        ftol: float = 1e-12, nsteps: int = 150, log_tol: float = 1e-12, init=None,
                        floor_rtol: float = 1e-6, verbose: bool = False):
    """The discrete transport barycenter of ``refs`` with weights ``lam``, by
    intrinsic gradient descent: each iteration log-maps nu to every reference
    (warm-started from the previous iteration, retried cold if that stalls),
    forms the descent direction sum_i lam_i phi0_i as a potential, and moves
    along the geodesic it generates with exp_map.

    Works with every AdmissibleMean, including the exact LogarithmicMean that
    the SOCP cannot represent, and requires every reference with lam_i > 0
    strictly positive.

    Steps are accepted only on a strict decrease of J, so the objective
    history is monotone. The step h is halved on a positivity-floor hit, an
    unreachable reference or an increase of J, and doubled back toward its
    initial value after an accepted step. The descent is first order, so it
    converges linearly; barycenter_socp remains the certificate.

    Returns (nu, J, info), info = {"iters", "status", "J_hist", "grad_hist",
    "h"}: the objective and the Riemannian gradient norm per iteration
    (including the final point), and the final step size. status is
    "converged" (gradient norm below ``tol``), "stalled" (no step decreases J
    by more than ``ftol`` relative: the descent has reached the precision of
    the log maps) or "maxiters" (which warns). A step decreases J by about
    h ||g||^2, so asking for ``tol`` below sqrt(ftol * J) typically ends as
    "stalled" rather than "converged".
    """  # fmt: skip
    h = _positive_float(h, "h")
    tol = _positive_float(tol, "tol")
    ftol = _positive_float(ftol, "ftol", allow_zero=True)
    log_tol = _positive_float(log_tol, "log_tol")
    if isinstance(maxiters, bool) or not isinstance(maxiters, (int, np.integer)) or maxiters < 1:
        raise ValueError(f"maxiters must be an integer >= 1, got {maxiters!r}")
    lam = np.asarray(lam, dtype=float)
    refs = [np.asarray(r, dtype=float) for r in refs]
    active = [int(i) for i in np.flatnonzero(lam > 0)]
    if not active:
        raise ValueError("at least one lam_i must be > 0")

    floor_val = rho_floor(G, rtol=floor_rtol)
    if init is None:
        nu = sum(lam[i] * refs[i] for i in active)
    else:
        nu = _check_interior(G, init, floor_val, "init")
    nu = nu / (nu @ G.pi)

    def logmaps(base, inits):
        # Log-map base to every active reference, warm-started from `inits` and
        # retried cold if the warm start stalls; None if one cannot be reached.
        out = {}
        for i in active:
            result = None
            for start in (inits[i], None):
                try:
                    result = log_map(G, base, refs[i], nsteps=nsteps, tol=log_tol, phi0_init=start,
                                     floor_rtol=floor_rtol)
                    break
                except (ShootingError, PositivityFloorError):
                    if start is None:
                        break
            if result is None:
                return None
            out[i] = result
        return out

    def objective(rs):
        return float(sum(lam[i] * rs[i].W2 for i in active))

    def gradient(base, rs):
        g = sum(lam[i] * rs[i].phi0 for i in active)  # the Riemannian descent direction, as a potential
        return g, np.sqrt(2 * hamiltonian(G, base, g, floor_rtol=floor_rtol))  # and its metric norm at base

    rs = logmaps(nu, {i: None for i in active})
    if rs is None:
        raise ShootingError(
            "barycenter(method='shooting'): a reference is not reachable by shooting from the initial point; "
            "use method='socp'"
        )
    J, h0 = objective(rs), h
    J_hist, grad_hist = [], []
    iters, status = 0, "maxiters"
    for k in range(1, maxiters + 1):
        g, gnorm = gradient(nu, rs)
        J_hist.append(J)
        grad_hist.append(gnorm)
        if verbose:
            logger.info("barycenter(shooting): iter %d  J %.12g  |grad| %.3e  h %g", k, J, gnorm, h)
        if gnorm < tol:
            status = "converged"
            break
        # Step along the geodesic, halving h on a floor hit, an unreachable
        # reference or an increase of J. The candidate's log maps are kept.
        accepted = decreased = unreachable = False
        for _ in range(20):
            try:
                # kind is explicit: on a cycle n == |E| and inference would refuse
                candidate = exp_map(G, nu, h * g, nsteps=nsteps, kind="potential", floor_rtol=floor_rtol)
            except PositivityFloorError:
                candidate = None
            unreachable = candidate is None or candidate.min() <= floor_val
            if not unreachable:
                rs_new = logmaps(candidate, {i: rs[i].phi0 for i in active})
                unreachable = rs_new is None
                if not unreachable and objective(rs_new) < J:
                    J_new = objective(rs_new)
                    decreased = J - J_new > ftol * abs(J)
                    nu, rs, J = candidate, rs_new, J_new
                    accepted = True
                    h = min(2 * h, h0)  # let the step recover after a halving
                    break
            h /= 2
        iters = k
        if not accepted:
            if unreachable:
                raise ShootingError(
                    f"barycenter(method='shooting'): no admissible step found at iteration {k} (h={h:g}); the "
                    "barycenter may touch the boundary. Use method='socp'."
                )
            status = "stalled"  # no step decreases J: at the precision of the log maps
            break
        if not decreased:
            status = "stalled"
            break
    if status == "stalled" and len(J_hist) == iters:  # record the accepted final point
        _, gnorm = gradient(nu, rs)
        J_hist.append(J)
        grad_hist.append(gnorm)
        if gnorm < tol:
            status = "converged"
    if status == "maxiters":
        warnings.warn(
            f"barycenter(method='shooting') reached maxiters={maxiters} with gradient norm {grad_hist[-1]:.3e} "
            f"> tol={tol:g}",
            stacklevel=2,
        )
    info = {"iters": iters, "status": status, "J_hist": np.array(J_hist), "grad_hist": np.array(grad_hist), "h": h}
    return nu, J, info
