"""Unified entry points, ported from GraphTransportation.jl's API.jl.

One public function per task with a ``method`` keyword selecting the
numerical algorithm:

- ``"shooting"`` (default): Newton shooting on the Hamiltonian flow. Exact in
  time, no conic solver, every AdmissibleMean; requires **strictly
  positive** densities, and falls back to the SOCP with a
  ShootingFallbackWarning when it cannot take the data or fails on it
  (``fallback=False`` raises instead).
- ``"socp"``: a second-order-cone program (needs ``graphtransport[socp]``).
  Handles densities supported on part of the graph, which shooting cannot;
  time-discretisation error O(1/N).
- ``"sinkhorn"``: entropic regularisation for a ground cost -- a different
  object from the other two, which compute the same discrete transport
  geodesic.

The default diverges from the Julia package, which defaults to ``:socp``.
What shooting offers is exactness in time and no conic solver, not speed:
the SOCP at its default N=10 is faster on the graphs measured (see the
README). Shooting also cannot take boundary-supported data, and gets stiff
as densities approach zero; both are reasons to pass ``method="socp"``, and
the fallback warning says so.

Densities: as in the Julia package, ``rhoA``, ``rhoB``, ``refs``, ``target``
and returned barycenters are densities with respect to the graph's
stationary distribution ``G.pi`` (``sum(rho * G.pi) == 1``), not probability
vectors. The Sinkhorn method converts to probability vectors internally.
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np

from graphtransport.graph import MarkovGraph
from graphtransport.sinkhorn.core import (
    build_geodesic,
    regularize_cost,
    simplex_regression,
    sinkhorn_barycenter,
    sinkhorn_plan,
)

# A density must integrate to 1 against pi to this tolerance. It is loose
# enough to accept the output of an iterative solver (a barycenter fed back
# into analysis) and tight enough to catch a density that was never normalised.
MASS_TOL = 1e-6


@dataclass
class GeodesicSolution:
    """A discrete transport geodesic.

    W2: the squared transport distance. rho: the (n, steps+1) density path.
    m: the (|E|, steps) momentum path and m0 its first column. phi0, phi1:
    endpoint potentials (gradients of W2 with respect to each endpoint).
    status: solver status; for method="sinkhorn", "converged" if the endpoint
    plan's marginals are met to `tol`, else "iteration_limit".
    solvetime: wall-clock seconds.
    ref_index: for a geodesic that is part of a barycenter, the index of its
    reference in the ``refs`` passed to barycenter; None for a standalone
    geodesic. References with lam_i == 0 are not solved, so the list of
    geodesics can be shorter than refs and this is what ties each one back.

    Methods that do not produce a quantity fill it with NaN (the Sinkhorn
    method has no momenta or potentials).
    """

    W2: float
    rho: np.ndarray
    m: np.ndarray
    m0: np.ndarray
    phi0: np.ndarray
    phi1: np.ndarray
    status: str
    solvetime: float
    ref_index: int | None = None


DEFAULT_METHOD = "shooting"


def _check_method(method: str, table: dict, what: str):
    if method not in table:
        raise ValueError(f"{what}: method must be one of {tuple(table)}, got {method!r}")


# The keywords each method owns. A keyword owned by some method but not the
# selected one is rejected by name: otherwise it falls through **kwargs and
# surfaces far away -- as "Clarabel: unrecognized solver setting 'cost'" for
# the SOCP, or a TypeError from a function the caller never called. Keywords
# owned by several methods (tol, N, qp_method) pass for each owner; keywords
# owned by none (verbose, compute_condition, a solver's own options) are left
# to the method's signature.
METHOD_KEYWORDS = {
    "shooting": frozenset({"nsteps", "tol", "maxiters", "phi0_init", "phi0_inits", "floor_rtol", "h", "ftol",
                           "log_tol", "init", "qp_method", "qp_solver", "fallback"}),
    "socp": frozenset({"N", "solver", "check", "convention", "qp_method", "qp_solver"}),
    "sinkhorn": frozenset({"N", "cost", "epsilon", "iters", "tol", "alpha0"}),
}  # fmt: skip


def _check_kwargs(method: str, kwargs: dict, what: str) -> None:
    owned_elsewhere = set().union(*METHOD_KEYWORDS.values()) - METHOD_KEYWORDS[method]
    stray = [k for k in kwargs if k in owned_elsewhere]
    if not stray:
        return
    described = []
    for k in stray:
        owners = " and ".join(f"method={m!r}" for m, keys in METHOD_KEYWORDS.items() if k in keys)
        described.append(f"{k} (a keyword of {owners})")
    hint = ""
    if method == "shooting" and "N" in stray:
        hint = " Shooting is exact in time; its integrator's resolution is nsteps= (default 150)."
    raise TypeError(f"{what}: method={method!r} does not take {', '.join(described)}.{hint}")


# Input checks shared by every method. They live in the public entry points,
# not in the per-method functions, so a new method cannot forget them.


def _check_density(G: MarkovGraph, rho, name: str) -> np.ndarray:
    """rho as a float array, checked to be a probability density w.r.t. G.pi."""
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (G.n,):
        raise ValueError(f"{name} must be a density of shape ({G.n},), one value per node, got {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"{name} has non-finite entries")
    floor = -1e-9 * max(1.0, float(np.abs(rho).max()))
    if rho.min() < floor:
        raise ValueError(f"{name} has negative entries (min {rho.min():.3e}); a density must be nonnegative")
    rho = np.maximum(rho, 0.0)  # round-off negatives from an upstream solver
    mass = float(rho @ G.pi)
    if abs(mass - 1.0) > MASS_TOL:
        raise ValueError(
            f"{name} is not a probability density with respect to G.pi: sum(rho * G.pi) = {mass:.6g}, expected 1. "
            "Densities are relative to pi; a probability vector mu corresponds to the density mu / G.pi."
        )
    return rho


def _check_refs(G: MarkovGraph, refs) -> list[np.ndarray]:
    if isinstance(refs, np.ndarray) and refs.ndim != 1:
        raise ValueError(
            f"refs must be a sequence of densities, each of shape ({G.n},); got a {refs.ndim}-D array of shape "
            f"{refs.shape}. Pass list(A) for references in the rows of A, or list(A.T) for columns."
        )
    refs = [_check_density(G, r, f"refs[{i}]") for i, r in enumerate(refs)]
    if not refs:
        raise ValueError("refs must contain at least one density")
    return refs


def _check_weights(lam, count: int) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    if lam.shape != (count,):
        raise ValueError(f"lam must have one weight per reference ({count}), got shape {lam.shape}")
    if not np.all(np.isfinite(lam)) or lam.min() < 0:
        raise ValueError(f"lam must be finite and nonnegative, got {lam.tolist()}")
    if abs(lam.sum() - 1.0) > 1e-8:
        raise ValueError(f"lam must sum to 1 (barycentric weights), got sum {lam.sum():.6g}")
    return lam


def _require_sinkhorn_kwargs(what: str, cost, epsilon, G: MarkovGraph):
    if cost is None:
        raise ValueError(f"{what}(method='sinkhorn') requires cost= (see ground_cost)")
    if epsilon is None:
        raise ValueError(f"{what}(method='sinkhorn') requires epsilon=")
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (G.n, G.n):
        raise ValueError(f"cost must be {G.n}x{G.n}, got {cost.shape}")
    return cost


def _barycenter_sinkhorn(G: MarkovGraph, refs, lam, *, cost=None, epsilon=None, iters: int = 256):
    cost = _require_sinkhorn_kwargs("barycenter", cost, epsilon, G)
    mu = np.column_stack([r * G.pi for r in refs])
    p = sinkhorn_barycenter(lam, mu, cost, epsilon, iters=iters)
    K = regularize_cost(cost, epsilon)
    active = np.flatnonzero(lam > 0)
    plans = [sinkhorn_plan(K, mu[:, i], p, iters=iters) for i in active]
    J = float(sum(lam[i] * np.sum(cost * P) for i, P in zip(active, plans)))
    marginal_errors = [float(np.abs(P.sum(axis=1) - mu[:, i]).sum()) for i, P in zip(active, plans)]
    info = {"cost": cost, "epsilon": epsilon, "iters": iters, "marginal_errors": marginal_errors, "method": "sinkhorn"}
    return p / G.pi, J, info


def _endpoint_plan(G: MarkovGraph, rhoA, rhoB, cost, epsilon, iters: int):
    """(W2, marginal error) of the entropic plan between the two endpoints.

    sinkhorn_plan ends on the update that fixes the second marginal, so the
    l1 error of the first marginal is the convergence measure."""
    muA = rhoA * G.pi
    P = sinkhorn_plan(regularize_cost(cost, epsilon), muA, rhoB * G.pi, iters=iters)
    return float(np.sum(cost * P)), float(np.abs(P.sum(axis=1) - muA).sum())


def _geodesic_sinkhorn(
    G: MarkovGraph, rhoA, rhoB, *, N: int = 10, cost=None, epsilon=None, iters: int = 256, tol: float = 1e-6
):
    t0 = time.perf_counter()
    cost = _require_sinkhorn_kwargs("geodesic", cost, epsilon, G)
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError(f"N must be an integer >= 1, got {N!r}")
    # Only the path is needed here, so skip _barycenter_sinkhorn's objective
    # and marginal-error plan solves (two per column).
    mu = np.column_stack([rhoA * G.pi, rhoB * G.pi])
    rho = build_geodesic(mu, cost, epsilon=epsilon, steps=N, iters=iters) / G.pi[:, np.newaxis]
    W2, marginal_error = _endpoint_plan(G, rhoA, rhoB, cost, epsilon, iters)
    status = "converged" if marginal_error <= tol else "iteration_limit"
    n_edges = G.E.shape[0]
    nan_E = np.full((n_edges, N), np.nan)
    nan_n = np.full(G.n, np.nan)
    return GeodesicSolution(W2, rho, nan_E, nan_E[:, 0].copy(), nan_n, nan_n.copy(), status, time.perf_counter() - t0)


def _transport_cost_sinkhorn(
    G: MarkovGraph, rhoA, rhoB, *, cost=None, epsilon=None, iters: int = 256, tol: float = 1e-6, N=None
):
    """W2 needs one plan solve; the geodesic would compute N + 1 barycenters
    to get it. N is accepted (it is a geodesic keyword) and has no effect."""
    cost = _require_sinkhorn_kwargs("transport_cost", cost, epsilon, G)
    W2, marginal_error = _endpoint_plan(G, rhoA, rhoB, cost, epsilon, iters)
    if marginal_error > tol:
        warnings.warn(
            f"transport_cost: the Sinkhorn plan has marginal error {marginal_error:.2e} > tol = {tol:.0e} after "
            f"{iters} iterations; the cost may not have converged. Raise iters or epsilon.",
            stacklevel=3,
        )
    return W2


def _analysis_sinkhorn(
    G: MarkovGraph,
    target,
    refs,
    *,
    cost=None,
    epsilon=None,
    iters: int = 256,
    alpha0=None,
    compute_condition: bool = False,
    return_system: bool = False,
):
    cost = _require_sinkhorn_kwargs("analysis", cost, epsilon, G)
    if compute_condition or return_system:
        raise ValueError(
            "analysis(method='sinkhorn') is not a Gram-matrix method; "
            "compute_condition/return_system are not available"
        )
    mu = np.column_stack([r * G.pi for r in refs])
    return simplex_regression(mu, target * G.pi, cost, epsilon, iters=iters, alpha0=alpha0)


# The SOCP backend is imported lazily: cvxpy is optional, and socp/ imports
# GeodesicSolution from this module.
def _geodesic_socp(G: MarkovGraph, rhoA, rhoB, **kwargs) -> GeodesicSolution:
    from graphtransport.socp import geodesic_socp

    return geodesic_socp(G, rhoA, rhoB, **kwargs)


def _barycenter_socp(G: MarkovGraph, refs, lam, **kwargs):
    from graphtransport.socp import barycenter_socp

    nu, J, geodesics = barycenter_socp(G, refs, lam, **kwargs)
    return nu, J, {"geodesics": geodesics, "method": "socp"}


def _analysis_socp(G: MarkovGraph, target, refs, **kwargs):
    from graphtransport.socp import analyze_socp

    return analyze_socp(G, target, refs, **kwargs)


class ShootingFallbackWarning(UserWarning):
    """method="shooting" could not take the data, and the SOCP was used instead.

    Raised by the default method when a density is not strictly positive, or
    when shooting fails on the data -- because a density is close to zero, or
    because Newton ran out of iterations (maxiters) on a hard transport. Filter it to
    silence the fallback, or escalate it with warnings.simplefilter("error",
    ShootingFallbackWarning); pass fallback=False to get the shooting error
    itself instead."""


class _NotInterior(ValueError):
    """A density shooting cannot take: at or below the positivity floor."""

    def __init__(self, message: str, summary: str):
        super().__init__(message)
        self.summary = summary


# Shooting needs every density strictly positive: its flow divides by the
# density. Checked here, per argument, so the error names what the caller
# passed and says what to do instead; log_map's own check would name its
# internal arguments. The floor is the caller's floor_rtol, the same one the
# solver will use -- with the default instead, a density between the two
# floors passed this check and failed inside log_map under the wrong name.
def _check_shooting_density(G: MarkovGraph, rho: np.ndarray, name: str, floor_rtol: float) -> np.ndarray:
    from graphtransport.shooting import rho_floor

    floor = rho_floor(G, rtol=floor_rtol)
    if rho.min() <= floor:
        summary = (
            f"{name} is not strictly positive (min {rho.min():.3e}, {int(np.sum(rho <= floor))} of {G.n} nodes at "
            "or below the positivity floor)"
        )
        raise _NotInterior(
            f"{summary}, and method='shooting' needs every density in the interior: its Hamiltonian flow divides "
            "by the density. Use method='socp', which is exact for densities supported on part of the graph, or "
            "shooting.log_map_mollified for an approximate distance.",
            summary,
        )
    # _check_density admits a mass error of MASS_TOL; the flow conserves mass
    # exactly, so give shooting the exact mass it needs.
    return rho / (rho @ G.pi)


class _explain_shooting_failure:
    """Re-raise a shooting failure with the context the caller needs.

    Data that passes _check_shooting_density can still defeat shooting, for
    two reasons this cannot tell apart. A density close to zero makes the flow
    stiff: on a 5x5 grid, a row of nodes at 1e-4 solves and one at 1e-5 does
    not (the line search fails). And a long transport through thin densities
    can need more damped Newton steps than maxiters allows: on a 10x10 grid,
    corner bumps over a floor of 1e-3 converge in 112 iterations, in Julia as
    here, against the default 50. So the message reports the failure and the
    smallest input density, and names both causes rather than guessing one.

    (An earlier version blamed "the boundary" for every failure, then "long
    transports making single shooting ill-conditioned". The long-transport
    failures were the forward-difference Jacobian's; with the exact one,
    Newton converges where Julia does.)"""

    def __init__(self, what: str, **densities):
        self.what, self.densities = what, densities

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        from graphtransport.shooting import PositivityFloorError, ShootingError

        if exc_type is None or not issubclass(exc_type, (ShootingError, PositivityFloorError)):
            return False
        name, rho = min(self.densities.items(), key=lambda item: float(np.min(item[1])))
        smallest = f"{float(np.min(rho)):.1e}, in {name}"
        first_clause = re.split(r"[.;] ", str(exc), maxsplit=1)[0]
        err = ShootingError(
            f"{self.what}(method='shooting') failed: {exc} The smallest input density is {smallest}. Shooting "
            "gets stiff as densities approach zero, where method='socp' has no such limit; a long transport "
            "through thin densities may instead just need more Newton iterations (maxiters=, default 50)."
        )
        err.summary = f"method='shooting' did not converge ({first_clause}; smallest input density {smallest})"
        raise err from exc


def _with_fallback(what: str, fallback: bool, run_shooting, run_socp):
    """run_shooting(), or -- if shooting cannot take the data and ``fallback``
    is set -- warn and run_socp() instead.

    Only the two data-driven failures fall back: a density at the boundary
    (_NotInterior) and a solve that fails on data that passed that check
    (ShootingError from _explain_shooting_failure). A bad argument or a
    solver bug still raises. Without cvxpy there is no SOCP to fall back to,
    and the shooting error is raised with a note saying so rather than being
    replaced by an ImportError that hides the real problem."""
    from graphtransport.shooting import ShootingError
    from graphtransport.solvers import cvxpy_available

    try:
        return run_shooting()
    except (_NotInterior, ShootingError) as exc:
        if not fallback:
            raise
        if not cvxpy_available():
            note = f"{exc} (The automatic fallback to method='socp' needs graphtransport[socp], which is not installed.)"
            if isinstance(exc, _NotInterior):
                raise _NotInterior(note, exc.summary) from exc
            raise ShootingError(note) from exc
        if isinstance(exc, _NotInterior):
            summary = f"method='shooting' cannot take this data: {exc.summary}"
        else:
            summary = getattr(exc, "summary", str(exc))
        warnings.warn(
            f"{what}: {summary}. Falling back to method='socp' with its "
            "default N=10 (time-discretisation error O(1/N)); pass method='socp' to choose N, or fallback=False "
            "to raise instead.",
            ShootingFallbackWarning,
            stacklevel=4,  # _with_fallback < the method wrapper < the entry point < the caller
        )
        return run_socp()


def _geodesic_shooting(G: MarkovGraph, rhoA, rhoB, *, fallback: bool = True, floor_rtol: float = 1e-6,
                       **kwargs) -> GeodesicSolution:
    from graphtransport.shooting import geodesic_shooting

    def run():
        a = _check_shooting_density(G, rhoA, "rhoA", floor_rtol)
        b = _check_shooting_density(G, rhoB, "rhoB", floor_rtol)
        with _explain_shooting_failure("geodesic", rhoA=a, rhoB=b):
            return geodesic_shooting(G, a, b, floor_rtol=floor_rtol, **kwargs)

    return _with_fallback("geodesic", fallback, run, lambda: _geodesic_socp(G, rhoA, rhoB))


def _transport_cost_shooting(G: MarkovGraph, rhoA, rhoB, *, fallback: bool = True, nsteps: int = 150,
                             tol: float = 1e-9, maxiters: int = 50, phi0_init=None, floor_rtol: float = 1e-6,
                             verbose: bool = False) -> float:
    """W2 from log_map alone: the path integration geodesic() adds is not
    needed. Takes geodesic's keywords, verbose included."""
    from graphtransport.shooting import log_map

    def run():
        a = _check_shooting_density(G, rhoA, "rhoA", floor_rtol)
        b = _check_shooting_density(G, rhoB, "rhoB", floor_rtol)
        with _explain_shooting_failure("transport_cost", rhoA=a, rhoB=b):
            return log_map(G, a, b, nsteps=nsteps, tol=tol, maxiters=maxiters, phi0_init=phi0_init,
                           floor_rtol=floor_rtol, verbose=verbose).W2  # fmt: skip

    return _with_fallback("transport_cost", fallback, run, lambda: _geodesic_socp(G, rhoA, rhoB).W2)


def _barycenter_shooting(G: MarkovGraph, refs, lam, *, fallback: bool = True, floor_rtol: float = 1e-6, **kwargs):
    from graphtransport.shooting import barycenter_shooting

    def run():
        # a reference at weight zero is never log-mapped, so it may touch the boundary
        checked = [
            _check_shooting_density(G, r, f"refs[{i}]", floor_rtol) if lam[i] > 0 else r for i, r in enumerate(refs)
        ]
        with _explain_shooting_failure("barycenter", **{f"refs[{i}]": r for i, r in enumerate(checked) if lam[i] > 0}):
            nu, J, info = barycenter_shooting(G, checked, lam, floor_rtol=floor_rtol, **kwargs)
        return nu, J, {**info, "method": "shooting"}

    def socp():
        nu, J, info = _barycenter_socp(G, refs, lam)
        return nu, J, {**info, "method": "socp"}

    return _with_fallback("barycenter", fallback, run, socp)


# analysis keywords both methods understand, forwarded on a fallback
_SHARED_ANALYSIS_KEYWORDS = ("compute_condition", "return_system", "qp_method", "qp_solver")


def _analysis_shooting(G: MarkovGraph, target, refs, *, fallback: bool = True, floor_rtol: float = 1e-6, **kwargs):
    from graphtransport.shooting import analyze_shooting

    def run():
        t = _check_shooting_density(G, target, "target", floor_rtol)
        checked = [_check_shooting_density(G, r, f"refs[{i}]", floor_rtol) for i, r in enumerate(refs)]
        with _explain_shooting_failure("analysis", target=t, **{f"refs[{i}]": r for i, r in enumerate(checked)}):
            return analyze_shooting(G, t, checked, floor_rtol=floor_rtol, **kwargs)

    shared = {k: v for k, v in kwargs.items() if k in _SHARED_ANALYSIS_KEYWORDS}
    return _with_fallback("analysis", fallback, run, lambda: _analysis_socp(G, target, refs, **shared))


# ----- torch tensors -----
#
# geodesic and transport_cost accept torch tensors for rhoA and rhoB and
# return tensors. With method="shooting" they are differentiable end to end
# (shooting.differentiable: exact gradients of the discrete solve, by the
# implicit function theorem). The other methods, and shooting's fallback to
# the SOCP, are not differentiable: with inputs that require grad they raise
# rather than hand back an answer that silently carries no gradient; without
# grad they run as usual and convert their outputs.


def _is_torch(*values) -> bool:
    # without torch loaded no value can be a tensor; checking first keeps a
    # numpy-only session (the SOCP, say) from importing torch here
    torch = sys.modules.get("torch")
    return torch is not None and any(isinstance(v, torch.Tensor) for v in values)


def _as_float64_tensor(value, name: str):
    import torch

    t = torch.as_tensor(value)
    if t.device.type != "cpu":
        raise ValueError(f"{name} is on {t.device}; graphtransport computes on the CPU (move it with .cpu())")
    return t.to(torch.float64)


def _solution_to_torch(sol: GeodesicSolution) -> GeodesicSolution:
    import torch

    as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=torch.float64)  # noqa: E731
    return GeodesicSolution(as_t(sol.W2), as_t(sol.rho), as_t(sol.m), as_t(sol.m0), as_t(sol.phi0),
                            as_t(sol.phi1), sol.status, sol.solvetime, sol.ref_index)  # fmt: skip


def _torch_geodesic(G: MarkovGraph, rhoA, rhoB, method: str, kwargs: dict, what: str, cost_only: bool):
    """geodesic / transport_cost for torch inputs: a GeodesicSolution of
    tensors, or (cost_only) the tensor W2."""
    import torch

    from graphtransport.shooting import ShootingError
    from graphtransport.shooting.differentiable import geodesic_shooting_torch, transport_cost_shooting_torch

    A, B = _as_float64_tensor(rhoA, "rhoA"), _as_float64_tensor(rhoB, "rhoB")
    a, b = _check_density(G, A.detach().numpy(), "rhoA"), _check_density(G, B.detach().numpy(), "rhoB")
    needs_grad = A.requires_grad or B.requires_grad

    if method != "shooting":
        if needs_grad:
            raise TypeError(
                f"{what}: gradients are available for method='shooting' only; method={method!r} is not "
                "differentiable. Detach the inputs (or pass numpy arrays) to use it."
            )
        if cost_only and method in TRANSPORT_COST_METHODS:
            return torch.as_tensor(float(TRANSPORT_COST_METHODS[method](G, a, b, **kwargs)), dtype=torch.float64)
        sol = _solution_to_torch(GEODESIC_METHODS[method](G, a, b, **kwargs))
        return sol.W2 if cost_only else sol

    if needs_grad and not G.mean.has_torch_autodiff:
        # the numpy-backed defaults carry no autograd graph through theta: the
        # gradient would come back silently incomplete
        raise TypeError(
            f"{what}: gradients need a mean with torch versions of theta and partial_s; {G.mean!r} has numpy "
            "methods only (enough for shooting without gradients). Implement torch_theta and torch_partial_s, "
            "or detach the inputs."
        )
    fallback = kwargs.pop("fallback", True)
    floor_rtol = kwargs.pop("floor_rtol", 1e-6)
    solve = transport_cost_shooting_torch if cost_only else geodesic_shooting_torch

    def run():
        _check_shooting_density(G, a, "rhoA", floor_rtol)
        _check_shooting_density(G, b, "rhoB", floor_rtol)
        with _explain_shooting_failure(what, rhoA=a, rhoB=b):
            return solve(G, A, B, floor_rtol=floor_rtol, **kwargs)

    if needs_grad:
        try:
            return run()
        except (_NotInterior, ShootingError) as exc:
            note = (f"{exc} (No fallback to method='socp': the inputs require grad, and the SOCP is not "
                    "differentiable.)")  # fmt: skip
            if isinstance(exc, _NotInterior):
                raise _NotInterior(note, exc.summary) from exc
            raise ShootingError(note) from exc

    def socp():
        sol = _solution_to_torch(_geodesic_socp(G, a, b))
        return sol.W2 if cost_only else sol

    return _with_fallback(what, fallback, run, socp)


GEODESIC_METHODS = {"shooting": _geodesic_shooting, "socp": _geodesic_socp, "sinkhorn": _geodesic_sinkhorn}
BARYCENTER_METHODS = {"shooting": _barycenter_shooting, "socp": _barycenter_socp, "sinkhorn": _barycenter_sinkhorn}
ANALYSIS_METHODS = {"shooting": _analysis_shooting, "socp": _analysis_socp, "sinkhorn": _analysis_sinkhorn}
# Methods that can produce W2 without building the whole geodesic.
TRANSPORT_COST_METHODS = {"shooting": _transport_cost_shooting, "sinkhorn": _transport_cost_sinkhorn}


def geodesic(G: MarkovGraph, rhoA, rhoB, *, method: str = DEFAULT_METHOD, **kwargs) -> GeodesicSolution:
    """The discrete transport geodesic between densities rhoA and rhoB on G.

    method="shooting" (default): Newton shooting on the Hamiltonian flow
    (shooting.geodesic_shooting), then the flow integrated for the path.
    Exact in time up to RK4 truncation; both endpoints must be strictly
    positive. If one is not, or shooting fails on them (densities close to
    zero, or a transport that needs more than maxiters Newton steps), this warns
    (ShootingFallbackWarning) and returns method="socp"'s answer at its
    default N=10 instead -- or raises, with ``fallback=False`` or
    without cvxpy installed. Keywords: ``fallback``, ``nsteps`` (integrator steps, default 150; rho
    then has nsteps + 1 columns), ``tol``, ``maxiters``, ``phi0_init``,
    ``verbose``. Honours every AdmissibleMean, including the exact
    LogarithmicMean. phi0 and phi1 use the SOCP's W2-gradient convention, so
    the two methods' potentials are directly comparable; status is
    "converged".

    method="socp": a single second-order-cone program
    (socp.geodesic_socp). Handles any densities, including boundary-supported
    ones; time-discretisation error O(1/N). Keywords: ``N``, ``solver``,
    ``check``, ``verbose``, plus solver options. Honours every conic
    AdmissibleMean in G.mean (QuadLogMean for the logarithmic mean).

    method="sinkhorn": the entropic displacement interpolation for a ground
    cost -- the path is the entropic barycenter of the two endpoints at
    weights (1-t, t) for t = 0, 1/N, ..., 1, a different object from the
    discrete transport geodesic of the other methods. Requires ``cost`` (see
    ground_cost) and ``epsilon``; keywords ``N`` (default 10) and ``iters``
    (Sinkhorn budget, default 256). W2 is the entropic transport cost
    <cost, P> of the plan between the endpoints (no entropy term), and
    status is "converged" when that plan meets its marginals to ``tol``
    (l1, default 1e-6), otherwise "iteration_limit". The
    path's end columns are the blurred endpoints the Sinkhorn barycenter
    returns, not rhoA/rhoB exactly; m, phi0, phi1 are NaN-filled.

    Torch tensors: rhoA and rhoB may be torch tensors (computed in float64 on
    the CPU), and the solution's fields are then tensors. With
    method="shooting" all of them -- W2, the path, the momenta, the
    potentials -- are differentiable with respect to both endpoints, exactly
    for the discrete solve (shooting.differentiable). The other methods are
    not differentiable: they raise if an input requires grad, and otherwise
    return tensors with no graph. Shooting's fallback to the SOCP is disabled
    when an input requires grad; shooting raises instead. Gradients are first
    order only: differentiating a gradient again (create_graph=True, for a
    gradient penalty or a Hessian-vector product) raises.
    """
    _check_method(method, GEODESIC_METHODS, "geodesic")
    _check_kwargs(method, kwargs, "geodesic")
    if _is_torch(rhoA, rhoB):
        return _torch_geodesic(G, rhoA, rhoB, method, dict(kwargs), "geodesic", cost_only=False)
    rhoA, rhoB = _check_density(G, rhoA, "rhoA"), _check_density(G, rhoB, "rhoB")
    return GEODESIC_METHODS[method](G, rhoA, rhoB, **kwargs)


def transport_cost(G: MarkovGraph, rhoA, rhoB, *, method: str = DEFAULT_METHOD, **kwargs) -> float:
    """The discrete transport distance W(rhoA, rhoB) (not squared):
    sqrt(geodesic(...).W2). See geodesic for the methods and keywords.

    method="shooting" (default) needs only the log map, not the path.
    method="sinkhorn" solves only the endpoint plan rather than the whole
    path, and warns if that plan has not converged (there is no status to
    return).

    Takes torch tensors as geodesic does, returning a 0-d tensor. W is not
    differentiable where W = 0 (rhoA == rhoB); its gradient there is 0, the
    subgradient at W's minimum and torch.linalg.norm's convention, rather than
    the nan the square root's infinite derivative would give. Gradients are
    first order only (see geodesic)."""
    _check_method(method, GEODESIC_METHODS, "transport_cost")
    _check_kwargs(method, kwargs, "transport_cost")
    if _is_torch(rhoA, rhoB):
        return _sqrt_zero_subgradient(_torch_geodesic(G, rhoA, rhoB, method, dict(kwargs), "transport_cost", cost_only=True))
    if method in TRANSPORT_COST_METHODS:
        rhoA, rhoB = _check_density(G, rhoA, "rhoA"), _check_density(G, rhoB, "rhoB")
        return float(np.sqrt(TRANSPORT_COST_METHODS[method](G, rhoA, rhoB, **kwargs)))
    return float(np.sqrt(geodesic(G, rhoA, rhoB, method=method, **kwargs).W2))


def _sqrt_zero_subgradient(W2):
    """sqrt(W2) with gradient 0 where W2 == 0, torch.linalg.norm's convention:
    sqrt's own derivative there is infinite, and inf * 0 gives a nan that
    would poison a training loop whose prediction matches its target. Zero is
    also the right subgradient: W >= 0 is smallest at rhoA == rhoB. The inner
    sqrt sees a harmless 1 where W2 == 0, so its gradient never forms."""
    import torch

    positive = W2 > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, W2, torch.ones_like(W2))), torch.zeros_like(W2))


def _reject_torch(what: str, *values) -> None:
    if _is_torch(*values):
        raise TypeError(
            f"{what} does not take torch tensors yet (differentiable {what} is planned); pass numpy arrays. "
            "geodesic and transport_cost take tensors, with gradients for method='shooting'."
        )


def barycenter(G: MarkovGraph, refs, lam, *, method: str = DEFAULT_METHOD, **kwargs):
    """The discrete transport barycenter of the reference densities ``refs``
    with weights ``lam``: the minimiser of J(nu) = sum_i lam_i W^2(refs_i, nu).

    method="shooting" (default): intrinsic gradient descent with exact-in-
    time geodesics (shooting.barycenter_shooting). Every reference with
    lam_i > 0 must be strictly positive. info = {"iters", "status",
    "J_hist", "grad_hist", "h"}; status is "converged", "stalled" (at the
    precision of the log maps) or "maxiters" (which warns). Keywords: ``h``,
    ``maxiters``, ``tol`` (gradient norm, default 1e-5), ``ftol``,
    ``log_tol``, ``nsteps``, ``init``, ``verbose``. First order, so it
    converges linearly; method="socp" solves the same problem to its global
    optimum and is the certificate.

    method="socp": one joint second-order-cone program
    (socp.barycenter_socp), solved to its global optimum.
    info = {"geodesics": [...]} holds one GeodesicSolution per reference
    with lam_i > 0, each carrying the index of its reference as
    ``ref_index``: references at weight zero are not solved, so the list can
    be shorter than ``refs`` and ``geodesics[k]`` need not be the geodesic of
    ``refs[k]``. Keywords: ``N``, ``solver``, ``check``, ``verbose``.

    method="sinkhorn": the entropically regularised Wasserstein barycenter
    for a ground cost (Benamou et al. 2015; Bonneel, Peyré & Cuturi 2016).
    This is a different object from the discrete transport barycenter of
    the other methods: it depends on ``cost`` (see ground_cost) and
    ``epsilon``, both required, and even for a single reference returns a
    blurred copy of it. ``iters`` (default 256) is the Sinkhorn budget.
    J = sum_i lam_i <cost, P_i> over the entropic plans (no entropy term);
    info = {cost, epsilon, iters, marginal_errors}.

    Returns (nu, J, info).
    """
    _check_method(method, BARYCENTER_METHODS, "barycenter")
    _check_kwargs(method, kwargs, "barycenter")
    _reject_torch("barycenter", lam, *(refs if isinstance(refs, (list, tuple)) else [refs]))
    refs = _check_refs(G, refs)
    lam = _check_weights(lam, len(refs))
    return BARYCENTER_METHODS[method](G, refs, lam, **kwargs)


def analysis(G: MarkovGraph, target, refs, *, method: str = DEFAULT_METHOD, **kwargs) -> np.ndarray:
    """Recover the barycentric coordinates of ``target`` with respect to the
    reference densities ``refs``.

    method="shooting" (default): shooting.analyze_shooting -- potentials from
    log_map at ``target``, then the same Gram matrix and simplex QP as the
    SOCP. ``target`` and every reference must be strictly positive.
    Keywords: ``nsteps``, ``tol``, ``phi0_inits``, ``compute_condition``,
    ``return_system``, ``qp_method``, ``qp_solver``.

    A barycenter is recovered to solver tolerance only by the method that
    synthesised it; the others recover it to their discretisation error,
    since each checks stationarity in its own discrete convention.

    method="socp": socp.analyze_socp -- geodesic SOCPs from the
    target to each reference, the Gram matrix of their endpoint potentials
    (or initial momenta, ``convention="momentum"``), and the simplex QP.
    Keywords: ``N``, ``solver``, ``convention``, ``compute_condition``,
    ``return_system`` (also return the Gram matrix A).

    method="sinkhorn": Wasserstein barycentric coordinates for a ground cost
    (Bonneel, Peyré & Cuturi 2016; simplex_regression): L-BFGS over the
    simplex on 1/2 ||P(lam) - target||^2, differentiated through the
    Sinkhorn iterations. Not a Gram-matrix method, so ``return_system`` and
    ``compute_condition`` raise. Requires ``cost`` and ``epsilon``; keywords
    ``iters`` (default 256) and ``alpha0`` (initial pre-softmax point,
    default 0 = uniform weights). Recovers exactly the weights of a
    barycenter synthesised by barycenter(method="sinkhorn") with the same
    cost, epsilon and iters.

    Returns lam_hat on the simplex.
    """
    _check_method(method, ANALYSIS_METHODS, "analysis")
    _check_kwargs(method, kwargs, "analysis")
    _reject_torch("analysis", target, *(refs if isinstance(refs, (list, tuple)) else [refs]))
    target, refs = _check_density(G, target, "target"), _check_refs(G, refs)
    return ANALYSIS_METHODS[method](G, target, refs, **kwargs)
