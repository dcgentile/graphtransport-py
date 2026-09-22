"""Unified entry points, ported from GraphTransportation.jl's API.jl.

One public function per task with a ``method`` keyword selecting the
numerical algorithm: ``"socp"`` (default, as in Julia; needs the optional
``graphtransport[socp]`` extra) or ``"sinkhorn"``. The shooting method
plugs into the same dispatch tables in a later step.

Densities: as in the Julia package, ``rhoA``, ``rhoB``, ``refs``, ``target``
and returned barycenters are densities with respect to the graph's
stationary distribution ``G.pi`` (``sum(rho * G.pi) == 1``), not probability
vectors. The Sinkhorn method converts to probability vectors internally.
"""

from __future__ import annotations

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


def _check_method(method: str, table: dict, what: str):
    if method not in table:
        raise ValueError(f"{what}: method must be one of {tuple(table)}, got {method!r}")


# Keywords only method="sinkhorn" understands. Every other method passes its
# unknown keywords to a solver, where they surface as e.g. "Clarabel:
# unrecognized solver setting 'cost'" -- an error that names everything except
# the thing to change. Since method="socp" is the default, code written for the
# Sinkhorn API hits that on its first call, so the check is worth doing here.
SINKHORN_ONLY = ("cost", "epsilon", "iters", "tol", "alpha0")


def _check_kwargs(method: str, kwargs: dict, what: str) -> None:
    if method == "sinkhorn":
        return
    stray = [k for k in SINKHORN_ONLY if k in kwargs]
    if stray:
        raise TypeError(
            f"{what}: {', '.join(stray)} "
            f"{'is a' if len(stray) == 1 else 'are'} method='sinkhorn' keyword"
            f"{'' if len(stray) == 1 else 's'}, but method={method!r}. "
            "Pass method='sinkhorn' to use them."
        )


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
    info = {"cost": cost, "epsilon": epsilon, "iters": iters, "marginal_errors": marginal_errors}
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
    return nu, J, {"geodesics": geodesics}


def _analysis_socp(G: MarkovGraph, target, refs, **kwargs):
    from graphtransport.socp import analyze_socp

    return analyze_socp(G, target, refs, **kwargs)


GEODESIC_METHODS = {"socp": _geodesic_socp, "sinkhorn": _geodesic_sinkhorn}
BARYCENTER_METHODS = {"socp": _barycenter_socp, "sinkhorn": _barycenter_sinkhorn}
ANALYSIS_METHODS = {"socp": _analysis_socp, "sinkhorn": _analysis_sinkhorn}
# Methods that can produce W2 without building the whole geodesic.
TRANSPORT_COST_METHODS = {"sinkhorn": _transport_cost_sinkhorn}


def geodesic(G: MarkovGraph, rhoA, rhoB, *, method: str = "socp", **kwargs) -> GeodesicSolution:
    """The discrete transport geodesic between densities rhoA and rhoB on G.

    method="socp" (default): a single second-order-cone program
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
    """
    _check_method(method, GEODESIC_METHODS, "geodesic")
    _check_kwargs(method, kwargs, "geodesic")
    rhoA, rhoB = _check_density(G, rhoA, "rhoA"), _check_density(G, rhoB, "rhoB")
    return GEODESIC_METHODS[method](G, rhoA, rhoB, **kwargs)


def transport_cost(G: MarkovGraph, rhoA, rhoB, *, method: str = "socp", **kwargs) -> float:
    """The discrete transport distance W(rhoA, rhoB) (not squared):
    sqrt(geodesic(...).W2). See geodesic for the methods and keywords.

    method="sinkhorn" solves only the endpoint plan rather than the whole
    path, and warns if that plan has not converged (there is no status to
    return)."""
    _check_method(method, GEODESIC_METHODS, "transport_cost")
    _check_kwargs(method, kwargs, "transport_cost")
    if method in TRANSPORT_COST_METHODS:
        rhoA, rhoB = _check_density(G, rhoA, "rhoA"), _check_density(G, rhoB, "rhoB")
        return float(np.sqrt(TRANSPORT_COST_METHODS[method](G, rhoA, rhoB, **kwargs)))
    return float(np.sqrt(geodesic(G, rhoA, rhoB, method=method, **kwargs).W2))


def barycenter(G: MarkovGraph, refs, lam, *, method: str = "socp", **kwargs):
    """The discrete transport barycenter of the reference densities ``refs``
    with weights ``lam``: the minimiser of J(nu) = sum_i lam_i W^2(refs_i, nu).

    method="socp" (default): one joint second-order-cone program
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
    refs = _check_refs(G, refs)
    lam = _check_weights(lam, len(refs))
    return BARYCENTER_METHODS[method](G, refs, lam, **kwargs)


def analysis(G: MarkovGraph, target, refs, *, method: str = "socp", **kwargs) -> np.ndarray:
    """Recover the barycentric coordinates of ``target`` with respect to the
    reference densities ``refs``.

    method="socp" (default): socp.analyze_socp -- geodesic SOCPs from the
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
    target, refs = _check_density(G, target, "target"), _check_refs(G, refs)
    return ANALYSIS_METHODS[method](G, target, refs, **kwargs)
