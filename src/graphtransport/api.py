"""Unified entry points, ported from GraphTransportation.jl's API.jl.

One public function per task with a ``method`` keyword selecting the
numerical algorithm. Only ``"sinkhorn"`` is registered so far; the SOCP and
shooting methods plug into the same dispatch tables in later steps.

Densities: as in the Julia package, ``rhoA``, ``rhoB``, ``refs``, ``target``
and returned barycenters are densities with respect to the graph's
stationary distribution ``G.pi`` (``sum(rho * G.pi) == 1``), not probability
vectors. The Sinkhorn method converts to probability vectors internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from graphtransport.graph import MarkovGraph
from graphtransport.sinkhorn.core import regularize_cost, simplex_regression, sinkhorn_barycenter, sinkhorn_plan


@dataclass
class GeodesicSolution:
    """A discrete transport geodesic.

    W2: the squared transport distance. rho: the (n, steps+1) density path.
    m: the (|E|, steps) momentum path and m0 its first column. phi0, phi1:
    endpoint potentials (gradients of W2 with respect to each endpoint).
    status: solver status. solvetime: wall-clock seconds.

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


def _check_method(method: str, table: dict, what: str):
    if method not in table:
        raise ValueError(f"{what}: method must be one of {tuple(table)}, got {method!r}")


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
    lam = np.asarray(lam, dtype=float)
    mu = np.column_stack([np.asarray(r, dtype=float) * G.pi for r in refs])
    p = sinkhorn_barycenter(lam, mu, cost, epsilon, iters=iters)
    K = regularize_cost(cost, epsilon)
    active = np.flatnonzero(lam > 0)
    plans = [sinkhorn_plan(K, mu[:, i], p, iters=iters) for i in active]
    J = float(sum(lam[i] * np.sum(cost * P) for i, P in zip(active, plans)))
    marginal_errors = [float(np.abs(P.sum(axis=1) - mu[:, i]).sum()) for i, P in zip(active, plans)]
    info = {"cost": cost, "epsilon": epsilon, "iters": iters, "marginal_errors": marginal_errors}
    return p / G.pi, J, info


def _geodesic_sinkhorn(G: MarkovGraph, rhoA, rhoB, *, N: int = 10, cost=None, epsilon=None, iters: int = 256):
    t0 = time.perf_counter()
    cost = _require_sinkhorn_kwargs("geodesic", cost, epsilon, G)
    rhoA = np.asarray(rhoA, dtype=float)
    rhoB = np.asarray(rhoB, dtype=float)
    rho = np.empty((G.n, N + 1))
    for k, t in enumerate(np.linspace(0.0, 1.0, N + 1)):
        rho[:, k] = _barycenter_sinkhorn(G, [rhoA, rhoB], [1 - t, t], cost=cost, epsilon=epsilon, iters=iters)[0]
    K = regularize_cost(cost, epsilon)
    P = sinkhorn_plan(K, rhoA * G.pi, rhoB * G.pi, iters=iters)
    W2 = float(np.sum(cost * P))
    n_edges = G.E.shape[0]
    nan_E = np.full((n_edges, N), np.nan)
    nan_n = np.full(G.n, np.nan)
    return GeodesicSolution(W2, rho, nan_E, nan_E[:, 0].copy(), nan_n, nan_n.copy(), "converged", time.perf_counter() - t0)


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
    mu = np.column_stack([np.asarray(r, dtype=float) * G.pi for r in refs])
    return simplex_regression(mu, np.asarray(target, dtype=float) * G.pi, cost, epsilon, iters=iters, alpha0=alpha0)


GEODESIC_METHODS = {"sinkhorn": _geodesic_sinkhorn}
BARYCENTER_METHODS = {"sinkhorn": _barycenter_sinkhorn}
ANALYSIS_METHODS = {"sinkhorn": _analysis_sinkhorn}


def geodesic(G: MarkovGraph, rhoA, rhoB, *, method: str = "sinkhorn", **kwargs) -> GeodesicSolution:
    """The discrete transport geodesic between densities rhoA and rhoB on G.

    method="sinkhorn": the entropic displacement interpolation for a ground
    cost -- the path is the entropic barycenter of the two endpoints at
    weights (1-t, t) for t = 0, 1/N, ..., 1, a different object from the
    discrete transport geodesic of the other methods. Requires ``cost`` (see
    ground_cost) and ``epsilon``; keywords ``N`` (default 10) and ``iters``
    (Sinkhorn budget, default 256). W2 is the entropic transport cost
    <cost, P> of the plan between the endpoints (no entropy term). The
    path's end columns are the blurred endpoints the Sinkhorn barycenter
    returns, not rhoA/rhoB exactly; m, phi0, phi1 are NaN-filled.
    """
    _check_method(method, GEODESIC_METHODS, "geodesic")
    return GEODESIC_METHODS[method](G, rhoA, rhoB, **kwargs)


def transport_cost(G: MarkovGraph, rhoA, rhoB, **kwargs) -> float:
    """The discrete transport distance W(rhoA, rhoB) (not squared):
    sqrt(geodesic(...).W2). See geodesic for the methods and keywords."""
    return float(np.sqrt(geodesic(G, rhoA, rhoB, **kwargs).W2))


def barycenter(G: MarkovGraph, refs, lam, *, method: str = "sinkhorn", **kwargs):
    """The discrete transport barycenter of the reference densities ``refs``
    with weights ``lam``: the minimiser of J(nu) = sum_i lam_i W^2(refs_i, nu).

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
    return BARYCENTER_METHODS[method](G, refs, lam, **kwargs)


def analysis(G: MarkovGraph, target, refs, *, method: str = "sinkhorn", **kwargs) -> np.ndarray:
    """Recover the barycentric coordinates of ``target`` with respect to the
    reference densities ``refs``.

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
    return ANALYSIS_METHODS[method](G, target, refs, **kwargs)
