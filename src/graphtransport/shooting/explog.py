"""The Riemannian exponential and logarithmic maps, by shooting.

Ported from GraphTransportation.jl's shooting/ExpLog.jl. `exp_map` integrates
the Hamiltonian flow from a base density and a tangent (a potential or a
momentum); `log_map` inverts it by single shooting -- Newton on the endpoint
residual -- which is what turns the flow into a geodesic solver between two
given densities. `analyze_shooting` and `log_map_mollified` are built on it.

Scope. Everything here requires strictly positive densities; data supported
on part of the graph belongs to the SOCP (exact) or `log_map_mollified`
(approximate). The module is written for graphs of at most a few hundred
nodes: `log_map`'s cost is dominated by its Jacobian, n full trajectories per
Newton step. A Jacobian-free Newton-Krylov variant is the natural next step
if larger graphs are needed.

Departure from the Julia original: Julia differentiates the flow map with
ForwardDiff. Here the Jacobian is a forward finite difference, computed as a
single *batched* integration of all n perturbed trajectories with a shared
step schedule, so every difference quotient is taken on the same branch of
the (piecewise-smooth, because of the positivity bisection) discrete flow map
-- the property that made ForwardDiff's derivative meaningful. The Jacobian
is accurate to roughly 1e-7, which leaves Newton's convergence to 1e-9
essentially unchanged: the error in the Newton step is that factor times the
residual.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from graphtransport.graph import MarkovGraph, graph_gradient, metric_tensor
from graphtransport.shooting.hamiltonian import (
    PositivityFloorError,
    _check_interior,
    _integrate_end,
    hamiltonian,
    integrate_hamiltonian,
    rho_floor,
)

logger = logging.getLogger(__name__)

# Relative step of the finite-difference Jacobian. The discrete flow map is
# evaluated to ~1e-13 after 150 RK4 steps, so the step balancing truncation
# (~h) against that noise (~1e-13 / h) is about 3e-7.
_FD_STEP = 3e-7


class ShootingError(RuntimeError):
    """log_map could not solve the shooting problem: Newton did not converge,
    the line search failed, or no admissible initial potential was found.

    log_map_mollified catches this (and PositivityFloorError) to skip a
    mollification level; nothing broader."""


@dataclass
class LogMapResult:
    """The Riemannian logarithm of ``target`` at ``nu``.

    phi0: the initial potential, gauge <phi0, 1>_pi = 0. m0 = theta(nu) *
    grad phi0, the initial momentum. W2 = 2 H(nu, phi0), the squared transport
    distance. iters: Newton iterations taken (0 if the initial guess already
    met ``tol``). residual: the final ||rho(1) - target||_pi.
    """

    phi0: np.ndarray
    m0: np.ndarray
    W2: float
    iters: int
    residual: float


@dataclass
class MollifiedLogMapResult:
    """log_map_mollified's extrapolation to epsilon -> 0.

    W, W2: the extrapolated distance and its square. phi0, m0: from the
    smallest solved epsilon, at the *mollified* base point, so only
    approximately a tangent at nu. fit: (W0, a) in W(eps) ~ W0 + a sqrt(eps).
    epsilons, Ws: the levels actually solved and their raw distances.
    approximate: always True -- the SOCP is the exact method for such data.
    """

    W2: float
    W: float
    phi0: np.ndarray
    m0: np.ndarray
    fit: tuple
    epsilons: np.ndarray
    Ws: np.ndarray
    approximate: bool = True


def _check_mass(G: MarkovGraph, rho: np.ndarray, what: str) -> None:
    mass = float(rho @ G.pi)
    if abs(mass - 1.0) > 1e-8:
        raise ValueError(f"{what} must be a probability density with respect to G.pi: sum(rho * G.pi) = {mass:.10g}")


def _check_boundary_density(G: MarkovGraph, rho, what: str) -> np.ndarray:
    """A probability density that may have zeros -- what log_map_mollified
    exists to take -- but not negative or non-finite entries."""
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (G.n,):
        raise ValueError(f"{what} must have shape ({G.n},), got {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"{what} has non-finite entries")
    floor = -1e-9 * max(1.0, float(np.abs(rho).max()))  # the round-off allowance of api._check_density
    if rho.min() < floor:
        raise ValueError(f"{what} has negative entries (min {rho.min():.3e}); a density must be nonnegative")
    rho = np.maximum(rho, 0.0)
    _check_mass(G, rho, what)
    return rho


def weighted_laplacian(G: MarkovGraph, nu) -> sparse.csr_matrix:
    """The n x n weighted graph Laplacian L(nu) = grad^T Diag(kappa theta(nu)) grad,

        (L phi)(x) = sum_y kappa_xy theta(nu_x, nu_y) (phi(x) - phi(y)).

    Symmetric positive semidefinite, with kernel the constants on a connected
    graph, and tied to the Hamiltonian flow by pi * rho_dot = L(nu) phi.
    """
    w = G.kappa * metric_tensor(G, nu)
    x, y = G.E[:, 0], G.E[:, 1]
    rows = np.concatenate([x, y, x, y])
    cols = np.concatenate([x, y, y, x])
    vals = np.concatenate([w, w, -w, -w])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(G.n, G.n))


def solve_weighted_laplacian(G: MarkovGraph, nu, b) -> np.ndarray:
    """Solve L(nu) phi = b for the unique solution with <phi, 1>_pi = 0.

    b must be orthogonal to the constants (sum(b) == 0), as every right-hand
    side arising here is: pi * (target - nu) for two probability densities,
    or grad^T(kappa m) for any momentum. Implemented as the rank-one
    regularisation (L + pi pi^T) phi = b, which for such b has the same
    solution and enforces the gauge by itself (summing both sides gives
    (1^T pi)(pi^T phi) = 0).
    """
    b = np.asarray(b, dtype=float)
    if b.shape != (G.n,):
        raise ValueError(f"b must have shape ({G.n},), got {b.shape}")
    if abs(b.sum()) > 1e-8 * max(1.0, float(np.abs(b).max())):
        raise ValueError(f"right-hand side must be orthogonal to the constants, got sum(b) = {b.sum():.3e}")
    A = weighted_laplacian(G, nu).toarray() + np.outer(G.pi, G.pi)
    try:
        return cho_solve(cho_factor(A), b)
    except LinAlgError as exc:
        raise ValueError(
            "the weighted Laplacian is singular beyond its constant kernel; the graph is disconnected, "
            "or theta(nu) vanishes on a cut"
        ) from exc


def momentum_to_potential(G: MarkovGraph, nu, m) -> np.ndarray:
    """The potential phi with m = theta(nu) * grad phi, gauge <phi, 1>_pi = 0,
    by solving L(nu) phi = grad^T(kappa m). If m is not exactly a gradient
    field this is the theta(nu)-weighted least-squares projection: the
    potential of the gradient part of m in the Hodge sense."""
    m = np.asarray(m, dtype=float)
    if m.shape != (G.E.shape[0],):
        raise ValueError(f"m must have one entry per edge, shape ({G.E.shape[0]},), got {m.shape}")
    b = np.zeros(G.n)
    np.add.at(b, G.E[:, 0], G.kappa * m)
    np.add.at(b, G.E[:, 1], -G.kappa * m)
    return solve_weighted_laplacian(G, nu, b)


def _gauge(G: MarkovGraph, phi: np.ndarray) -> np.ndarray:
    """phi shifted to <phi, 1>_pi = 0 (the flow's rho does not depend on the shift)."""
    return phi - (phi @ G.pi) / G.pi.sum()


def exp_map(G: MarkovGraph, nu, tangent, *, t: float = 1.0, nsteps: int = 150, kind: str = "auto",
            floor_rtol: float = 1e-6) -> np.ndarray:
    """The Riemannian exponential map at ``nu``: integrate the Hamiltonian flow
    from (nu, phi0) for time ``t`` and return the endpoint density.

    ``tangent`` is a potential phi0 (length n, ``kind="potential"``) or a
    momentum m0 (length |E|, ``kind="momentum"``), the latter converted with
    momentum_to_potential. ``kind="auto"`` infers it from the length, which is
    ambiguous when n == |E| (a cycle, for instance) and then raises.

    Raises PositivityFloorError rather than returning garbage if the flow hits
    the positivity floor before time ``t``.
    """  # fmt: skip
    n, n_edges = G.n, G.E.shape[0]
    # Checked first: the momentum branch solves a Laplacian at nu, and a
    # boundary nu there surfaces as "the graph is disconnected".
    nu = _check_interior(G, nu, rho_floor(G, rtol=floor_rtol), "nu")
    tangent = np.asarray(tangent, dtype=float)
    if kind == "auto":
        length = tangent.shape[0] if tangent.ndim == 1 else -1
        if length not in (n, n_edges):
            raise ValueError(f"tangent has shape {tangent.shape}; expected ({n},) (potential) or ({n_edges},) (momentum)")
        if n == n_edges:
            raise ValueError(
                f"n == |E| == {n}, so the kind of the tangent cannot be inferred from its length; "
                "pass kind='potential' or kind='momentum'"
            )
        kind = "potential" if length == n else "momentum"
    if kind == "potential":
        if tangent.shape != (n,):
            raise ValueError(f"a potential must have shape ({n},), got {tangent.shape}")
        phi0 = _gauge(G, tangent)
    elif kind == "momentum":
        phi0 = momentum_to_potential(G, nu, tangent)
    else:
        raise ValueError(f"kind must be 'auto', 'potential' or 'momentum', got {kind!r}")

    return integrate_hamiltonian(G, nu, phi0, nsteps=nsteps, T=t, floor_rtol=floor_rtol)[0][:, -1]


def _reduced_to_potential(G: MarkovGraph, z: np.ndarray) -> np.ndarray:
    """z in R^(n-1) (or (n-1, k)) to phi0 in R^n with <phi0, 1>_pi = 0, solved
    for the last component. Removes the flow's gauge freedom so the Newton
    system is square and generically nonsingular."""
    last = -(G.pi[:-1] @ z) / G.pi[-1]
    return np.concatenate([z, last[np.newaxis]], axis=0)


def log_map(G: MarkovGraph, nu, target, *, phi0_init=None, tol: float = 1e-9, maxiters: int = 50,
            nsteps: int = 150, floor_rtol: float = 1e-6, verbose: bool = False) -> LogMapResult:
    """The Riemannian logarithm of ``target`` at ``nu``, by single shooting.

    Solves F(phi0) = rho(1; nu, phi0) - target = 0 by damped Newton over the
    mean-zero potentials: n - 1 unknowns, since the gauge <phi0, 1>_pi = 0 and
    the conserved mass each remove one dimension. The Jacobian is a batched
    forward finite difference (see the module docstring), the step is chosen
    by backtracking on ||F||_pi, and a step whose trajectory hits the
    positivity floor counts as a failed step and is shortened.

    Initialisation is the linearised geodesic L(nu) phi0 = pi * (target - nu),
    exact to first order in target - nu, or ``phi0_init`` if given -- the
    warm-start mechanism: callers keep their own phi0 from a nearby solve. If
    the first-order guess overshoots through the positivity floor, as it does
    for far-apart concentrated endpoints, it is halved until the first shot
    survives.

    Returns a LogMapResult. Raises ShootingError if Newton has not reached
    ``tol`` after ``maxiters`` steps, the line search fails, or no admissible
    initial potential exists; the SOCP is the fallback in every case, or
    log_map_mollified for data near the boundary.
    """  # fmt: skip
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive and finite, got {tol!r}")
    if isinstance(maxiters, bool) or not isinstance(maxiters, (int, np.integer)) or maxiters < 0:
        raise ValueError(f"maxiters must be an integer >= 0, got {maxiters!r}")
    if isinstance(nsteps, bool) or not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
        raise ValueError(f"nsteps must be an integer >= 1, got {nsteps!r}")
    floor_val = rho_floor(G, rtol=floor_rtol)
    nu = _check_interior(G, nu, floor_val, "nu")
    target = _check_interior(G, target, floor_val, "target")
    _check_mass(G, nu, "nu")
    _check_mass(G, target, "target")
    # The flow conserves mass exactly, so a mass gap between the endpoints is a
    # component of the residual that Newton cannot reduce. _check_mass admits
    # gaps up to 1e-8 -- the output of an iterative solver, a barycenter from
    # the SOCP say, is typically off by ~1e-9 -- which would make a tol below
    # the gap unreachable. Renormalising removes that floor at the cost of a
    # perturbation of the same, negligible, size.
    nu = nu / (nu @ G.pi)
    target = target / (target @ G.pi)

    n = G.n
    sqrt_pi = np.sqrt(G.pi)

    def shoot(z):
        # z of shape (n-1,) or (n-1, k); returns the full residual, same trailing shape
        phi0 = _reduced_to_potential(G, z)
        rho0 = np.broadcast_to(nu.reshape(-1, *(1,) * (z.ndim - 1)), phi0.shape)
        rho1, _ = _integrate_end(G, rho0, phi0, nsteps, 1.0, floor_val)
        return rho1 - target.reshape(-1, *(1,) * (z.ndim - 1))

    def resnorm(F):
        return float(np.linalg.norm(F * sqrt_pi))

    def jacobian(z):
        # Column 0 is the unperturbed shot, integrated in the same batch so it
        # shares the perturbed columns' step schedule.
        steps = _FD_STEP * np.maximum(1.0, np.abs(z))
        Z = np.tile(z[:, np.newaxis], (1, n))
        Z[np.arange(n - 1), np.arange(1, n)] += steps
        F = shoot(Z)[: n - 1]
        return (F[:, 1:] - F[:, [0]]) / steps

    if phi0_init is None:
        phi0 = solve_weighted_laplacian(G, nu, G.pi * (target - nu))
    else:
        phi0 = np.asarray(phi0_init, dtype=float)
        if phi0.shape != (n,):
            raise ValueError(f"phi0_init must have shape ({n},), got {phi0.shape}")
        if not np.all(np.isfinite(phi0)):
            # otherwise the damping loop halves nan and reports "too far apart"
            raise ValueError("phi0_init has non-finite entries")
        phi0 = _gauge(G, phi0)
    z = phi0[: n - 1].copy()

    # The first-order guess can overshoot through the floor for far-apart
    # endpoints; damp it until the first shot survives.
    F = None
    for _ in range(13):
        try:
            F = shoot(z)
            break
        except PositivityFloorError:
            z = z / 2
    if F is None:
        raise ShootingError(
            "log_map: no admissible initial potential found (the endpoints are too far apart for single "
            "shooting). Fall back to method='socp', or mollify the endpoints with log_map_mollified."
        )

    r = resnorm(F)
    iters = 0
    while r > tol:
        if iters >= maxiters:
            raise ShootingError(
                f"log_map: Newton did not converge in {maxiters} iterations (residual {r:.3e} > tol {tol:.0e}). "
                "Fall back to method='socp', or mollify the endpoints with log_map_mollified."
            )
        try:
            J = jacobian(z)
        except PositivityFloorError as exc:
            raise ShootingError(
                "log_map: the Jacobian's perturbed trajectories hit the positivity floor, so the current iterate "
                "is at the edge of what shooting can reach. Fall back to method='socp' or log_map_mollified."
            ) from exc
        delta = -np.linalg.solve(J, F[: n - 1])

        # Backtracking on ||F||_pi; a floor hit is a failed step, shortened the same way.
        alpha, accepted = 1.0, False
        for _ in range(12):
            z_try = z + alpha * delta
            try:
                F_try = shoot(z_try)
            except PositivityFloorError:
                F_try = None
            if F_try is not None and resnorm(F_try) <= (1 - 1e-4 * alpha) * r:
                z, F, r = z_try, F_try, resnorm(F_try)
                accepted = True
                break
            alpha /= 2
        if not accepted:
            raise ShootingError(
                f"log_map: line search failed at iteration {iters + 1} (residual {r:.3e}). Near the positivity "
                "floor the integrator's step bisection makes the residual piecewise-smooth in phi0, so Newton can "
                "stall at the scale of those jumps; otherwise the target may be too far from nu for single "
                "shooting. Fall back to method='socp', or mollify the endpoints with log_map_mollified."
            )
        iters += 1
        if verbose:
            logger.info("log_map: iter %d  residual %.3e  step %g", iters, r, alpha)

    phi0 = _reduced_to_potential(G, z)
    m0 = metric_tensor(G, nu) * graph_gradient(G, phi0)
    return LogMapResult(phi0, m0, 2 * hamiltonian(G, nu, phi0, floor_rtol=floor_rtol), iters, r)


def analyze_shooting(G: MarkovGraph, target, refs, *, nsteps: int = 150, tol: float = 1e-9, phi0_inits=None,
                     compute_condition: bool = False, return_system: bool = False, qp_method: str = "auto",
                     qp_solver=None):
    """The shooting analysis backend: like analyze_socp, but each reference's
    potential is log_map(G, target, ref).phi0 -- the Hamiltonian velocity
    potential at ``target`` -- instead of the SOCP's endpoint dual. The Gram
    matrix and simplex QP are shared (gram.potential_gram_qp).

    Requires strictly positive ``target`` and ``refs``; a failure on any one
    reference propagates. ``phi0_inits``, if given, holds one warm-start
    potential per reference.

    This checks stationarity in the Hamiltonian flow's discretisation, which
    differs from barycenter_socp's, so a barycenter synthesised by the SOCP is
    recovered only to O(h) in the SOCP's time step, not to solver tolerance.

    Returns lam_hat, or (lam_hat, A) with return_system=True.
    """  # fmt: skip
    from graphtransport.gram import potential_gram_qp

    if isinstance(refs, np.ndarray) and refs.ndim != 1:
        raise ValueError(
            f"refs must be a sequence of densities, each of shape ({G.n},); got a {refs.ndim}-D array of shape "
            f"{refs.shape}. Pass list(A) for references in the rows of A, or list(A.T) for columns."
        )
    refs = list(refs)
    if phi0_inits is not None and len(phi0_inits) != len(refs):
        raise ValueError(f"phi0_inits must have one entry per reference ({len(refs)}), got {len(phi0_inits)}")
    potentials = [
        log_map(G, target, ref, nsteps=nsteps, tol=tol, phi0_init=None if phi0_inits is None else phi0_inits[i]).phi0
        for i, ref in enumerate(refs)
    ]
    return potential_gram_qp(
        G, target, potentials, compute_condition=compute_condition, return_system=return_system,
        method=qp_method, solver=qp_solver,
    )  # fmt: skip


def log_map_mollified(G: MarkovGraph, nu, target, *, epsilons=(1e-2, 1e-3, 1e-4), tol: float = 1e-7,
                      **kwargs) -> MollifiedLogMapResult:
    """log_map for endpoints with zero or near-zero entries, where shooting
    cannot run directly.

    Both endpoints are mollified toward the uniform density,
    rho_eps = (1 - eps) rho + eps (still a probability density), log_map is
    run at each eps in ``epsilons`` (warm-starting each from the previous,
    coarser level), and the distance is extrapolated to eps -> 0 by a least-
    squares fit W(eps) ~ W0 + a sqrt(eps). Levels whose shooting fails --
    very small eps makes the near-boundary geodesic stiff -- are skipped with
    a warning, as long as two remain.

    The result is approximate by design, and flagged so; the SOCP handles
    boundary-supported data exactly and is the reference. Empirically (the
    Julia package's 5x5-grid probes) the raw distance at the smallest eps is
    often *more* accurate than the extrapolation: the mollification error
    decays faster than sqrt(eps). Both are returned so they can be compared.
    ``tol`` defaults to a looser 1e-7; the rest of ``kwargs`` go to log_map.
    """  # fmt: skip
    levels = sorted((float(e) for e in epsilons), reverse=True)
    if len(set(levels)) < 2:
        # Coincident levels make the fit rank-deficient, and lstsq answers that
        # with a minimum-norm solution rather than an error.
        raise ValueError(f"log_map_mollified needs at least two distinct epsilon levels, got {tuple(epsilons)}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"epsilon levels must be distinct, got {tuple(epsilons)}")
    if levels[0] >= 1 or levels[-1] <= 0:
        raise ValueError(f"every epsilon must lie in (0, 1), got {tuple(levels)}")
    # Validated before mollifying: (1 - eps) rho + eps can lift a negative
    # entry above the floor, after which every level's log_map would accept it.
    nu = _check_boundary_density(G, nu, "nu")
    target = _check_boundary_density(G, target, "target")

    def mollify(rho, eps):
        return (1 - eps) * rho + eps

    Ws, used, result = [], [], None
    for eps in levels:
        try:
            level = log_map(
                G, mollify(nu, eps), mollify(target, eps),
                phi0_init=None if result is None else result.phi0, tol=tol, **kwargs,
            )  # fmt: skip
        except (ShootingError, PositivityFloorError) as exc:
            warnings.warn(f"log_map_mollified: shooting failed at epsilon={eps:g}, skipping this level ({exc})",
                          stacklevel=2)
            continue
        result = level
        Ws.append(np.sqrt(level.W2))
        used.append(eps)
    if len(used) < 2:  # levels are distinct, so the fit below has full rank
        raise ShootingError("log_map_mollified: fewer than two epsilon levels solved; fall back to method='socp'")

    X = np.column_stack([np.ones(len(used)), np.sqrt(used)])
    W0, a = np.linalg.lstsq(X, np.array(Ws), rcond=None)[0]
    return MollifiedLogMapResult(
        W0**2, W0, result.phi0, result.m0, (W0, a), np.array(used), np.array(Ws)
    )
