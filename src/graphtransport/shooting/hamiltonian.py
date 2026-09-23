"""The Hamiltonian ODE system underlying the exponential and logarithmic maps.

Ported from GraphTransportation.jl's shooting/Hamiltonian.jl. The state is
(rho, phi) in R^n x R^n; the mobility theta and its partial derivative come
from the graph's mean, so every AdmissibleMean works here, including the
exact LogarithmicMean (the SOCP needs QuadLogMean for that one, which makes
QuadLogMean(8) the right thing to cross-validate against).

Valid only for **strictly positive** densities: partial_s(s, t) divides by s
for most means, so a density at the boundary is not merely inaccurate but
undefined. `geodesic_socp` is the exact method for boundary-supported data.
Under the arithmetic mean the mobility does not vanish at an empty node, so
the flow can legitimately drive a density through zero and hit the floor;
prefer the SOCP for that mean near the boundary.

The equations of motion are Hamilton's equations for the pi-weighted pairing,
rho_dot(z) = (1/pi(z)) dH/dphi(z) and phi_dot(z) = -(1/pi(z)) dH/drho(z).
They contain the identity rho_dot = -div(theta(rho) . grad phi), with div
the graph divergence (G.D's sign convention).

The flow and the integrator are implemented once, in torch (see "the flow in
torch" below), so the shooting Jacobian can be exact; the public functions
here take and return numpy arrays.
"""

from __future__ import annotations

import sys
import warnings
from typing import Literal, overload

import numpy as np
import torch

from graphtransport.graph import MarkovGraph, graph_gradient


class TorchThreadsWarning(UserWarning):
    """torch is using more than one thread for the shooting solver's small
    operations, which costs CPU time without speeding them up. Issued once per
    process; filter it to silence it."""


_threads_warned = False


def _in_package(module: str) -> bool:
    return module == "graphtransport" or module.startswith("graphtransport.")


def _stacklevel_outside_package() -> int:
    """The warnings.warn stacklevel of the first caller outside graphtransport,
    so a warning raised deep in the solver points at the user's own line."""
    frame, level = sys._getframe(1), 1
    while frame is not None and _in_package(frame.f_globals.get("__name__", "")):
        frame, level = frame.f_back, level + 1
    return level


def _warn_about_threads() -> None:
    """At the graph sizes shooting is for, torch's thread pool adds CPU time
    without adding speed, and several solves run in parallel would
    oversubscribe the machine. Changing torch's global thread count is the
    caller's decision, so this only says so, once."""
    global _threads_warned
    if _threads_warned:
        return
    _threads_warned = True
    threads = torch.get_num_threads()
    if threads > 1:
        warnings.warn(
            f"torch is using {threads} threads. The shooting solver runs many small torch operations, which "
            "gain nothing from threads at graph sizes up to a few hundred nodes but cost several times the CPU "
            "time. Consider torch.set_num_threads(1), or OMP_NUM_THREADS=1, especially when running solves in "
            "parallel. Filter TorchThreadsWarning to silence this.",
            TorchThreadsWarning,
            stacklevel=_stacklevel_outside_package(),
        )


class PositivityFloorError(Exception):
    """A trajectory would drive some density below the positivity floor even
    after repeated step bisection.

    `log_map` catches exactly this type -- and nothing broader -- to damp its
    initial guess and shorten line-search steps, so the flow's other failures
    must not be raised as this.
    """


def rho_floor(G: MarkovGraph, *, rtol: float = 1e-6) -> float:
    """The smallest density the shooting machinery tolerates: ``rtol``
    relative to min(G.pi). Below it, a mean's partial_s divides by a
    near-zero argument and the flow is numerically meaningless; fall back to
    the SOCP (exact) or a mollified approximation."""
    return float(rtol * np.min(G.pi))


def _check_interior(G: MarkovGraph, rho, floor_val: float, what: str) -> np.ndarray:
    """rho as a float array, checked to be strictly above the floor. An
    explicit raise rather than an assert: `python -O` strips asserts, and the
    flow's output for a boundary density is silently nan, not an error."""
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (G.n,):
        raise ValueError(f"{what} must have shape ({G.n},), got {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"{what} has non-finite entries")
    smallest = float(rho.min())
    if smallest <= floor_val:
        raise ValueError(
            f"{what} violates the positivity floor (min {smallest:.3e} <= {floor_val:.3e}): shooting "
            "requires strictly positive densities. Use method='socp' (exact, handles densities "
            "supported on part of the graph) for this instance."
        )
    return rho


def _edge_densities(G: MarkovGraph, rho):
    """(rho_x, rho_y) along the oriented edges."""
    return rho[G.E[:, 0]], rho[G.E[:, 1]]


def hamiltonian(G: MarkovGraph, rho, phi, *, floor_rtol: float = 1e-6) -> float:
    """H(rho, phi) = 1/2 sum_e kappa_e theta(rho_x, rho_y) (grad phi)_e^2.

    Twice H is the squared transport distance along the geodesic this state
    generates, which is how the flow's own endpoints can be checked against
    the SOCP. Requires rho strictly positive, and says so rather than
    returning the plausible finite number a boundary density produces.
    """
    rho = _check_interior(G, rho, rho_floor(G, rtol=floor_rtol), "rho")
    grad_phi = graph_gradient(G, phi)
    s, t = _edge_densities(G, rho)
    return float(0.5 * np.sum(G.kappa * G.mean(s, t) * grad_phi**2))


def hamiltonian_flow(G: MarkovGraph, rho, phi, *, floor_rtol: float = 1e-6):
    """The equations of motion, (rho_dot, phi_dot):

        rho_dot(x) = sum_y theta(rho_x, rho_y) (phi_x - phi_y) Q(x, y)
                   = -div(theta(rho) . grad phi)(x)
        phi_dot(x) = -1/2 sum_y partial_s theta(rho_x, rho_y) (phi_x - phi_y)^2 Q(x, y)

    Requires rho strictly positive, and checks it: at the boundary phi_dot is
    -inf (partial_s(0, t) genuinely diverges) and just inside it the result is
    finite but meaningless, neither of which announces itself downstream.

    The integrator's RK4 stages call the unchecked torch flow instead, so it
    still validates once per call rather than four times per step.
    """
    rho = _check_interior(G, rho, rho_floor(G, rtol=floor_rtol), "rho")
    return _hamiltonian_flow(G, rho, phi)


def _hamiltonian_flow(G: MarkovGraph, rho, phi):
    """hamiltonian_flow without the domain check, on numpy arrays; rho must
    already be strictly above the floor. A thin wrapper over the torch flow,
    which is the one implementation of the equations of motion."""
    rho_dot, phi_dot = _torch_flow(G, _as_tensor(rho), _as_tensor(phi))
    return rho_dot.numpy(), phi_dot.numpy()


# ----- the flow in torch -----
#
# The equations of motion and the integrator are written once, in torch, so
# that torch's autodiff can differentiate the discrete flow map exactly: that
# is what log_map's Jacobian is (Julia uses ForwardDiff for the same thing; a
# finite-difference Jacobian loses accuracy as Newton moves into the regions
# where long transports thin out, and stalls there). Everything is float64.
#
# The integrator bisects a step whenever it would cross the positivity floor,
# which is a branch on the data. torch.func's batched autodiff cannot trace
# such a branch, so the integration is split in two: _torch_integrate runs the
# shot without autodiff and records the step schedule it took, and
# _torch_replay re-runs a given schedule with no branches, which autodiff can
# differentiate. Replaying the schedule of the shot being differentiated gives
# the derivative of the branch the shot actually took -- ForwardDiff's
# semantics exactly.

_DTYPE = torch.float64


def _as_tensor(a) -> torch.Tensor:
    a = np.asarray(a, dtype=float)
    if not a.flags.writeable:  # a broadcast view, say: torch warns on non-writable arrays
        a = a.copy()
    return torch.as_tensor(a, dtype=_DTYPE)


def _torch_edges(G: MarkovGraph):
    """(x, y, Q[x, y], Q[y, x]) per oriented edge, as tensors. Cached on the
    graph; with_mean copies the instance dict, so graphs differing only in
    their mean share them, correctly: they depend on Q alone."""
    cached = G.__dict__.get("_torch_edges")
    if cached is None:
        x, y = G.E[:, 0], G.E[:, 1]
        q_xy = np.asarray(G.Q[x, y], dtype=float).ravel()
        q_yx = np.asarray(G.Q[y, x], dtype=float).ravel()
        cached = (
            torch.as_tensor(x),
            torch.as_tensor(y),
            torch.as_tensor(q_xy, dtype=_DTYPE),
            torch.as_tensor(q_yx, dtype=_DTYPE),
        )
        G.__dict__["_torch_edges"] = cached
    return cached


def _torch_flow(G: MarkovGraph, rho: torch.Tensor, phi: torch.Tensor):
    """(rho_dot, phi_dot) on tensors of shape (n,) or, with a trailing batch
    axis, (n, k): column j is an independent state.

    rho_dot = -div(theta . grad phi): edge e = (x, y) carries theta_e (grad
    phi)_e out of x at rate Q[x, y] and into y at rate Q[y, x], which is G.D's
    sign convention. phi_dot gathers -1/2 partial theta (grad phi)^2 onto both
    ends the same way."""
    x, y, q_xy, q_yx = _torch_edges(G)
    if rho.dim() == 2:
        q_xy, q_yx = q_xy[:, None], q_yx[:, None]
    grad_phi = phi[x] - phi[y]
    s, t = rho[x], rho[y]
    flux = G.mean.torch_theta(s, t) * grad_phi
    half_grad_sq = 0.5 * grad_phi * grad_phi
    zeros = torch.zeros_like(rho)
    rho_dot = zeros.index_add(0, x, q_xy * flux).index_add(0, y, -q_yx * flux)
    phi_dot = zeros.index_add(0, x, -q_xy * G.mean.torch_partial_s(s, t) * half_grad_sq).index_add(
        0, y, -q_yx * G.mean.torch_partial_s(t, s) * half_grad_sq
    )
    return rho_dot, phi_dot


def _torch_flow_tangent(G: MarkovGraph, rho, phi, d_rho, d_phi):
    """The flow and its directional derivatives: (rho_dot, phi_dot, d_rho_dot,
    d_phi_dot), for one state (rho, phi) of shape (n,) and a block of tangents
    (d_rho, d_phi) of shape (n, k).

    This is forward-mode differentiation written out, the tangent-linear model:
    the chain rule through the graph is linear and explicit here, and only the
    elementwise second derivatives of the mean come from the mean itself
    (closed forms, or one autodiff call on |E|-sized vectors). Carrying the k
    tangents as one batch is what makes the Jacobian cheap; torch.func.jacfwd
    pushes them through vmap instead, one small operation per tangent, and is
    much slower."""
    x, y, q_xy, q_yx = _torch_edges(G)
    mean = G.mean
    g = phi[x] - phi[y]
    s, t = rho[x], rho[y]
    theta = mean.torch_theta(s, t)
    a, b = mean.torch_partial_s(s, t), mean.torch_partial_s(t, s)  # d theta / ds, d theta / dt
    # second derivatives: a = P(s, t) and b = P(t, s) with P = partial_s, both
    # evaluated in one call on the edges stacked with their reverses
    m = s.shape[0]
    P_1, P_2 = mean.torch_partial_s_grad(torch.cat([s, t]), torch.cat([t, s]))
    a_s, b_t = P_1[:m], P_1[m:]
    a_t, b_s = P_2[:m], P_2[m:]

    half_g2 = 0.5 * g * g
    zeros = torch.zeros_like(rho)
    flux = theta * g
    rho_dot = zeros.index_add(0, x, q_xy * flux).index_add(0, y, -q_yx * flux)
    phi_dot = zeros.index_add(0, x, -q_xy * a * half_g2).index_add(0, y, -q_yx * b * half_g2)

    ds, dt, dg = d_rho[x], d_rho[y], d_phi[x] - d_phi[y]
    c = lambda v: v[:, None]  # noqa: E731 -- edge coefficient against a (|E|, k) block
    d_flux = (c(a) * ds + c(b) * dt) * c(g) + c(theta) * dg
    d_a, d_b = c(a_s) * ds + c(a_t) * dt, c(b_s) * ds + c(b_t) * dt
    d_half_g2 = c(g) * dg
    tzeros = torch.zeros_like(d_rho)
    d_rho_dot = tzeros.index_add(0, x, c(q_xy) * d_flux).index_add(0, y, -c(q_yx) * d_flux)
    d_phi_dot = tzeros.index_add(0, x, -c(q_xy) * (d_a * c(half_g2) + c(a) * d_half_g2)).index_add(
        0, y, -c(q_yx) * (d_b * c(half_g2) + c(b) * d_half_g2)
    )
    return rho_dot, phi_dot, d_rho_dot, d_phi_dot


def _torch_rk4_tangent(G: MarkovGraph, rho, phi, d_rho, d_phi, h: float):
    """One RK4 step of the state and, linearized, of the tangent block."""
    k1r, k1p, l1r, l1p = _torch_flow_tangent(G, rho, phi, d_rho, d_phi)
    k2r, k2p, l2r, l2p = _torch_flow_tangent(
        G, rho + (h / 2) * k1r, phi + (h / 2) * k1p, d_rho + (h / 2) * l1r, d_phi + (h / 2) * l1p
    )
    k3r, k3p, l3r, l3p = _torch_flow_tangent(
        G, rho + (h / 2) * k2r, phi + (h / 2) * k2p, d_rho + (h / 2) * l2r, d_phi + (h / 2) * l2p
    )
    k4r, k4p, l4r, l4p = _torch_flow_tangent(G, rho + h * k3r, phi + h * k3p, d_rho + h * l3r, d_phi + h * l3p)
    rho_next = rho + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
    phi_next = phi + (h / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
    d_rho_next = d_rho + (h / 6) * (l1r + 2 * l2r + 2 * l3r + l4r)
    d_phi_next = d_phi + (h / 6) * (l1p + 2 * l2p + 2 * l3p + l4p)
    return rho_next, phi_next, d_rho_next, d_phi_next


def _torch_replay_tangent(G: MarkovGraph, rho, phi, d_rho, d_phi, schedule):
    """_torch_replay with a block of tangents carried along: returns the end
    state and d(end state) in the directions (d_rho, d_phi), exactly."""
    with torch.no_grad():
        for steps in schedule:
            for dt in steps:
                rho, phi, d_rho, d_phi = _torch_rk4_tangent(G, rho, phi, d_rho, d_phi, dt)
    return rho, phi, d_rho, d_phi


def _torch_rk4(G: MarkovGraph, rho, phi, h: float):
    k1r, k1p = _torch_flow(G, rho, phi)
    k2r, k2p = _torch_flow(G, rho + (h / 2) * k1r, phi + (h / 2) * k1p)
    k3r, k3p = _torch_flow(G, rho + (h / 2) * k2r, phi + (h / 2) * k2p)
    k4r, k4p = _torch_flow(G, rho + h * k3r, phi + h * k3p)
    return rho + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r), phi + (h / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)


def _advance_interval(G: MarkovGraph, rho, phi, dt: float, floor_val: float, depth: int, schedule: list):
    """Advance by exactly dt, bisecting (and recursing on each half) whenever a
    step would cross the positivity floor, and append each step length taken to
    ``schedule``. Bisecting rather than retrying with a smaller h keeps the
    integrator's clock in step with the nsteps * h = T the caller asked for.

    An RK4 stage evaluates the flow at *intermediate* proposed states, which
    can dip below the floor even when the accepted output would not have; the
    means then produce nan (or inf) rather than an error, so the guard tests for
    finiteness as well as for the floor. With a batch axis every column takes
    the same steps: if any column needs a bisection, all bisect.
    """
    rho_next, phi_next = _torch_rk4(G, rho, phi, dt)
    if (
        bool(torch.isfinite(rho_next).all())
        and bool(torch.isfinite(phi_next).all())
        and float(rho_next.min()) > floor_val
    ):
        schedule.append(dt)
        return rho_next, phi_next
    if depth <= 0:
        raise PositivityFloorError(
            "the Hamiltonian flow hit the positivity floor after repeated step halving. Fall back to "
            "method='socp' (exact, handles densities supported on part of the graph) for this instance."
        )
    rho_mid, phi_mid = _advance_interval(G, rho, phi, dt / 2, floor_val, depth - 1, schedule)
    return _advance_interval(G, rho_mid, phi_mid, dt / 2, floor_val, depth - 1, schedule)


@overload
def _torch_integrate(
    G: MarkovGraph,
    rho,
    phi,
    nsteps: int,
    T: float,
    floor_val: float,
    max_halvings: int = ...,
    path: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor, list, None]: ...
@overload
def _torch_integrate(
    G: MarkovGraph, rho, phi, nsteps: int, T: float, floor_val: float, max_halvings: int = ..., *, path: Literal[True]
) -> tuple[torch.Tensor, torch.Tensor, list, tuple[torch.Tensor, torch.Tensor]]: ...
def _torch_integrate(
    G: MarkovGraph, rho, phi, nsteps: int, T: float, floor_val: float, max_halvings: int = 4, path: bool = False
):
    """Run the flow from tensors (rho, phi) over [0, T] in nsteps steps, without
    autodiff. Returns (rho_end, phi_end, schedule, paths): schedule[i] is the
    list of step lengths that step i was taken in (one entry unless it was
    bisected), and paths is (rho_path, phi_path), each (n, nsteps + 1), if
    ``path`` else None."""
    _warn_about_threads()
    schedule = []
    rho_path, phi_path = [rho], [phi]
    h = T / nsteps
    with torch.no_grad():
        for _ in range(nsteps):
            steps: list = []
            rho, phi = _advance_interval(G, rho, phi, h, floor_val, max_halvings, steps)
            schedule.append(steps)
            if path:
                rho_path.append(rho)
                phi_path.append(phi)
    paths = (torch.stack(rho_path, dim=1), torch.stack(phi_path, dim=1)) if path else None
    return rho, phi, schedule, paths


def _torch_replay(G: MarkovGraph, rho, phi, schedule):
    """The end state of the flow from (rho, phi) along a recorded schedule: the
    same arithmetic as _torch_integrate, with no branches, so torch's autodiff
    can differentiate it."""
    for steps in schedule:
        for dt in steps:
            rho, phi = _torch_rk4(G, rho, phi, dt)
    return rho, phi


def integrate_hamiltonian(
    G: MarkovGraph, rho0, phi0, *, nsteps: int = 150, T: float = 1.0, floor_rtol: float = 1e-6, max_halvings: int = 4
):
    """Integrate the flow forward from (rho0, phi0) over [0, T], returning the
    paths as (n, nsteps + 1) arrays.

    Fixed-step classical RK4, not a symplectic integrator: over a unit time
    interval at 100-200 steps, symplecticity is a nicety rather than a
    requirement, and H is conserved to RK4 truncation error (~1e-5 at
    nsteps=200 on the test graphs).

    Raises PositivityFloorError if a density would fall below
    ``rho_floor(G, rtol=floor_rtol)`` even after ``max_halvings`` bisections
    of the offending step. Callers are expected to catch it and fall back.
    """
    if isinstance(nsteps, bool) or not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
        raise ValueError(f"nsteps must be an integer >= 1, got {nsteps!r}")
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be a positive, finite time, got {T!r}")
    if isinstance(max_halvings, bool) or not isinstance(max_halvings, (int, np.integer)) or max_halvings < 0:
        # A negative value would disable bisection silently: depth <= 0 holds on
        # entry, so the first floor crossing raises instead of being retried.
        raise ValueError(f"max_halvings must be an integer >= 0, got {max_halvings!r}")
    floor_val = rho_floor(G, rtol=floor_rtol)
    rho = _check_interior(G, rho0, floor_val, "rho0")
    phi = np.asarray(phi0, dtype=float)
    if phi.shape != (G.n,):
        raise ValueError(f"phi0 must have shape ({G.n},), got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi0 has non-finite entries")
    _, _, _, (rho_path, phi_path) = _torch_integrate(
        G, _as_tensor(rho), _as_tensor(phi), nsteps, T, floor_val, max_halvings, path=True
    )
    return rho_path.numpy(), phi_path.numpy()
