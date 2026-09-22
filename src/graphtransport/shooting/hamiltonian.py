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
They contain the identity rho_dot = -div(theta(rho) . grad phi), which is why
rho_dot reuses graph_divergence and graph_gradient directly.
"""

from __future__ import annotations

import numpy as np

from graphtransport.graph import MarkovGraph, graph_divergence, graph_gradient


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


def hamiltonian(G: MarkovGraph, rho, phi) -> float:
    """H(rho, phi) = 1/2 sum_e kappa_e theta(rho_x, rho_y) (grad phi)_e^2.

    Twice H is the squared transport distance along the geodesic this state
    generates, which is how the flow's own endpoints can be checked against
    the SOCP.
    """
    rho = np.asarray(rho, dtype=float)
    grad_phi = graph_gradient(G, phi)
    s, t = _edge_densities(G, rho)
    return float(0.5 * np.sum(G.kappa * G.mean(s, t) * grad_phi**2))


def hamiltonian_flow(G: MarkovGraph, rho, phi):
    """The equations of motion, (rho_dot, phi_dot):

        rho_dot(x) = sum_y theta(rho_x, rho_y) (phi_x - phi_y) Q(x, y)
                   = -div(theta(rho) . grad phi)(x)
        phi_dot(x) = -1/2 sum_y partial_s theta(rho_x, rho_y) (phi_x - phi_y)^2 Q(x, y)

    Requires rho strictly positive; the caller is responsible for the check
    (integrate_hamiltonian does it once per call rather than per RK4 stage).
    """
    rho = np.asarray(rho, dtype=float)
    grad_phi = graph_gradient(G, phi)
    s, t = _edge_densities(G, rho)
    rho_dot = -graph_divergence(G, G.mean(s, t) * grad_phi)

    # Q[x, y] == kappa_e / pi[x] and Q[y, x] == kappa_e / pi[y] by the
    # definition of kappa, so the rates need no sparse lookup per edge.
    x, y = G.E[:, 0], G.E[:, 1]
    half_grad_sq = 0.5 * grad_phi**2
    phi_dot = np.zeros(G.n, dtype=float)
    np.add.at(phi_dot, x, -G.mean.partial_s(s, t) * half_grad_sq * G.kappa / G.pi[x])
    np.add.at(phi_dot, y, -G.mean.partial_s(t, s) * half_grad_sq * G.kappa / G.pi[y])
    return rho_dot, phi_dot


def _rk4_step(G: MarkovGraph, rho, phi, h: float):
    k1r, k1p = hamiltonian_flow(G, rho, phi)
    k2r, k2p = hamiltonian_flow(G, rho + (h / 2) * k1r, phi + (h / 2) * k1p)
    k3r, k3p = hamiltonian_flow(G, rho + (h / 2) * k2r, phi + (h / 2) * k2p)
    k4r, k4p = hamiltonian_flow(G, rho + h * k3r, phi + h * k3p)
    rho_next = rho + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
    phi_next = phi + (h / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
    return rho_next, phi_next


def _advance_interval(G: MarkovGraph, rho, phi, dt: float, floor_val: float, depth: int):
    """Advance by exactly dt, bisecting (and recursing on each half) whenever a
    step would cross the positivity floor. Bisecting rather than retrying with
    a smaller h keeps the integrator's clock in step with the nsteps * h = T
    the caller asked for.

    An RK4 stage evaluates the flow at *intermediate* proposed states, which
    can dip below the floor even when the accepted output would not have. In
    Julia that surfaces as a DomainError from inside the flow; numpy instead
    produces nan silently (the means return nan outside their domain by
    design), so the guard tests for finiteness as well as for the floor.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_next, phi_next = _rk4_step(G, rho, phi, dt)
    ok = np.all(np.isfinite(rho_next)) and np.all(np.isfinite(phi_next)) and rho_next.min() > floor_val
    if ok:
        return rho_next, phi_next
    if depth <= 0:
        raise PositivityFloorError(
            "the Hamiltonian flow hit the positivity floor after repeated step halving. Fall back to "
            "method='socp' (exact, handles densities supported on part of the graph) for this instance."
        )
    rho_mid, phi_mid = _advance_interval(G, rho, phi, dt / 2, floor_val, depth - 1)
    return _advance_interval(G, rho_mid, phi_mid, dt / 2, floor_val, depth - 1)


def integrate_hamiltonian(G: MarkovGraph, rho0, phi0, *, nsteps: int = 150, T: float = 1.0,
                          floor_rtol: float = 1e-6, max_halvings: int = 4):
    """Integrate the flow forward from (rho0, phi0) over [0, T], returning the
    paths as (n, nsteps + 1) arrays.

    Fixed-step classical RK4, not a symplectic integrator: over a unit time
    interval at 100-200 steps, symplecticity is a nicety rather than a
    requirement, and H is conserved to RK4 truncation error (~1e-5 at
    nsteps=200 on the test graphs).

    Raises PositivityFloorError if a density would fall below
    ``rho_floor(G, rtol=floor_rtol)`` even after ``max_halvings`` bisections
    of the offending step. Callers are expected to catch it and fall back.
    """  # fmt: skip
    if isinstance(nsteps, bool) or not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
        raise ValueError(f"nsteps must be an integer >= 1, got {nsteps!r}")
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be a positive, finite time, got {T!r}")
    floor_val = rho_floor(G, rtol=floor_rtol)
    rho = _check_interior(G, rho0, floor_val, "rho0")
    phi = np.asarray(phi0, dtype=float)
    if phi.shape != (G.n,):
        raise ValueError(f"phi0 must have shape ({G.n},), got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi0 has non-finite entries")

    rho_path = np.empty((G.n, nsteps + 1))
    phi_path = np.empty((G.n, nsteps + 1))
    rho_path[:, 0], phi_path[:, 0] = rho, phi
    h = T / nsteps
    for i in range(nsteps):
        rho, phi = _advance_interval(G, rho, phi, h, floor_val, max_halvings)
        rho_path[:, i + 1], phi_path[:, i + 1] = rho, phi
    return rho_path, phi_path
