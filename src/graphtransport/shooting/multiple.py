"""Multiple shooting for the logarithmic map.

Single shooting (explog.log_map) solves for one unknown, the initial
potential, by integrating the flow over the whole of [0, 1]. Multiple
shooting splits [0, 1] into K segments and makes the state at the start of
every segment an unknown: the potential phi_k of each segment and the density
rho_k of each interior junction. Newton then solves for all of them together,
with the residuals

    rho_k(end) - rho_{k+1}    and    gauge(phi_k(end)) - phi_{k+1}

at each junction and rho_{K-1}(end) - target at t = 1. Each segment is short,
so its flow map bends less than the whole transport's, and on long transports
Newton needs fewer steps (about half as many, corner to corner on a grid). Each
step costs more -- a Jacobian carries about twice as many tangents -- so on
short transports single shooting stays cheaper.

The segments are cut on the integrator's own step grid: with nsteps RK4 steps
of length 1/nsteps, segment k takes nsteps_k of them. The composed discrete
flow map is then exactly single shooting's, so wherever both converge they
solve the same discrete problem and agree to Newton's tolerance.

Coordinates. A potential is gauge fixed (<phi, 1>_pi = 0) and stored by its
first n - 1 entries, as in log_map; a junction density is mass fixed
(<rho, 1>_pi = 1) and stored the same way. Each junction therefore
contributes 2(n - 1) unknowns and 2(n - 1) residuals, and the system is square.

The Jacobian is exact, as single shooting's is: each segment's block is the
flow's tangent-linear model carried along the step schedule that segment took
in the residual evaluation (hamiltonian._torch_replay_tangent), with tangents
starting at d(rho_k, phi_k) / d(y_k, z_k). It is block bidiagonal -- each
junction residual also depends on the next segment's start through -I -- and
is solved as a sparse system.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from graphtransport.graph import MarkovGraph
from graphtransport.shooting.hamiltonian import (
    _DTYPE,
    PositivityFloorError,
    _as_tensor,
    _torch_integrate,
    _torch_replay_tangent,
)

logger = logging.getLogger(__name__)


def segment_steps(nsteps: int, segments: int) -> list[int]:
    """How many of the nsteps integrator steps each segment takes: as equal as
    possible, the longer segments first."""
    base, extra = divmod(nsteps, segments)
    return [base + (k < extra) for k in range(segments)]


def _full(G: MarkovGraph, w: np.ndarray, total: float) -> np.ndarray:
    """w in R^(n-1) to v in R^n with <v, 1>_pi = total, solved for the last
    component. total=0 is the potential gauge, total=1 the mass."""
    return np.concatenate([w, [(total - G.pi[:-1] @ w) / G.pi[-1]]])


def _gauge(G: MarkovGraph, phi: np.ndarray) -> np.ndarray:
    return phi - (G.pi @ phi) / G.pi.sum()


def initial_states_from_potential(G: MarkovGraph, nu: np.ndarray, phi0: np.ndarray, segments: int, nsteps: int,
                                  floor_val: float):
    """Segment start states (rho_starts, phi_starts), each (n, segments), from
    one sweep of the flow from (nu, phi0): the warm start from a single-shooting
    potential. None if the sweep hits the positivity floor."""  # fmt: skip
    try:
        _, _, _, (rho_path, phi_path) = _torch_integrate(G, _as_tensor(nu), _as_tensor(phi0), nsteps, 1.0,
                                                         floor_val, path=True)  # fmt: skip
    except PositivityFloorError:
        return None
    starts = np.concatenate([[0], np.cumsum(segment_steps(nsteps, segments))[:-1]])
    return rho_path.numpy()[:, starts], phi_path.numpy()[:, starts]


class _System:
    """The multiple-shooting system for (nu, target) with K segments: packing
    of the unknowns, the residual and its exact Jacobian.

    Unknown vector x: [z_0, (y_1, z_1), ..., (y_{K-1}, z_{K-1})], y a junction
    density and z a potential, each in reduced coordinates."""

    def __init__(self, G: MarkovGraph, nu, target, segments: int, nsteps: int, floor_val: float):
        self.G, self.nu, self.target, self.K = G, nu, target, segments
        self.n, self.m = G.n, G.n - 1
        self.steps = segment_steps(nsteps, segments)
        self.T = [s / nsteps for s in self.steps]
        self.floor_val = floor_val
        pi = torch.as_tensor(G.pi, dtype=_DTYPE)
        # d(full vector) / d(reduced coordinates): the identity with the constraint row
        self.lift = torch.cat([torch.eye(self.m, dtype=_DTYPE), (-pi[:-1] / pi[-1]).reshape(1, -1)])

    def pack(self, rho_s, phi_s):
        parts = [phi_s[: self.m, 0]]
        for k in range(1, self.K):
            parts += [rho_s[: self.m, k], phi_s[: self.m, k]]
        return np.concatenate(parts)

    def unpack(self, x):
        G, m, K = self.G, self.m, self.K
        rho_s, phi_s = np.empty((self.n, K)), np.empty((self.n, K))
        rho_s[:, 0] = self.nu
        phi_s[:, 0] = _full(G, x[:m], 0.0)
        for k in range(1, K):
            off = m + 2 * m * (k - 1)
            rho_s[:, k] = _full(G, x[off:off + m], 1.0)
            phi_s[:, k] = _full(G, x[off + m:off + 2 * m], 0.0)
        return rho_s, phi_s

    def residual(self, x):
        """The residual blocks and each segment's step schedule. Raises
        PositivityFloorError if a junction or a segment crosses the floor."""
        G, K = self.G, self.K
        rho_s, phi_s = self.unpack(x)
        if rho_s.min() <= self.floor_val:
            raise PositivityFloorError("a junction density is at or below the positivity floor")
        R, schedules = [], []
        for k in range(K):
            rho_e, phi_e, schedule, _ = _torch_integrate(G, _as_tensor(rho_s[:, k]), _as_tensor(phi_s[:, k]),
                                                         self.steps[k], self.T[k], self.floor_val)  # fmt: skip
            schedules.append(schedule)
            rho_e, phi_e = rho_e.numpy(), phi_e.numpy()
            if k < K - 1:
                R += [rho_e - rho_s[:, k + 1], _gauge(G, phi_e) - phi_s[:, k + 1]]
            else:
                R.append(rho_e - self.target)
        return R, schedules

    def reduced(self, R):
        return np.concatenate([r[: self.m] for r in R])

    def jacobian(self, x, schedules):
        """d reduced(R) / dx at x, exactly; ``schedules`` must be the residual's at x."""
        G, m, n, K = self.G, self.m, self.n, self.K
        rho_s, phi_s = self.unpack(x)
        zero = torch.zeros(n, m, dtype=_DTYPE)
        rows, cols, vals = [], [], []

        def put(block, r0, c0):
            i, j = np.nonzero(block)
            rows.append(i + r0)
            cols.append(j + c0)
            vals.append(block[i, j])

        for k in range(K):
            # tangents of the segment's start state w.r.t. its unknowns: z_0 alone for
            # the first segment (nu is fixed), (y_k, z_k) for the others
            if k == 0:
                col0, d_rho, d_phi = 0, zero, self.lift
            else:
                col0 = m + 2 * m * (k - 1)
                d_rho, d_phi = torch.cat([self.lift, zero], dim=1), torch.cat([zero, self.lift], dim=1)
            _, _, d_rho_e, d_phi_e = _torch_replay_tangent(G, _as_tensor(rho_s[:, k]), _as_tensor(phi_s[:, k]),
                                                           d_rho, d_phi, schedules[k])  # fmt: skip
            d_rho_e, d_phi_e = d_rho_e.numpy(), d_phi_e.numpy()
            if k < K - 1:
                d_gauge = d_phi_e - (G.pi @ d_phi_e) / G.pi.sum()
                put(np.vstack([d_rho_e[:m], d_gauge[:m]]), 2 * m * k, col0)
                put(-np.eye(2 * m), 2 * m * k, m + 2 * m * k)  # minus the next segment's start
            else:
                put(d_rho_e[:m], 2 * m * k, col0)
        size = (2 * K - 1) * m
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)
        )


def solve_multiple_shooting(G: MarkovGraph, nu, target, *, segments: int, nsteps: int, tol: float, maxiters: int,
                            floor_val: float, verbose: bool = False, init=None, give_up_early: bool = False):
    """Newton on the multiple-shooting system. nu and target must already be
    checked interior probability densities (log_map does that).

    ``init`` is None or (rho_starts, phi_starts), each (n, segments): the
    state at the start of every segment. The default puts the junction
    densities on the straight line from nu to target and each segment's
    potential at the linearized geodesic between consecutive junctions.

    ``give_up_early`` (log_map's segments="auto") raises as soon as Newton is
    evidently stalling, so the caller can go back to single shooting: if the
    first step is cut to 1/8 or less, or the residual has not halved after
    three steps. On the grid transports measured, every multiple-shooting
    solve that converged took a first step of 1/4 or more and halved its
    residual within three steps, while the ones that stalled did neither
    (tests/test_shooting_multiple.py has a stalling case).

    Returns (rho_starts, phi_starts, iters, residual). phi_starts are gauge
    fixed; the flow's potential drifts by a constant along a segment, so a
    continuous potential path is obtained by adding that drift back (see
    geodesic_shooting). Raises ShootingError on failure.
    """  # fmt: skip
    from graphtransport.shooting.explog import ShootingError, solve_weighted_laplacian

    n, K = G.n, segments
    system = _System(G, nu, target, segments, nsteps, floor_val)
    T = system.T
    t_start = np.concatenate([[0.0], np.cumsum(T)[:-1]])
    sqrt_pi = np.sqrt(G.pi)

    if init is None:
        rho_starts = np.column_stack([(1 - t) * nu + t * target for t in t_start])
        rho_ends = np.column_stack([rho_starts[:, 1:], target])
        phi_starts = np.column_stack([
            solve_weighted_laplacian(G, rho_starts[:, k], G.pi * (rho_ends[:, k] - rho_starts[:, k])) / T[k]
            for k in range(K)
        ])  # fmt: skip
    else:
        rho_starts, phi_starts = (np.array(a, dtype=float) for a in init)
        if rho_starts.shape != (n, K) or phi_starts.shape != (n, K):
            raise ValueError(f"init must be two arrays of shape ({n}, {K})")
        rho_starts[:, 0] = nu
        rho_starts[:, 1:] /= G.pi @ rho_starts[:, 1:]
        phi_starts = phi_starts - (G.pi @ phi_starts) / G.pi.sum()

    def norm(R):
        return float(np.sqrt(sum(np.sum((r * sqrt_pi) ** 2) for r in R)))

    x = system.pack(rho_starts, phi_starts)
    R = schedules = None
    for _ in range(13):  # damp the initial potentials until every segment survives
        try:
            R, schedules = system.residual(x)
            break
        except PositivityFloorError:
            rho_s, phi_s = system.unpack(x)
            x = system.pack(rho_s, phi_s / 2)
    if R is None:
        raise ShootingError(
            f"log_map(segments={K}): no admissible initial state found. Fall back to method='socp'."
        )

    r = r_start = norm(R)
    iters = 0
    while r > tol:
        if iters >= maxiters:
            raise ShootingError(
                f"log_map: Newton did not converge in {maxiters} iterations (residual {r:.3e} > tol {tol:.0e}; "
                f"segments={K}). Fall back to method='socp'."
            )
        delta = -spsolve(system.jacobian(x, schedules).tocsc(), system.reduced(R))
        if not np.all(np.isfinite(delta)):
            raise ShootingError(f"log_map(segments={K}): singular Newton system. Fall back to method='socp'.")
        alpha, accepted = 1.0, False
        for _ in range(12):  # single shooting's budget: both give up at the same step length
            x_try = x + alpha * delta
            try:
                R_try, schedules_try = system.residual(x_try)
            except PositivityFloorError:
                R_try = schedules_try = None
            if R_try is not None and norm(R_try) <= (1 - 1e-4 * alpha) * r:
                x, R, schedules, r = x_try, R_try, schedules_try, norm(R_try)
                accepted = True
                break
            alpha /= 2
        if not accepted:
            raise ShootingError(
                f"log_map: line search failed at iteration {iters + 1} (residual {r:.3e}; segments={K}). "
                "Fall back to method='socp'."
            )
        iters += 1
        if verbose:
            logger.info("log_map(segments=%d): iter %d  residual %.3e  step %g", K, iters, r, alpha)
        if give_up_early and r > tol and ((iters == 1 and alpha <= 1 / 8) or (iters == 3 and r > r_start / 2)):
            raise ShootingError(
                f"log_map(segments={K}): Newton is stalling (step {alpha:g} at iteration {iters}, residual {r:.3e} "
                f"from {r_start:.3e})."
            )

    rho_s, phi_s = system.unpack(x)
    return rho_s, phi_s, iters, r
