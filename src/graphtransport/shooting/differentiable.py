"""Differentiable shooting: geodesics and transport costs as torch functions.

The shooting solver runs Newton on F(z) = rho(1; nu, phi0(z)) - target = 0,
where z is the gauge-reduced initial potential (explog.log_map). Its solution
z* is a function of the endpoints, and everything a geodesic reports -- W2 =
2 H(nu, phi0), the density path, the momenta and potentials -- is a smooth
function of (nu, z*). So differentiating a geodesic comes down to one custom
autograd function, _LogMapPotential, which returns z* and whose backward
applies the implicit function theorem:

    dz* = J^-1 (d target - dF/dnu d nu),   J = dF/dz at z*,

so for a cotangent z_bar, with w = J^-T z_bar, the target receives w and the
start density receives -(dF/dnu)^T w, one reverse-mode pass through the
replayed flow. Everything downstream is ordinary torch code, and autograd
composes it. The gradients are exact for the discrete problem the solver
solves -- the RK4 flow along the step schedule the shot took -- not merely
for its continuous limit, so they agree with finite differences of the
returned values.

Inputs are normalized by their pi-mass inside the graph, so a gradient is
defined in every direction and the derivative along a pure rescaling of a
density is zero. Tensors are float64 on the CPU.

Gradients are first order only. The backward solves with numpy and is marked
once_differentiable, so differentiating a gradient again -- create_graph=True,
for a gradient penalty or a Hessian-vector product -- raises torch's
"trying to differentiate twice a function that was marked with
@once_differentiable", where the function is _LogMapPotential.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from graphtransport.graph import MarkovGraph
from graphtransport.shooting.hamiltonian import (
    _DTYPE,
    _torch_edges,
    _torch_integrate,
    _torch_replay,
    _torch_rk4,
    rho_floor,
)


def _torch_potential(G: MarkovGraph, z: torch.Tensor) -> torch.Tensor:
    from graphtransport.shooting.explog import _torch_potential

    return _torch_potential(G, z)


class _LogMapPotential(torch.autograd.Function):
    """z* = the reduced initial potential of the geodesic from nu to target,
    with an implicit-function-theorem backward. nu and target must be
    probability densities w.r.t. G.pi (the caller normalizes them)."""

    @staticmethod
    def forward(ctx, nu, target, G, nsteps, tol, maxiters, floor_rtol, verbose, phi0_init, segments):
        from graphtransport.shooting.explog import log_map

        # With segments > 1 z* comes from multiple shooting; the backward's
        # Jacobian is still single shooting's at z*, which is exact: both solve
        # the same discrete problem.
        r = log_map(
            G,
            nu.detach().numpy(),
            target.detach().numpy(),
            tol=tol,
            maxiters=maxiters,
            nsteps=nsteps,
            floor_rtol=floor_rtol,
            verbose=verbose,
            phi0_init=phi0_init,
            segments=segments,
        )
        z = torch.as_tensor(r.phi0[: G.n - 1], dtype=_DTYPE)
        ctx.save_for_backward(nu.detach(), z)
        ctx.G, ctx.nsteps, ctx.floor_val = G, nsteps, rho_floor(G, rtol=floor_rtol)
        return z

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):  # pyright: ignore[reportIncompatibleMethodOverride]
        # (once_differentiable returns an untyped wrapper; this is torch's documented pattern)
        from graphtransport.shooting.explog import _shoot, _shooting_jacobian

        (z_bar,) = grad_outputs

        nu, z = ctx.saved_tensors
        G, n = ctx.G, ctx.G.n
        _, schedule = _shoot(G, nu, z.numpy(), ctx.nsteps, ctx.floor_val)
        J = _shooting_jacobian(G, nu, z.numpy(), schedule)
        w = torch.as_tensor(np.linalg.solve(J.T, z_bar.numpy()), dtype=_DTYPE)
        target_bar = torch.cat([w, torch.zeros(1, dtype=_DTYPE)])  # F uses target[:n-1] only
        nu_bar = None
        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                nu_req = nu.clone().requires_grad_(True)
                rho1 = _torch_replay(G, nu_req, _torch_potential(G, z), schedule)[0][: n - 1]
                (vjp,) = torch.autograd.grad(rho1, nu_req, grad_outputs=w)
            nu_bar = -vjp
        return nu_bar, target_bar, None, None, None, None, None, None, None, None


def _torch_hamiltonian(G: MarkovGraph, rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    x, y, _, _ = _torch_edges(G)
    kappa = torch.as_tensor(G.kappa, dtype=_DTYPE)
    grad_phi = phi[x] - phi[y]
    return 0.5 * torch.sum(kappa * G.mean.torch_theta(rho[x], rho[y]) * grad_phi * grad_phi)


def _torch_replay_path(G: MarkovGraph, rho, phi, schedule):
    """_torch_replay keeping the state after each of the nsteps steps:
    (rho_path, phi_path), each (n, nsteps + 1), differentiable."""
    rho_path, phi_path = [rho], [phi]
    for steps in schedule:
        for dt in steps:
            rho, phi = _torch_rk4(G, rho, phi, dt)
        rho_path.append(rho)
        phi_path.append(phi)
    return torch.stack(rho_path, dim=1), torch.stack(phi_path, dim=1)


def _normalize(G: MarkovGraph, rho: torch.Tensor) -> torch.Tensor:
    return rho / (rho @ torch.as_tensor(G.pi, dtype=_DTYPE))


def _solve(G, rhoA, rhoB, nsteps, tol, maxiters, floor_rtol, verbose, phi0_init, segments):
    a, b = _normalize(G, rhoA), _normalize(G, rhoB)
    z = _LogMapPotential.apply(a, b, G, nsteps, tol, maxiters, floor_rtol, verbose, phi0_init, segments)
    return a, _torch_potential(G, z)


def transport_cost_shooting_torch(
    G: MarkovGraph,
    rhoA: torch.Tensor,
    rhoB: torch.Tensor,
    *,
    nsteps: int = 150,
    tol: float = 1e-9,
    maxiters: int = 50,
    floor_rtol: float = 1e-6,
    verbose: bool = False,
    phi0_init=None,
    segments="auto",
) -> torch.Tensor:
    """W2 between rhoA and rhoB (densities w.r.t. G.pi, float64 tensors) by
    shooting, as a 0-d tensor differentiable to first order (see the module
    docstring)."""
    a, phi0 = _solve(G, rhoA, rhoB, nsteps, tol, maxiters, floor_rtol, verbose, phi0_init, segments)
    return 2 * _torch_hamiltonian(G, a, phi0)


def geodesic_shooting_torch(
    G: MarkovGraph,
    rhoA: torch.Tensor,
    rhoB: torch.Tensor,
    *,
    nsteps: int = 150,
    tol: float = 1e-9,
    maxiters: int = 50,
    floor_rtol: float = 1e-6,
    verbose: bool = False,
    phi0_init=None,
    segments="auto",
):
    """The shooting geodesic as a GeodesicSolution of differentiable tensors:
    W2, the density and potential paths, the momenta and the endpoint
    potentials all carry gradients back to rhoA and rhoB, to first order (see
    the module docstring). Same conventions as geodesic_shooting."""
    from graphtransport.api import GeodesicSolution

    t0 = time.perf_counter()
    a, phi0 = _solve(G, rhoA, rhoB, nsteps, tol, maxiters, floor_rtol, verbose, phi0_init, segments)
    _, _, schedule, _ = _torch_integrate(G, a.detach(), phi0.detach(), nsteps, 1.0, rho_floor(G, rtol=floor_rtol))
    rho_path, phi_path = _torch_replay_path(G, a, phi0, schedule)
    x, y, _, _ = _torch_edges(G)
    rho_t, phi_t = rho_path[:, :-1], phi_path[:, :-1]
    m = G.mean.torch_theta(rho_t[x], rho_t[y]) * (phi_t[x] - phi_t[y])
    W2 = 2 * _torch_hamiltonian(G, a, phi0)
    return GeodesicSolution(
        W2, rho_path, m, m[:, 0], -2 * phi0, 2 * phi_path[:, -1], "converged", time.perf_counter() - t0
    )
