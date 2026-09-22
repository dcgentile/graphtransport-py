"""The SOCP analysis backend, ported from GraphTransportation.jl's socp/Analysis.jl."""

from __future__ import annotations

import numpy as np

from graphtransport.gram import potential_gram_qp, solve_barycentric_coordinates_qp
from graphtransport.graph import MarkovGraph, dense_metric_tensor
from graphtransport.socp.geodesic import geodesic_socp

CONVENTIONS = ("potential", "momentum")


def analyze_socp(G: MarkovGraph, target, refs, *, N: int = 10, solver=None, convention: str = "potential",
                 compute_condition: bool = False, return_system: bool = False, qp_method: str = "auto",
                 qp_solver=None, **solver_kwargs):
    """Recover the barycentric coordinates of ``target`` with respect to
    ``refs``: solve the geodesic SOCP from target to each reference, build the
    Gram matrix of the resulting tangent vectors at target, and solve the
    simplex QP min_{lam in simplex} lam^T A lam.

    convention="potential" (default): the Riemannian Gram matrix at target
    built from the endpoint potentials phi0 -- the exact gradient of
    W_h^2(target, ref_i). A barycenter_socp solution satisfies
    sum_i lam_i phi_i = const by its KKT conditions, so this recovers the
    synthesis weights to solver tolerance at any N, including N=2.

    convention="momentum": the initial momentum m0 as a dense antisymmetric
    tangent vector with the dense metric tensor at target (the
    Chambolle-Pock convention). m0 lives on the first time *interval*, so it
    is only an O(h) proxy for the endpoint potential; kept for comparison.

    ``qp_method`` and ``qp_solver`` go to the simplex QP (gram.simplex_qp) and
    are named apart from ``solver`` so the conic solver for the geodesics and
    the QP backend can be chosen separately; every other keyword is passed to
    the geodesic solver.

    Returns lam_hat, or (lam_hat, A) with return_system=True.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    target = np.asarray(target, dtype=float)
    geodesics = [geodesic_socp(G, target, ref, N=N, solver=solver, **solver_kwargs) for ref in refs]

    if convention == "potential":
        return potential_gram_qp(
            G, target, [geo.phi0 for geo in geodesics],
            compute_condition=compute_condition, return_system=return_system, method=qp_method,
            solver=qp_solver,
        )  # fmt: skip

    tangent_vectors = []
    for geo in geodesics:
        m_dense = np.zeros((G.n, G.n))
        m_dense[G.E[:, 0], G.E[:, 1]] = geo.m0
        m_dense[G.E[:, 1], G.E[:, 0]] = -geo.m0
        tangent_vectors.append(m_dense)
    g = dense_metric_tensor(target, G.mean)
    return solve_barycentric_coordinates_qp(
        tangent_vectors, g, compute_condition=compute_condition, return_system=return_system,
        method=qp_method, solver=qp_solver,
    )
