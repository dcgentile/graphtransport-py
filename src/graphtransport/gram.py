"""Shared analysis machinery: the simplex QP on a Gram matrix of tangent
vectors, used by every Gram-matrix analysis backend (SOCP, shooting).
Ported from GraphTransportation.jl's core/Analysis.jl.
"""

from __future__ import annotations

import numpy as np

from graphtransport.graph import MarkovGraph, graph_gradient, metric_tensor
from graphtransport.solvers import solve_conic


def gram_matrix(tangent_vectors, g) -> np.ndarray:
    """A[i, j] = sum(tangent_vectors[i] * tangent_vectors[j] * g), elementwise
    over whatever shape the tangent vectors have (edge vectors with an edge
    weighting, or dense antisymmetric matrices with a dense metric tensor)."""
    g = np.asarray(g, dtype=float)
    vecs = [np.asarray(v, dtype=float) for v in tangent_vectors]
    p = len(vecs)
    A = np.empty((p, p))
    for i in range(p):
        for j in range(i, p):
            A[i, j] = A[j, i] = float(np.sum(vecs[i] * vecs[j] * g))
    return A


def simplex_qp(A, *, solver=None) -> np.ndarray:
    """argmin_{lam >= 0, sum(lam) = 1} lam^T A lam for a symmetric PSD A."""
    import cvxpy as cp

    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2
    p = A.shape[0]
    x = cp.Variable(p)
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(A))), [x >= 0, cp.sum(x) == 1])
    solve_conic(problem, solver, "simplex_qp")
    lam = np.asarray(x.value, dtype=float).reshape(p)
    return lam


def solve_barycentric_coordinates_qp(tangent_vectors, g, *, compute_condition: bool = False,
                                     return_system: bool = False, solver=None):
    """Gram-matrix assembly and simplex-QP solve shared by every analysis
    backend: given the initial tangent vectors of the geodesics from a target
    to each reference and the target's metric tensor g, assemble
    A[i, j] = sum(tangent_vectors[i] * tangent_vectors[j] * g) and solve
    min_{lam in simplex} lam^T A lam. Returns lam, or (lam, A) with
    return_system=True. compute_condition prints the condition number of A,
    as the Julia original does."""
    A = gram_matrix(tangent_vectors, g)
    if compute_condition:
        e = np.abs(np.linalg.eigvalsh(A))
        print(f"Estimated condition number of analysis matrix: {e.max() / e.min()}")
    lam = simplex_qp(A, solver=solver)
    return (lam, A) if return_system else lam


def potential_gram_qp(G: MarkovGraph, target, potentials, *, compute_condition: bool = False,
                      return_system: bool = False, solver=None):
    """Gram matrix and simplex QP for potential-based backends: given one
    potential phi_i per reference (the geodesic from target to ref_i, in any
    sign/scale convention common to all i), assemble the Riemannian Gram
    matrix at the target,
    A_ij = sum_e kappa_e theta(target)_e (grad phi_i)_e (grad phi_j)_e
    (theta = G.mean), and solve min_{lam in simplex} lam^T A lam."""
    tangent_vectors = [graph_gradient(G, phi) for phi in potentials]
    g = G.kappa * metric_tensor(G, target)
    return solve_barycentric_coordinates_qp(
        tangent_vectors, g, compute_condition=compute_condition, return_system=return_system, solver=solver
    )
