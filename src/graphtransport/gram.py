"""Shared analysis machinery: the simplex QP on a Gram matrix of tangent
vectors, used by every Gram-matrix analysis backend (SOCP, shooting).
Ported from GraphTransportation.jl's core/Analysis.jl.
"""

from __future__ import annotations

import numpy as np

from graphtransport.graph import MarkovGraph, graph_gradient, metric_tensor
from graphtransport.solvers import cvxpy_available, import_cvxpy, solve_conic


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


QP_METHODS = ("auto", "cvxpy", "scipy")


def simplex_qp(A, *, method: str = "auto", solver=None) -> np.ndarray:
    """argmin_{lam >= 0, sum(lam) = 1} lam^T A lam for a symmetric PSD A.

    method="cvxpy" solves it as a conic program (``solver`` picks the conic
    solver; needs the ``socp`` extra); method="scipy" uses SLSQP with the
    analytic gradient and needs nothing beyond scipy; "auto" (default) is
    cvxpy when installed, otherwise scipy. Both agree to the conic solver's
    tolerance (~1e-8).

    The minimiser is invariant under A -> c A (c > 0), so A is divided by its
    largest entry first: a conic solver stops on absolute tolerances, and a
    Gram matrix of nearby measures (entries ~ W^2, possibly 1e-8 or less)
    would otherwise be "solved" at its starting point, uniform weights. A zero
    A makes every lam optimal and returns uniform weights. The result is
    projected onto the simplex (clip at 0, renormalise) to remove solver
    round-off. Raises if A is not square, not finite, or not PSD.
    """
    if method not in QP_METHODS:
        raise ValueError(f"simplex_qp: method must be one of {QP_METHODS}, got {method!r}")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError(f"simplex_qp: A must be a nonempty square matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("simplex_qp: A has non-finite entries")
    A = (A + A.T) / 2
    p = A.shape[0]
    scale = np.abs(A).max()
    if p == 1 or scale == 0:
        return np.full(p, 1.0 / p)
    A = A / scale
    eig = np.linalg.eigvalsh(A)
    if eig[0] < -1e-8 * max(eig[-1], 1.0):
        raise ValueError(
            f"simplex_qp: A is not positive semidefinite (smallest eigenvalue {eig[0] * scale:.3e}); "
            "a Gram matrix of tangent vectors always is"
        )
    if method == "auto":
        method = "cvxpy" if cvxpy_available() else "scipy"
    lam = _simplex_qp_cvxpy(A, solver) if method == "cvxpy" else _simplex_qp_scipy(A)
    lam = np.maximum(lam, 0.0)
    return lam / lam.sum()


def _simplex_qp_cvxpy(A, solver) -> np.ndarray:
    cp = import_cvxpy()
    p = A.shape[0]
    x = cp.Variable(p)
    # psd_wrap skips cvxpy's own PSD check, which round-off negative
    # eigenvalues of a Gram matrix would fail; simplex_qp checked A above.
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(A))), [x >= 0, cp.sum(x) == 1])
    solve_conic(problem, solver, "simplex_qp", hint="Try method='scipy'.")
    return np.asarray(x.value, dtype=float).reshape(p)


def _simplex_qp_scipy(A) -> np.ndarray:
    from scipy.optimize import minimize

    p = A.shape[0]
    result = minimize(
        lambda x: x @ A @ x,
        np.full(p, 1.0 / p),
        jac=lambda x: 2 * A @ x,
        method="SLSQP",
        bounds=[(0.0, None)] * p,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(p)}],
        options={"ftol": 1e-15, "maxiter": 50 * p + 100},
    )
    if not result.success:
        raise RuntimeError(f"simplex_qp: SLSQP did not converge ({result.message}); try method='cvxpy'")
    return result.x


def solve_barycentric_coordinates_qp(tangent_vectors, g, *, compute_condition: bool = False,
                                     return_system: bool = False, method: str = "auto", solver=None):
    """Gram-matrix assembly and simplex-QP solve shared by every analysis
    backend: given the initial tangent vectors of the geodesics from a target
    to each reference and the target's metric tensor g, assemble
    A[i, j] = sum(tangent_vectors[i] * tangent_vectors[j] * g) and solve
    min_{lam in simplex} lam^T A lam. Returns lam, or (lam, A) with
    return_system=True. compute_condition prints the condition number of A,
    as the Julia original does (or, when A is numerically singular -- as it is
    when the target is exactly a barycenter -- its rank). ``method`` and
    ``solver`` are passed to simplex_qp."""
    A = gram_matrix(tangent_vectors, g)
    if compute_condition:
        print(_describe_condition(A))
    lam = simplex_qp(A, method=method, solver=solver)
    return (lam, A) if return_system else lam


def _describe_condition(A) -> str:
    e = np.linalg.eigvalsh(A)
    top = np.abs(e).max()
    if top == 0:
        return "Analysis matrix is zero (every reference coincides with the target)"
    rank = int(np.sum(e > 1e-12 * top))
    if rank < len(e):
        return f"Analysis matrix is numerically singular: rank {rank} of {len(e)}"
    return f"Estimated condition number of analysis matrix: {e[-1] / e[0]}"


def potential_gram_qp(G: MarkovGraph, target, potentials, *, compute_condition: bool = False,
                      return_system: bool = False, method: str = "auto", solver=None):
    """Gram matrix and simplex QP for potential-based backends: given one
    potential phi_i per reference (the geodesic from target to ref_i, in any
    sign/scale convention common to all i), assemble the Riemannian Gram
    matrix at the target,
    A_ij = sum_e kappa_e theta(target)_e (grad phi_i)_e (grad phi_j)_e
    (theta = G.mean), and solve min_{lam in simplex} lam^T A lam."""
    tangent_vectors = [graph_gradient(G, phi) for phi in potentials]
    g = G.kappa * metric_tensor(G, target)
    return solve_barycentric_coordinates_qp(
        tangent_vectors, g, compute_condition=compute_condition, return_system=return_system, method=method,
        solver=solver,
    )
