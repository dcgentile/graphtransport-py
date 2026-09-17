"""Entropic Wasserstein barycenters by Sinkhorn iteration.

Ported from GraphTransportation.jl's sinkhorn/Sinkhorn.jl: the forward
iteration of Bonneel, Peyré & Cuturi (2016), Algorithm 1 (iterative Bregman
projections, Benamou et al. 2015). Measures are probability vectors on the
nodes (not densities w.r.t. pi -- the unified API converts). The gradient
with respect to the weights (the backward pass) is a separate step.
"""

from __future__ import annotations

import numpy as np


def regularize_cost(cost, epsilon: float) -> np.ndarray:
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise."""
    return np.exp(-np.asarray(cost, dtype=float) / epsilon)


def logarithmic_change_of_variable(coords) -> np.ndarray:
    """Softmax: project an unconstrained vector onto the simplex."""
    v = np.exp(np.asarray(coords, dtype=float))
    return v / v.sum()


def _sinkhorn_forward(coords, measures, K, iters: int):
    """Forward Sinkhorn iterations for the barycenter with weights `coords`.

    Index convention matches the Julia original: `iters` slots, with the
    initial scaling b^(0) = 1 in slot 0 and the Sinkhorn iterations in slots
    1..iters-1, so `iters` slots give L = iters - 1 iterations. The full
    per-slot history (b, phi) is kept for the backward pass.

    Returns (p, b, phi) with p of shape (n,), b and phi of shape (n, S, iters).
    """
    coords = np.asarray(coords, dtype=float)
    measures = np.asarray(measures, dtype=float)
    n, S = measures.shape
    b = np.ones((n, S, iters))
    phi = np.empty((n, S, iters))
    p = np.full(n, 1.0 / n)
    for l in range(1, iters):
        phi[:, :, l] = K.T @ (measures / (K @ b[:, :, l - 1]))
        p = np.exp(np.log(phi[:, :, l]) @ coords)
        b[:, :, l] = p[:, np.newaxis] / phi[:, :, l]
    return p, b, phi


def sinkhorn_barycenter(coords, measures, cost, epsilon: float, *, iters: int = 256) -> np.ndarray:
    """Entropic Wasserstein barycenter of the columns of `measures`
    (probability vectors, shape (n, S)) with weights `coords` (length S,
    summing to 1) for the ground `cost` and regularisation `epsilon`.

    `iters` is the Sinkhorn budget (slots; L = iters - 1 iterations, as in
    the Julia original, so results agree at equal `iters`). The result is
    not renormalised: it is a probability vector to solver tolerance once
    the iterations have converged.
    """
    K = regularize_cost(cost, epsilon)
    p, _, _ = _sinkhorn_forward(coords, measures, K, iters)
    return p


def sinkhorn_plan(K, mu, nu, *, iters: int = 256) -> np.ndarray:
    """Two-marginal entropic transport plan diag(u) K diag(v) between the
    probability vectors mu and nu for the kernel K, by Sinkhorn scaling.
    Used to evaluate the entropic objective <cost, P> of a barycenter."""
    mu = np.asarray(mu, dtype=float)
    nu = np.asarray(nu, dtype=float)
    u = np.ones_like(mu)
    v = np.ones_like(nu)
    for _ in range(iters):
        u = mu / (K @ v)
        v = nu / (K.T @ u)
    return u[:, np.newaxis] * K * v[np.newaxis, :]


def build_geodesic(measures, cost, *, epsilon: float = 0.1, steps: int = 10, iters: int = 2048) -> np.ndarray:
    """Entropic displacement interpolation between the two columns of
    `measures`: column i of the result is the barycenter at weights
    (1 - i/steps, i/steps), i = 0..steps. Shape (n, steps + 1)."""
    measures = np.asarray(measures, dtype=float)
    n, count = measures.shape
    if count != 2:
        raise ValueError(f"build_geodesic needs exactly 2 measures, got {count}")
    K = regularize_cost(cost, epsilon)
    path = np.empty((n, steps + 1))
    for i in range(steps + 1):
        t = i / steps
        path[:, i], _, _ = _sinkhorn_forward([1 - t, t], measures, K, iters)
    return path
