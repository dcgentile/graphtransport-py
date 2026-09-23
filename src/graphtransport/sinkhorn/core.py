"""Entropic Wasserstein barycenters by Sinkhorn iteration.

Ported from GraphTransportation.jl's sinkhorn/Sinkhorn.jl: the forward
iteration of Bonneel, Peyré & Cuturi (2016), Algorithm 1 (iterative Bregman
projections, Benamou et al. 2015). Measures are probability vectors on the
nodes (not densities w.r.t. pi -- the unified API converts).

`sinkhorn_differentiate` is the same paper's backward pass -- the gradient of
the finite-iteration barycenter loss with respect to the weights -- and
`simplex_regression` uses it to recover barycentric coordinates.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize


def regularize_cost(cost, epsilon: float) -> np.ndarray:
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise."""
    _check_epsilon(epsilon)
    return np.exp(-np.asarray(cost, dtype=float) / epsilon)


# Input checks shared with the torch and jax backends, so that every
# implementation rejects the same inputs with the same message. They only
# look at Python scalars and static shapes, never at array values.


def _check_epsilon(epsilon) -> None:
    if not (np.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon!r}")


def _check_problem(measures_shape, n: int, coords_shape, iters) -> None:
    measures_shape, coords_shape = tuple(measures_shape), tuple(coords_shape)
    if len(measures_shape) != 2 or measures_shape[0] != n:
        raise ValueError(
            f"measures must have shape (n, S) with one column per measure and n = {n} nodes, "
            f"got {measures_shape}"
        )
    S = measures_shape[1]
    if coords_shape != (S,):
        raise ValueError(f"coords must have one weight per measure ({S}), got shape {coords_shape}")
    if isinstance(iters, bool) or not isinstance(iters, (int, np.integer)):
        raise TypeError(f"iters must be a Python int, got {type(iters).__name__}")
    if iters < 2:
        raise ValueError(f"iters must be at least 2 (iters slots give iters - 1 iterations), got {iters}")


def _underflow_error(what: str) -> FloatingPointError:
    return FloatingPointError(
        f"{what}: the Sinkhorn scalings are not finite. The kernel exp(-cost / epsilon) underflowed "
        "to zero where a measure has no mass nearby; use a larger epsilon (relative to the scale of "
        "the cost) or measures with full support."
    )


def logarithmic_change_of_variable(coords) -> np.ndarray:
    """Softmax: project an unconstrained vector onto the simplex."""
    coords = np.asarray(coords, dtype=float)
    v = np.exp(coords - coords.max())  # shifted so large coordinates cannot overflow
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
    _check_problem(measures.shape, K.shape[0], coords.shape, iters)
    n, S = measures.shape
    b = np.ones((n, S, iters))
    phi = np.empty((n, S, iters))
    p = np.full(n, 1.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for l in range(1, iters):
            phi[:, :, l] = K.T @ (measures / (K @ b[:, :, l - 1]))
            p = np.exp(np.log(phi[:, :, l]) @ coords)
            b[:, :, l] = p[:, np.newaxis] / phi[:, :, l]
    if not np.all(np.isfinite(p)):
        raise _underflow_error("sinkhorn_barycenter")
    return p, b, phi


def sinkhorn_barycenter(coords, measures, cost, epsilon: float, *, iters: int = 256) -> np.ndarray:
    """Entropic Wasserstein barycenter of the columns of `measures`
    (probability vectors, shape (n, S)) with weights `coords` (length S,
    summing to 1) for the ground `cost` and regularization `epsilon`.

    `iters` is the Sinkhorn budget (slots; L = iters - 1 iterations, as in
    the Julia original, so results agree at equal `iters`). The result is
    not renormalized: it is a probability vector to solver tolerance once
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
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iters):
            u = mu / (K @ v)
            v = nu / (K.T @ u)
    plan = u[:, np.newaxis] * K * v[np.newaxis, :]
    if not np.all(np.isfinite(plan)):
        raise _underflow_error("sinkhorn_plan")
    return plan


def build_geodesic(measures, cost, *, epsilon: float = 0.1, steps: int = 10, iters: int = 2048) -> np.ndarray:
    """Entropic displacement interpolation between the two columns of
    `measures`: column i of the result is the barycenter at weights
    (1 - i/steps, i/steps), i = 0..steps. Shape (n, steps + 1)."""
    measures = np.asarray(measures, dtype=float)
    n, count = measures.shape
    if count != 2:
        raise ValueError(f"build_geodesic needs exactly 2 measures, got {count}")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"steps must be an integer >= 1, got {steps!r}")
    K = regularize_cost(cost, epsilon)
    path = np.empty((n, steps + 1))
    for i in range(steps + 1):
        t = i / steps
        path[:, i], _, _ = _sinkhorn_forward([1 - t, t], measures, K, iters)
    return path


def sqeuc_loss(p, q) -> float:
    """Squared Euclidean loss 1/2 ||p - q||^2 between histograms."""
    d = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    return 0.5 * float(d @ d)


def sinkhorn_differentiate(coords, measures, target, cost, epsilon: float, iters: int):
    """Algorithm 1 of Bonneel, Peyré & Cuturi (2016): the barycenter p of
    `measures` with weights `coords`, and -- if `target` is not None -- the
    gradient w of the finite-L loss E_L(coords) = 1/2 ||p - target||^2 with
    respect to `coords`, by reverse-mode differentiation through all L =
    iters - 1 Sinkhorn iterations. Returns (p, w), with w = None when
    `target` is None.
    """
    coords = np.asarray(coords, dtype=float)
    measures = np.asarray(measures, dtype=float)
    K = regularize_cost(cost, epsilon)
    p, b, phi = _sinkhorn_forward(coords, measures, K, iters)
    if target is None:
        return p, None

    n, S = measures.shape
    target = np.asarray(target, dtype=float)
    if target.shape != (n,):
        raise ValueError(f"target must be a probability vector of shape ({n},), got {target.shape}")
    w = np.zeros(S)
    r = np.zeros((n, S))
    g = (p - target) * p
    # Reverse over every forward iteration, slots iters-1 .. 1. Starting one
    # slot lower would drop the top term of the sum, which gives a wrong
    # gradient at small L (wrong sign at L = 2).
    for l in range(iters - 1, 0, -1):
        for m in range(S):
            w[m] += np.log(phi[:, m, l]) @ g
            u = coords[m] * g - r[:, m]
            x = K @ (u / phi[:, m, l])
            y = measures[:, m] / (K @ b[:, m, l - 1]) ** 2
            r[:, m] = -(K.T @ (x * y)) * b[:, m, l - 1]
        g = r.sum(axis=1)
    return p, w


def barycentric_loss(alpha, measures, target, cost, epsilon: float, *, iters: int = 256) -> float:
    """The regression objective E_L (Bonneel et al., Eq. 12) with the squared
    Euclidean loss, as a function of the unconstrained variable alpha through
    the softmax change of variables lambda = softmax(alpha). Use the same
    `iters` for the objective and its gradient (`loss_gradient`)."""
    p, _ = sinkhorn_differentiate(logarithmic_change_of_variable(alpha), measures, None, cost, epsilon, iters)
    return sqeuc_loss(p, target)


def loss_gradient(alpha, measures, target, cost, epsilon: float, *, iters: int = 256) -> np.ndarray:
    """Gradient of `barycentric_loss` with respect to alpha: Algorithm 1's
    w = grad_lambda E_L, pushed through the softmax Jacobian,
    grad_alpha E = lambda * (w - <lambda, w>).

    Note: the Julia original's argument order is (alpha, measures, cost,
    target, ...); here it is (alpha, measures, target, cost, ...) to match
    `barycentric_loss`."""
    return _loss_and_gradient(alpha, measures, target, cost, epsilon, iters)[1]


def _loss_and_gradient(alpha, measures, target, cost, epsilon: float, iters: int):
    """(barycentric_loss, loss_gradient) at alpha from a single forward pass."""
    if target is None:
        raise ValueError("the regression loss needs a target")
    lam = logarithmic_change_of_variable(alpha)
    p, w = sinkhorn_differentiate(lam, measures, target, cost, epsilon, iters)
    return sqeuc_loss(p, target), lam * (w - lam @ w)


def simplex_regression(measures, target, cost, epsilon: float, *, iters: int = 256, alpha0=None, **minimize_options) -> np.ndarray:
    """Wasserstein barycentric coordinates of `target` with respect to the
    columns of `measures` (Bonneel, Peyré & Cuturi 2016, §4.3): minimize
    E_L(lambda) = 1/2 ||P^(L)(lambda) - target||^2 over the simplex by
    L-BFGS on alpha with lambda = softmax(alpha), using the analytic gradient
    from `sinkhorn_differentiate`. alpha0 = 0 (the default) is the paper's
    lambda0 = 1/S. Extra keyword arguments override scipy's L-BFGS-B
    `options`. Returns lambda-hat on the simplex, and warns if L-BFGS-B ran
    out of iterations or evaluations before converging.

    The objective is O(1e-10) near a recoverable optimum, below L-BFGS-B's
    default absolute `ftol` (2.2e-9), so by default the stopping test is
    the gradient norm alone (ftol=0, gtol=1e-12), which recovers synthesis
    weights to ~1e-8 like the Julia original's Optim.jl defaults."""
    measures = np.asarray(measures, dtype=float)
    S = measures.shape[1]
    alpha0 = np.zeros(S) if alpha0 is None else np.asarray(alpha0, dtype=float)
    options = {"ftol": 0.0, "gtol": 1e-12, "maxiter": 1000, **minimize_options}
    # One function returns loss and gradient (jac=True), so each point costs a
    # single Sinkhorn forward pass rather than two.
    result = minimize(
        _loss_and_gradient,
        alpha0,
        args=(measures, target, cost, epsilon, iters),
        jac=True,
        method="L-BFGS-B",
        options=options,
    )
    # Only the budget running out is reported: with ftol=0 L-BFGS-B routinely
    # ends on "abnormal termination in line search" at a good optimum, so
    # result.success is not a usable convergence test here.
    if result.status == 1:
        warnings.warn(
            f"simplex_regression: L-BFGS-B stopped at its iteration/evaluation limit after {result.nit} "
            f"iterations (loss {result.fun:.3e}); the weights may not have converged. Raise maxiter.",
            stacklevel=2,
        )
    return logarithmic_change_of_variable(result.x)
