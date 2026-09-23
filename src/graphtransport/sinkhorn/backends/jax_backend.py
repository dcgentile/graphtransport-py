"""JAX backend: the forward Sinkhorn barycenter as differentiable, jit-able ops.

Same iteration and slot convention as ``graphtransport.sinkhorn.core``
(``iters`` slots = ``iters - 1`` Sinkhorn iterations), so results agree with
the numpy implementation at equal ``iters``; gradients come from
``jax.grad`` rather than the hand-derived backward pass. The loop is a
``lax.fori_loop`` with a static trip count, so ``jax.jit`` and reverse-mode
differentiation both work (``iters`` must be a Python int, not traced).

JAX computes in float32 unless ``jax.config.update("jax_enable_x64", True)``
is set by the caller; this module does not change that global setting. In
float32 even float64 inputs are silently computed and returned in float32,
so agreement with the numpy implementation to rounding error needs x64.

Reverse-mode differentiation keeps every iteration's residuals, so its
memory is O(iters * n * S).

Checks under ``jit``: Python-level arguments (``iters``, a concrete
``epsilon``, the static shapes) are validated at trace time. A non-finite
result raises FloatingPointError only when the call is eager; inside ``jit``,
``grad`` or ``vmap`` the value is traced and cannot be inspected, so nan
passes through -- use ``jax.config.update("jax_debug_nans", True)`` or
``jax.experimental.checkify`` there.
"""

from __future__ import annotations

try:
    import jax
    import jax.core
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "graphtransport.sinkhorn.backends.jax_backend requires JAX; "
        "install it with `pip install 'graphtransport[jax]'`"
    ) from exc

from graphtransport.sinkhorn.core import _check_epsilon, _check_problem, _underflow_error


def _is_traced(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _as_float_array(x, dtype=None):
    """x as an array; a floating array keeps its dtype, anything else (ints,
    lists of ints) becomes JAX's default float, which follows jax_enable_x64."""
    x = jnp.asarray(x, dtype=dtype)
    return x if jnp.issubdtype(x.dtype, jnp.floating) else x.astype(jnp.result_type(float))


def regularize_cost(cost, epsilon):
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise. ``epsilon`` may
    be traced (it is differentiable and jit-able); a concrete value is validated."""
    if not _is_traced(epsilon):
        _check_epsilon(epsilon)
    return jnp.exp(-_as_float_array(cost) / epsilon)


def sinkhorn_barycenter(coords, measures, cost, epsilon, *, iters: int = 256):
    """Entropic Wasserstein barycenter of the columns of ``measures`` (shape
    (n, S), probability vectors) with weights ``coords`` (length S) for the
    ground ``cost`` and regularization ``epsilon``; differentiable in
    ``coords``, ``measures`` and ``cost``.

    Inputs may be arrays or array-likes; they are promoted to the dtype of
    ``measures`` (JAX's default float if ``measures`` is not floating).
    Returns an array of shape (n,). ``iters`` must be a Python int. See the
    module docstring for what is and is not checked under ``jit``.
    """
    measures = _as_float_array(measures)
    coords = _as_float_array(coords, dtype=measures.dtype)
    K = regularize_cost(_as_float_array(cost, dtype=measures.dtype), epsilon)
    _check_problem(measures.shape, K.shape[0], coords.shape, iters)
    n = measures.shape[0]

    def body(_, state):
        p, b = state
        phi = K.T @ (measures / (K @ b))
        p = jnp.exp(jnp.log(phi) @ coords)
        return p, p[:, None] / phi

    p0 = jnp.full((n,), 1.0 / n, dtype=measures.dtype)
    b0 = jnp.ones_like(measures)
    p, _ = jax.lax.fori_loop(0, iters - 1, body, (p0, b0))
    if not _is_traced(p) and not bool(jnp.isfinite(p).all()):
        raise _underflow_error("sinkhorn_barycenter")
    return p
