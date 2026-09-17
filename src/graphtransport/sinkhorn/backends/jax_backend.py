"""JAX backend: the forward Sinkhorn barycenter as differentiable, jit-able ops.

Same iteration and slot convention as ``graphtransport.sinkhorn.core``
(``iters`` slots = ``iters - 1`` Sinkhorn iterations), so results agree with
the numpy implementation at equal ``iters``; gradients come from
``jax.grad`` rather than the hand-derived backward pass. The loop is a
``lax.fori_loop`` with a static trip count, so ``jax.jit`` and reverse-mode
differentiation both work (``iters`` must be a Python int, not traced).

JAX computes in float32 unless ``jax.config.update("jax_enable_x64", True)``
is set by the caller; this module does not change that global setting.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "graphtransport.sinkhorn.backends.jax_backend requires JAX; "
        "install it with `pip install 'graphtransport[jax]'`"
    ) from exc


def regularize_cost(cost, epsilon: float):
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise."""
    return jnp.exp(-jnp.asarray(cost) / epsilon)


def sinkhorn_barycenter(coords, measures, cost, epsilon: float, *, iters: int = 256):
    """Entropic Wasserstein barycenter of the columns of ``measures`` (shape
    (n, S), probability vectors) with weights ``coords`` (length S) for the
    ground ``cost`` and regularisation ``epsilon``; differentiable in
    ``coords``, ``measures`` and ``cost``.

    Inputs may be arrays or array-likes; they are promoted to the dtype of
    ``measures``. Returns an array of shape (n,).
    """
    measures = jnp.asarray(measures)
    coords = jnp.asarray(coords, dtype=measures.dtype)
    K = regularize_cost(jnp.asarray(cost, dtype=measures.dtype), epsilon)
    n = measures.shape[0]

    def body(_, state):
        p, b = state
        phi = K.T @ (measures / (K @ b))
        p = jnp.exp(jnp.log(phi) @ coords)
        return p, p[:, None] / phi

    p0 = jnp.full((n,), 1.0 / n, dtype=measures.dtype)
    b0 = jnp.ones_like(measures)
    p, _ = jax.lax.fori_loop(0, iters - 1, body, (p0, b0))
    return p
