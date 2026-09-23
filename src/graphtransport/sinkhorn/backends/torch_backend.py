"""PyTorch backend: the forward Sinkhorn barycenter as differentiable tensor ops.

Same iteration and slot convention as ``graphtransport.sinkhorn.core``
(``iters`` slots = ``iters - 1`` Sinkhorn iterations), so results agree with
the numpy implementation at equal ``iters``; gradients come from autograd
rather than the hand-derived backward pass.
"""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "graphtransport.sinkhorn.backends.torch_backend requires PyTorch; install it with `pip install torch`"
    ) from exc

from graphtransport.sinkhorn.core import _check_epsilon, _check_problem, _underflow_error


def _as_float_tensor(x, **like):
    """x as a tensor; a floating tensor keeps its dtype (float32 on a GPU stays
    float32), anything else -- ints, lists, numpy ints -- becomes float64, as
    the numpy implementation would make it."""
    if like:
        return torch.as_tensor(x, **like)
    if isinstance(x, torch.Tensor):
        return x if x.is_floating_point() else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def regularize_cost(cost, epsilon):
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise. ``epsilon`` may
    be a tensor (it is differentiable); a plain number is validated."""
    if not isinstance(epsilon, torch.Tensor):
        _check_epsilon(epsilon)
    return torch.exp(-_as_float_tensor(cost) / epsilon)


def sinkhorn_barycenter(coords, measures, cost, epsilon, *, iters: int = 256, check: bool = True):
    """Entropic Wasserstein barycenter of the columns of ``measures`` (shape
    (n, S), probability vectors) with weights ``coords`` (length S) for the
    ground ``cost`` and regularization ``epsilon``; differentiable in
    ``coords``, ``measures`` and ``cost``.

    Inputs may be tensors or array-likes; they are promoted to the dtype and
    device of ``measures`` (float64 if ``measures`` is not a floating tensor).
    Returns a tensor of shape (n,).

    With ``check=True`` (default) a non-finite result -- the kernel
    underflowed; see ``graphtransport.sinkhorn.core`` -- raises
    FloatingPointError instead of returning nan. The test reads the result
    back from the device, which forces a synchronization on CUDA; pass
    ``check=False`` in a training loop where that matters.
    """
    measures = _as_float_tensor(measures)
    like = {"dtype": measures.dtype, "device": measures.device}
    coords = _as_float_tensor(coords, **like)
    K = regularize_cost(_as_float_tensor(cost, **like), epsilon)
    _check_problem(measures.shape, K.shape[0], coords.shape, iters)
    n = measures.shape[0]
    b = torch.ones_like(measures)
    p = torch.full((n,), 1.0 / n, dtype=measures.dtype, device=measures.device)
    for _ in range(iters - 1):
        phi = K.T @ (measures / (K @ b))
        p = torch.exp(torch.log(phi) @ coords)
        b = p[:, None] / phi
    if check and not bool(torch.isfinite(p).all()):
        raise _underflow_error("sinkhorn_barycenter")
    return p
