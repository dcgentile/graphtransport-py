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
        "graphtransport.sinkhorn.backends.torch_backend requires PyTorch; "
        "install it with `pip install 'graphtransport[torch]'`"
    ) from exc


def regularize_cost(cost, epsilon: float):
    """The Gibbs kernel K = exp(-cost / epsilon), elementwise."""
    return torch.exp(-torch.as_tensor(cost) / epsilon)


def sinkhorn_barycenter(coords, measures, cost, epsilon: float, *, iters: int = 256):
    """Entropic Wasserstein barycenter of the columns of ``measures`` (shape
    (n, S), probability vectors) with weights ``coords`` (length S) for the
    ground ``cost`` and regularisation ``epsilon``; differentiable in
    ``coords``, ``measures`` and ``cost``.

    Inputs may be tensors or array-likes; they are promoted to the dtype and
    device of ``measures``. Returns a tensor of shape (n,).
    """
    measures = torch.as_tensor(measures)
    coords = torch.as_tensor(coords, dtype=measures.dtype, device=measures.device)
    K = regularize_cost(torch.as_tensor(cost, dtype=measures.dtype, device=measures.device), epsilon)
    n = measures.shape[0]
    b = torch.ones_like(measures)
    p = torch.full((n,), 1.0 / n, dtype=measures.dtype, device=measures.device)
    for _ in range(iters - 1):
        phi = K.T @ (measures / (K @ b))
        p = torch.exp(torch.log(phi) @ coords)
        b = p[:, None] / phi
    return p
