"""Autograd-native re-expressions of the forward Sinkhorn barycenter.

Each backend module exposes the same small interface --
``regularize_cost(cost, epsilon)`` and
``sinkhorn_barycenter(coords, measures, cost, epsilon, *, iters=256)`` --
built from that framework's array ops, so the barycenter is differentiable
with respect to the weights (and the measures) by the framework's own
autograd. Callers pick a backend by which module they import. torch is a
core dependency; JAX is optional (``pip install graphtransport[jax]``), and
importing the JAX module without it raises an ImportError with an install
hint.

The numpy implementation in ``graphtransport.sinkhorn.core`` carries the
hand-derived backward pass and remains the reference; the backends are
checked against it in the tests.
"""
