"""Admissible means used by the graph Wasserstein metric tensor.

Ported from the "ADMISSIBLE MEANS" section of GraphTransportation.jl's
core/GraphCalculus.jl. Vectorized over numpy arrays (the Julia originals are
scalar-only) since these are evaluated once per graph edge.
"""

from __future__ import annotations

import numpy as np


def geomean(x, y):
    """Geometric mean sqrt(x*y). Returns -inf if either argument is negative."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    negative = (x < 0) | (y < 0)
    with np.errstate(invalid="ignore"):
        result = np.sqrt(np.where(negative, 0.0, x * y))
    return np.where(negative, -np.inf, result)


def logmean(s, t, *, tol: float = 1e-5):
    """Logarithmic mean L(s, t) = (s - t) / (log s - log t).

    Returns -inf if either argument is negative, and the first-order Taylor
    approximation (s + t) / 2 when |s - t| <= tol (this includes s == t,
    where the exact formula is a removable 0/0 singularity).
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    negative = (s < 0) | (t < 0)
    near_diagonal = np.abs(s - t) <= tol
    with np.errstate(divide="ignore", invalid="ignore"):
        formula = (s - t) / (np.log(s) - np.log(t))
    result = np.where(near_diagonal, (s + t) / 2, formula)
    return np.where(negative, -np.inf, result)


def logmean_partial_s(s, t, *, tol: float = 1e-5):
    """Partial derivative of logmean with respect to s."""
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    near_diagonal = np.abs(s - t) < tol
    with np.errstate(divide="ignore", invalid="ignore"):
        formula = (-s + t + s * np.log(s) - s * np.log(t)) / (s * (np.log(s) - np.log(t)) ** 2)
    return np.where(near_diagonal, 0.5, formula)


def logmean_partial_t(s, t, *, tol: float = 1e-5):
    """Partial derivative of logmean with respect to t."""
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    near_diagonal = np.abs(s - t) < tol
    with np.errstate(divide="ignore", invalid="ignore"):
        formula = (s - t - t * np.log(s) + t * np.log(t)) / (t * (np.log(s) - np.log(t)) ** 2)
    return np.where(near_diagonal, 0.5, formula)
