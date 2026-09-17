"""Admissible means (mobilities) defining a discrete transport metric.

Ported from GraphTransportation.jl's core/Means.jl. An admissible mean
theta(s, t) is continuous, symmetric, positively 1-homogeneous, concave,
positive on (0, inf)^2 and normalised so theta(s, s) = s (Maas 2011). The
metric is ||grad phi||^2_rho = sum_e kappa_e theta(rho_x, rho_y) (grad phi)_e^2;
concavity is what makes the action m^2/theta jointly convex.

Every mean is callable, theta(s, t), and has partial_s(s, t); partial_t
follows by symmetry. All are vectorized over numpy arrays (the Julia
originals are scalar, generic for ForwardDiff) since they are evaluated
once per graph edge.

For all s, t > 0: HarmonicMean <= GeometricMean <= LogarithmicMean <=
ArithmeticMean. The arithmetic mean is the only one with theta(0, t) != 0,
so it is the only one under which mass can flow out of an empty node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Below this |s/t - 1| the logarithmic mean and its derivative switch to a
# Taylor series so both are smooth through s == t (0/0 in the closed form).
_LOG_SERIES_SWITCH = 1e-3


class AdmissibleMean(ABC):
    @abstractmethod
    def __call__(self, s, t) -> np.ndarray:
        """theta(s, t)."""

    @abstractmethod
    def partial_s(self, s, t) -> np.ndarray:
        """d theta / d s (s, t)."""

    def partial_t(self, s, t) -> np.ndarray:
        """d theta / d t (s, t), i.e. partial_s(t, s) by symmetry."""
        return self.partial_s(t, s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class GeometricMean(AdmissibleMean):
    """theta(s, t) = sqrt(s t). The package default (Erbar et al. 2020)."""

    def __call__(self, s, t):
        return np.sqrt(np.asarray(s, dtype=float) * np.asarray(t, dtype=float))

    def partial_s(self, s, t):
        return np.sqrt(np.asarray(t, dtype=float) / np.asarray(s, dtype=float)) / 2


class ArithmeticMean(AdmissibleMean):
    """theta(s, t) = (s + t) / 2. Does not vanish at an empty node."""

    def __call__(self, s, t):
        return (np.asarray(s, dtype=float) + np.asarray(t, dtype=float)) / 2

    def partial_s(self, s, t):
        return np.full(np.broadcast(np.asarray(s), np.asarray(t)).shape, 0.5)


class HarmonicMean(AdmissibleMean):
    """theta(s, t) = 2 s t / (s + t). The smallest of the four; bounded partial_s near zero."""

    def __call__(self, s, t):
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        return 2 * s * t / (s + t)

    def partial_s(self, s, t):
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        return 2 * t**2 / (s + t) ** 2


class LogarithmicMean(AdmissibleMean):
    """theta(s, t) = (s - t) / (ln s - ln t). The mean for which the heat flow
    is the gradient flow of the relative entropy (Maas 2011). Not finitely
    conic-representable; the SOCP uses QuadLogMean instead."""

    def __call__(self, s, t):
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        delta = s / t - 1
        # (x-1)/ln x = 1 + d/2 - d^2/12 + d^3/24 - 19 d^4/720 + 3 d^5/160 + O(d^6)
        series = t * (1 + delta * (1 / 2 + delta * (-1 / 12 + delta * (1 / 24 + delta * (-19 / 720 + delta * 3 / 160)))))
        with np.errstate(divide="ignore", invalid="ignore"):
            closed = t * delta / np.log1p(delta)
        return np.where(np.abs(delta) < _LOG_SERIES_SWITCH, series, closed)

    def partial_s(self, s, t):
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        delta = s / t - 1
        # d/dx (x-1)/ln x = 1/2 - d/6 + d^2/8 - 19 d^3/180 + 3 d^4/32 + O(d^5)
        series = 1 / 2 + delta * (-1 / 6 + delta * (1 / 8 + delta * (-19 / 180 + delta * 3 / 32)))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log1p(delta)
            closed = (L - delta / (1 + delta)) / L**2
        return np.where(np.abs(delta) < _LOG_SERIES_SWITCH, series, closed)


class QuadLogMean(AdmissibleMean):
    """Gauss-Legendre approximation of the logarithmic mean with K nodes,
    Lambda_K(s, t) = sum_k w_k s^a_k t^(1 - a_k) ~ int_0^1 s^a t^(1-a) da.

    Lambda_K is itself an admissible mean, so a metric built on it is exact,
    not approximate; each term is a power-cone constraint, which is how the
    SOCP represents the logarithmic mean. Worst relative error against
    LogarithmicMean over density ratios up to 1000: 8.5e-4 (K=4), 6e-7 (K=6),
    1.2e-10 (K=8), 2e-15 (K=12).
    """

    def __init__(self, K: int = 8):
        if K < 1:
            raise ValueError("QuadLogMean needs K >= 1 nodes")
        # numpy's leggauss gives nodes on [-1, 1] with weights summing to 2;
        # map to [0, 1] and normalise the weights to 1.
        x, w = np.polynomial.legendre.leggauss(K)
        self.alpha = (x + 1) / 2
        self.w = w / w.sum()

    @property
    def K(self) -> int:
        return len(self.alpha)

    def __call__(self, s, t):
        s = np.asarray(s, dtype=float)[..., np.newaxis]
        t = np.asarray(t, dtype=float)[..., np.newaxis]
        return (self.w * s**self.alpha * t ** (1 - self.alpha)).sum(axis=-1)

    def partial_s(self, s, t):
        s = np.asarray(s, dtype=float)[..., np.newaxis]
        t = np.asarray(t, dtype=float)[..., np.newaxis]
        return (self.w * self.alpha * s ** (self.alpha - 1) * t ** (1 - self.alpha)).sum(axis=-1)

    def __repr__(self) -> str:
        return f"QuadLogMean({self.K})"

    def __eq__(self, other) -> bool:
        return isinstance(other, QuadLogMean) and other.K == self.K

    def __hash__(self) -> int:
        return hash((QuadLogMean, self.K))
