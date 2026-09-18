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

Boundary conventions, shared by every mean: theta extends continuously to
s = 0 or t = 0 (so theta(0, 0) = 0); partial_s takes its limiting value
there, which may be +inf, except at (0, 0) where no limit exists and it is
nan. A negative argument is outside the domain and gives nan. None of
these cases emits a numpy warning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Below this |s/t - 1| the logarithmic mean and its derivative switch to a
# Taylor series so both are smooth through s == t (0/0 in the closed form).
_LOG_SERIES_SWITCH = 1e-3


def _densities(s, t):
    """s and t as broadcast float arrays, nan wherever either is negative."""
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    outside = (s < 0) | (t < 0)
    return np.where(outside, np.nan, s), np.where(outside, np.nan, t)


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
        s, t = _densities(s, t)
        return np.sqrt(s * t)

    def partial_s(self, s, t):
        s, t = _densities(s, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(t / s) / 2


class ArithmeticMean(AdmissibleMean):
    """theta(s, t) = (s + t) / 2. Does not vanish at an empty node."""

    def __call__(self, s, t):
        s, t = _densities(s, t)
        return (s + t) / 2

    def partial_s(self, s, t):
        s, t = _densities(s, t)
        return np.where(np.isnan(s + t), np.nan, 0.5)


class HarmonicMean(AdmissibleMean):
    """theta(s, t) = 2 s t / (s + t). The smallest of the four; bounded partial_s near zero."""

    def __call__(self, s, t):
        s, t = _densities(s, t)
        with np.errstate(invalid="ignore"):
            value = 2 * s * t / (s + t)
        return np.where(s + t == 0, 0.0, value)

    def partial_s(self, s, t):
        s, t = _densities(s, t)
        with np.errstate(invalid="ignore"):
            return 2 * t**2 / (s + t) ** 2


class LogarithmicMean(AdmissibleMean):
    """theta(s, t) = (s - t) / (ln s - ln t). The mean for which the heat flow
    is the gradient flow of the relative entropy (Maas 2011). Not finitely
    conic-representable; the SOCP uses QuadLogMean instead."""

    def __call__(self, s, t):
        s, t = _densities(s, t)
        # theta(s, t) = hi * f(lo / hi): symmetric, and the ratio stays in [0, 1]
        # so it cannot overflow however small one argument is.
        lo, hi = np.minimum(s, t), np.maximum(s, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = hi * self._f(lo / hi)
        return np.where(hi == 0, 0.0, value)

    def partial_s(self, s, t):
        s, t = _densities(s, t)
        # theta = t f(s/t) = s f(t/s), so d theta / d s is f'(s/t) or, with
        # y = t/s, f(y) - y f'(y); use whichever keeps the ratio in [0, 1].
        # np.where evaluates both ratios; the discarded one may overflow.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            x = np.where(s <= t, s / t, t / s)
            value = np.where(s <= t, self._f_prime(x), self._f_minus_x_f_prime(x))
        value = np.where((s == 0) & (t > 0), np.inf, value)
        return np.where((s > 0) & (t == 0), 0.0, value)

    # f(x) = (x - 1) / ln x on [0, 1], with d = x - 1 in the series.

    @staticmethod
    def _log(x):
        # log1p(x - 1) is exact near x == 1 but loses x entirely below ~1e-16
        return np.where(x < 0.5, np.log(x), np.log1p(x - 1))

    @classmethod
    def _f(cls, x):
        d = x - 1
        # 1 + d/2 - d^2/12 + d^3/24 - 19 d^4/720 + 3 d^5/160 + O(d^6)
        series = 1 + d * (1 / 2 + d * (-1 / 12 + d * (1 / 24 + d * (-19 / 720 + d * 3 / 160))))
        return np.where(np.abs(d) < _LOG_SERIES_SWITCH, series, d / cls._log(x))

    @classmethod
    def _f_prime(cls, x):
        d = x - 1
        # 1/2 - d/6 + d^2/8 - 19 d^3/180 + 3 d^4/32 + O(d^5)
        series = 1 / 2 + d * (-1 / 6 + d * (1 / 8 + d * (-19 / 180 + d * 3 / 32)))
        L = cls._log(x)
        return np.where(np.abs(d) < _LOG_SERIES_SWITCH, series, (L - d / x) / L**2)

    @classmethod
    def _f_minus_x_f_prime(cls, x):
        d = x - 1
        # 1/2 + d/6 - d^2/24 + d^3/45 - 7 d^4/480 + O(d^5)
        series = 1 / 2 + d * (1 / 6 + d * (-1 / 24 + d * (1 / 45 + d * -7 / 480)))
        L = cls._log(x)
        return np.where(np.abs(d) < _LOG_SERIES_SWITCH, series, (d - L) / L**2)


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
        s, t = (x[..., np.newaxis] for x in _densities(s, t))
        return (self.w * s**self.alpha * t ** (1 - self.alpha)).sum(axis=-1)

    def partial_s(self, s, t):
        s, t = (x[..., np.newaxis] for x in _densities(s, t))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.w * self.alpha * s ** (self.alpha - 1) * t ** (1 - self.alpha)).sum(axis=-1)

    def __repr__(self) -> str:
        return f"QuadLogMean({self.K})"

    def __eq__(self, other) -> bool:
        return isinstance(other, QuadLogMean) and other.K == self.K

    def __hash__(self) -> int:
        return hash((QuadLogMean, self.K))
