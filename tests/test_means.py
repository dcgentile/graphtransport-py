import numpy as np
import pytest

from graphtransport.means import (
    AdmissibleMean,
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    QuadLogMean,
)

MEANS = [GeometricMean(), ArithmeticMean(), HarmonicMean(), LogarithmicMean(), QuadLogMean(8)]


def _pairs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 5.0, size=n), rng.uniform(0.05, 5.0, size=n)


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_admissibility(theta: AdmissibleMean):
    s, t = _pairs()
    value = theta(s, t)
    assert np.all(value > 0)
    np.testing.assert_allclose(value, theta(t, s))  # symmetric
    np.testing.assert_allclose(theta(3 * s, 3 * t), 3 * value)  # 1-homogeneous
    np.testing.assert_allclose(theta(s, s), s)  # normalised
    # concave along the diagonal direction: theta((s+s')/2, (t+t')/2) >= mean of thetas
    s2, t2 = _pairs(seed=1)
    assert np.all(theta((s + s2) / 2, (t + t2) / 2) >= (value + theta(s2, t2)) / 2 - 1e-12)


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_partials_match_finite_differences(theta: AdmissibleMean):
    s, t = _pairs(n=50)
    h = 1e-6
    fd_s = (theta(s + h, t) - theta(s - h, t)) / (2 * h)
    fd_t = (theta(s, t + h) - theta(s, t - h)) / (2 * h)
    np.testing.assert_allclose(theta.partial_s(s, t), fd_s, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(theta.partial_t(s, t), fd_t, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_scalar_inputs(theta: AdmissibleMean):
    assert float(theta(2.0, 2.0)) == pytest.approx(2.0)
    assert np.shape(theta(1.0, 2.0)) == ()


def test_known_values():
    assert GeometricMean()(4.0, 9.0) == pytest.approx(6.0)
    assert ArithmeticMean()(4.0, 9.0) == pytest.approx(6.5)
    assert HarmonicMean()(4.0, 9.0) == pytest.approx(72 / 13)
    assert LogarithmicMean()(1.0, np.e) == pytest.approx(np.e - 1.0)


def test_ordering_harmonic_geometric_logarithmic_arithmetic():
    s, t = _pairs()
    h, g, l, a = (m(s, t) for m in (HarmonicMean(), GeometricMean(), LogarithmicMean(), ArithmeticMean()))
    assert np.all(h <= g + 1e-12)
    assert np.all(g <= l + 1e-12)
    assert np.all(l <= a + 1e-12)


def test_logarithmic_series_branch_matches_closed_form_at_the_switch():
    theta = LogarithmicMean()
    t = 1.0
    # just inside the switch the implementation uses the series; compare it
    # against the closed form evaluated directly at the same point
    s = 1 + 0.999e-3
    closed_value = (s - t) / (np.log(s) - np.log(t))
    closed_partial = (np.log(s / t) - 1 + t / s) / np.log(s / t) ** 2
    assert theta(s, t) == pytest.approx(closed_value, rel=1e-12)
    assert theta.partial_s(s, t) == pytest.approx(closed_partial, rel=1e-10)
    # exactly on the diagonal: value s, derivative 1/2
    assert theta(2.0, 2.0) == pytest.approx(2.0)
    assert theta.partial_s(2.0, 2.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "K, bound",
    [(4, 9e-4), (6, 6e-7), (8, 1.2e-10), (12, 1e-14)],
)
def test_quadlog_converges_to_logarithmic_mean(K, bound):
    # Julia docstring's worst relative errors over density ratios up to 1000
    # (8.5e-4, 6e-7, 1.2e-10, 2e-15). K=4 measures 8.52e-4 on this grid, so
    # its bound is the docstring's two-significant-figure value rounded up;
    # K=12's 2e-15 is at float precision, so it's checked at 1e-14.
    ratios = np.logspace(-3, 3, 2001)
    exact = LogarithmicMean()(ratios, 1.0)
    approx = QuadLogMean(K)(ratios, 1.0)
    assert np.max(np.abs(approx - exact) / exact) <= bound


def test_quadlog_k1_is_geometric_mean():
    s, t = _pairs()
    np.testing.assert_allclose(QuadLogMean(1)(s, t), GeometricMean()(s, t))


def test_quadlog_rejects_bad_k():
    with pytest.raises(ValueError):
        QuadLogMean(0)


def test_repr_and_equality():
    assert repr(GeometricMean()) == "GeometricMean()"
    assert repr(QuadLogMean(6)) == "QuadLogMean(6)"
    assert GeometricMean() == GeometricMean()
    assert GeometricMean() != ArithmeticMean()
    assert QuadLogMean(6) == QuadLogMean(6)
    assert QuadLogMean(6) != QuadLogMean(8)
