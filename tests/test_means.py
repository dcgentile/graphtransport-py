import numpy as np
import pytest

from graphtransport.means import geomean, logmean, logmean_partial_s, logmean_partial_t


def test_geomean_basic():
    assert geomean(4.0, 9.0) == pytest.approx(6.0)


def test_geomean_negative_is_neg_inf():
    assert geomean(-1.0, 4.0) == -np.inf
    assert geomean(4.0, -1.0) == -np.inf


def test_geomean_vectorized():
    x = np.array([4.0, -1.0, 9.0])
    y = np.array([9.0, 4.0, 4.0])
    result = geomean(x, y)
    np.testing.assert_allclose(result, [6.0, -np.inf, 6.0])


def test_logmean_basic():
    # L(1, e) = (1 - e) / (log 1 - log e) = (1 - e) / -1 = e - 1
    assert logmean(1.0, np.e) == pytest.approx(np.e - 1.0)


def test_logmean_negative_is_neg_inf():
    assert logmean(-1.0, 2.0) == -np.inf


def test_logmean_near_diagonal_is_arithmetic_mean():
    assert logmean(2.0, 2.0 + 1e-8) == pytest.approx(2.0 + 5e-9)


def test_logmean_partial_s_matches_finite_difference():
    s, t = 3.0, 1.0
    h = 1e-6
    fd = (logmean(s + h, t) - logmean(s - h, t)) / (2 * h)
    assert logmean_partial_s(s, t) == pytest.approx(fd, rel=1e-4)


def test_logmean_partial_t_matches_finite_difference():
    s, t = 3.0, 1.0
    h = 1e-6
    fd = (logmean(s, t + h) - logmean(s, t - h)) / (2 * h)
    assert logmean_partial_t(s, t) == pytest.approx(fd, rel=1e-4)


def test_logmean_partials_at_diagonal_are_one_half():
    assert logmean_partial_s(2.0, 2.0) == pytest.approx(0.5)
    assert logmean_partial_t(2.0, 2.0) == pytest.approx(0.5)
