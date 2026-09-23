import numpy as np
import pytest
import torch
from torch.func import jvp

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
    np.testing.assert_allclose(theta(s, s), s)  # normalized
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
    h, g, lg, a = (m(s, t) for m in (HarmonicMean(), GeometricMean(), LogarithmicMean(), ArithmeticMean()))
    assert np.all(h <= g + 1e-12)
    assert np.all(g <= lg + 1e-12)
    assert np.all(lg <= a + 1e-12)


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


VANISHING = [m for m in MEANS if not isinstance(m, ArithmeticMean)]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("theta", VANISHING, ids=repr)
def test_empty_node_boundary(theta: AdmissibleMean):
    # theta vanishes whenever either node is empty, whichever way the edge points
    for s, t in [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]:
        assert theta(s, t) == 0.0
    assert theta.partial_s(1.0, 0.0) == 0.0
    assert theta.partial_t(0.0, 1.0) == 0.0
    assert np.isnan(theta.partial_s(0.0, 0.0))  # no limit exists at the origin


@pytest.mark.filterwarnings("error")
def test_partial_s_at_an_empty_node_is_the_limit():
    for theta in [GeometricMean(), LogarithmicMean(), QuadLogMean(8)]:
        assert theta.partial_s(0.0, 1.0) == np.inf
    assert HarmonicMean().partial_s(0.0, 1.0) == 2.0
    assert ArithmeticMean()(0.0, 1.0) == 0.5


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_negative_arguments_are_nan(theta: AdmissibleMean):
    s = np.array([-1.0, -1.0, 1.0, 1.0])
    t = np.array([-1.0, 1.0, -1.0, 1.0])
    expected = np.array([True, True, True, False])
    np.testing.assert_array_equal(np.isnan(theta(s, t)), expected)
    np.testing.assert_array_equal(np.isnan(theta.partial_s(s, t)), expected)


@pytest.mark.filterwarnings("error")
def test_logarithmic_mean_survives_extreme_ratios():
    L = LogarithmicMean()
    tiny = 1e-300  # s / t overflows; partial_s(tiny, 1) ~ 1 / (tiny ln^2 tiny) is still finite
    for value in [L(1.0, tiny), L(tiny, 1.0), L.partial_s(1.0, tiny), L.partial_s(tiny, 1.0)]:
        assert np.isfinite(value)
    assert L(1.0, tiny) == L(tiny, 1.0)
    # matches the closed form (s - t) / (ln s - ln t) away from s == t
    s, t = _pairs()
    np.testing.assert_allclose(L(s, t), (s - t) / (np.log(s) - np.log(t)), rtol=1e-12)
    np.testing.assert_allclose(L.partial_s(s, t), (L(s, t) - L(s, t) ** 2 / s) / (s - t), rtol=1e-9)


# ----- torch versions, used by the shooting flow -----


def _interior_pairs():
    s, t = _pairs(400, seed=3)
    # the diagonal, the log mean's series switch, and extreme ratios
    s = np.concatenate([s, [1.0, 1.0005, 1.0015, 2.0, 1e-8, 3.0]])
    t = np.concatenate([t, [1.0, 1.0, 1.0, 2.0000001, 1.0, 1e-8]])
    return s, t


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_torch_versions_match_numpy(theta: AdmissibleMean):
    s, t = _interior_pairs()
    S, T = torch.tensor(s), torch.tensor(t)
    np.testing.assert_allclose(theta.torch_theta(S, T).numpy(), theta(s, t), rtol=1e-14)
    np.testing.assert_allclose(theta.torch_partial_s(S, T).numpy(), theta.partial_s(s, t), rtol=1e-12)


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_torch_partial_s_is_the_derivative_of_torch_theta(theta: AdmissibleMean):
    s, t = _pairs(300, seed=4)  # away from the extreme ratios, where autodiff of theta itself cancels
    S = torch.tensor(s, requires_grad=True)
    (grad,) = torch.autograd.grad(theta.torch_theta(S, torch.tensor(t)).sum(), S)
    np.testing.assert_allclose(grad.numpy(), theta.torch_partial_s(torch.tensor(s), torch.tensor(t)).numpy(),
                               rtol=1e-9)  # fmt: skip


@pytest.mark.parametrize("theta", MEANS, ids=repr)
def test_torch_second_derivatives_match_autodiff(theta: AdmissibleMean):
    # torch_partial_s_grad is closed-form for most means; check it against
    # torch's own derivative of torch_partial_s, which must also be nan-free
    # across the log mean's series switch.
    s, t = _interior_pairs()
    S, T = torch.tensor(s), torch.tensor(t)
    ones, zeros = torch.ones_like(S), torch.zeros_like(S)
    d_s = jvp(theta.torch_partial_s, (S, T), (ones, zeros))[1]
    d_t = jvp(theta.torch_partial_s, (S, T), (zeros, ones))[1]
    got_s, got_t = theta.torch_partial_s_grad(S, T)
    assert torch.isfinite(d_s).all() and torch.isfinite(d_t).all()
    np.testing.assert_allclose(got_s.numpy(), d_s.numpy(), rtol=1e-8, atol=1e-12 * float(d_s.abs().max()))
    np.testing.assert_allclose(got_t.numpy(), d_t.numpy(), rtol=1e-8, atol=1e-12 * float(d_t.abs().max()))


class _NumpyOnlyMean(AdmissibleMean):
    """((sqrt s + sqrt t) / 2)^2: a user's own mean, numpy methods only."""

    def __call__(self, s, t):
        return ((np.sqrt(s) + np.sqrt(t)) / 2) ** 2

    def partial_s(self, s, t):
        return (np.sqrt(s) + np.sqrt(t)) / (2 * np.sqrt(s))


def test_a_numpy_only_mean_gets_numpy_backed_torch_versions():
    theta = _NumpyOnlyMean()
    assert not theta.has_torch_autodiff and GeometricMean().has_torch_autodiff
    s, t = _pairs(seed=5)
    S, T = torch.tensor(s), torch.tensor(t)
    np.testing.assert_allclose(theta.torch_theta(S, T).numpy(), theta(s, t))
    np.testing.assert_allclose(theta.torch_partial_s(S, T).numpy(), theta.partial_s(s, t))
    # second derivatives by central differences: d/ds partial_s = -sqrt(t) / (4 s^1.5), then Euler for d/dt
    d_s, d_t = theta.torch_partial_s_grad(S, T)
    np.testing.assert_allclose(d_s.numpy(), -np.sqrt(t) / (4 * s**1.5), rtol=1e-8)
    np.testing.assert_allclose(d_t.numpy(), 1 / (4 * np.sqrt(s * t)), rtol=1e-8)


def test_a_numpy_only_mean_works_with_shooting():
    # a regression caught in review: it raised a TypeError when the flow moved to torch
    from graphtransport import MarkovGraph, geodesic, grid_markov_chain

    G = MarkovGraph(*grid_markov_chain(3), mean=_NumpyOnlyMean())
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.5, 1.5, G.n), rng.uniform(0.5, 1.5, G.n)
    a, b = a / (a @ G.pi), b / (b @ G.pi)
    sol = geodesic(G, a, b)
    np.testing.assert_allclose(sol.rho[:, -1], b, atol=1e-9)
    assert sol.W2 == pytest.approx(0.2873409639733358, rel=1e-8)  # main, before the torch flow
