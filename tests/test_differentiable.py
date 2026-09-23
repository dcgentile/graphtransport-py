"""geodesic and transport_cost on torch tensors: exact gradients through the
shooting solve (implicit function theorem), and the policies for methods and
fallbacks that are not differentiable."""

import numpy as np
import pytest
import torch

from graphtransport import (
    LogarithmicMean,
    MarkovGraph,
    ShootingFallbackWarning,
    analysis,
    barycenter,
    geodesic,
    grid_markov_chain,
    ground_cost,
    transport_cost,
)

TIGHT = dict(tol=1e-13)  # central differences of the returned values need Newton well below their step


def _pair(G, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.5, 1.5, G.n), rng.uniform(0.5, 1.5, G.n)
    return a / (a @ G.pi), b / (b @ G.pi)


def _directional_check(G, f, x0, grad, seed, h=1e-5, rtol=1e-7):
    # Random directions, mass-changing ones included. The API rejects a density
    # whose mass is off by more than 1e-6 -- the guard against passing a
    # probability vector for a pi-density -- so the perturbed inputs are
    # renormalized; the solver normalizes inside the graph, so the gradient at
    # mass 1 is the gradient of that composition.
    pi = torch.tensor(G.pi)
    rng = np.random.default_rng(seed)
    for _ in range(3):
        v = torch.tensor(rng.standard_normal(x0.shape[0]))
        with torch.no_grad():
            up, down = x0 + h * v, x0 - h * v
            fd = (f(up / (up @ pi)) - f(down / (down @ pi))).item() / (2 * h)
        assert (grad @ v).item() == pytest.approx(fd, rel=rtol, abs=1e-9)


@pytest.fixture(scope="module")
def grid4():
    return MarkovGraph(*grid_markov_chain(4))


def test_values_match_the_numpy_api(grid4):
    a, b = _pair(grid4, 0)
    W = transport_cost(grid4, torch.tensor(a), torch.tensor(b))
    assert isinstance(W, torch.Tensor) and W.dtype == torch.float64
    assert W.item() == pytest.approx(transport_cost(grid4, a, b), rel=1e-10)
    sol = geodesic(grid4, torch.tensor(a), torch.tensor(b))
    ref = geodesic(grid4, a, b)
    for field in ("rho", "m", "phi0", "phi1"):
        np.testing.assert_allclose(getattr(sol, field).detach().numpy(), getattr(ref, field), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("which", ["rhoA", "rhoB"])
def test_the_W2_gradient_matches_central_differences(grid4, which):
    a, b = (torch.tensor(v) for v in _pair(grid4, 1))
    x = (a if which == "rhoA" else b).clone().requires_grad_(True)
    args = (x, b) if which == "rhoA" else (a, x)
    geodesic(grid4, *args, **TIGHT).W2.backward()

    def f(y):
        return geodesic(grid4, *((y, b) if which == "rhoA" else (a, y)), **TIGHT).W2

    _directional_check(grid4, f, x.detach(), x.grad, seed=2)


def test_path_momentum_and_potential_gradients_match_central_differences(grid4):
    # a random linear functional of everything the geodesic returns
    a, b = (torch.tensor(v) for v in _pair(grid4, 3))
    rng = np.random.default_rng(4)
    w_rho = torch.tensor(rng.standard_normal((grid4.n, 151)))
    w_m = torch.tensor(rng.standard_normal((grid4.E.shape[0], 150)))
    w_phi = torch.tensor(rng.standard_normal(grid4.n))

    def f(x):
        s = geodesic(grid4, x, b, **TIGHT)
        return (s.rho * w_rho).sum() + 0.1 * (s.m * w_m).sum() + (s.phi1 * w_phi).sum() + (s.phi0 * w_phi).sum()

    x = a.clone().requires_grad_(True)
    f(x).backward()
    _directional_check(grid4, f, a, x.grad, seed=5, rtol=1e-6)


def test_the_logarithmic_mean_differentiates_too():
    # exercises the generic (autodiff + Euler) second derivative
    G = MarkovGraph(*grid_markov_chain(3), mean=LogarithmicMean())
    a, b = (torch.tensor(v) for v in _pair(G, 6))
    x = a.clone().requires_grad_(True)
    transport_cost(G, x, b, **TIGHT).backward()
    _directional_check(G, lambda y: transport_cost(G, y, b, **TIGHT), a, x.grad, seed=7)


def test_rescaling_a_density_has_zero_derivative(grid4):
    a, b = (torch.tensor(v) for v in _pair(grid4, 8))
    x = a.clone().requires_grad_(True)
    transport_cost(grid4, x, b).backward()
    assert (x.grad @ a).item() == pytest.approx(0.0, abs=1e-12)


def test_float32_input_is_computed_in_float64(grid4):
    a, b = _pair(grid4, 9)
    W = transport_cost(grid4, torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))
    assert W.dtype == torch.float64


def test_a_non_cpu_tensor_is_rejected(grid4):
    a, b = _pair(grid4, 10)
    with pytest.raises(ValueError, match="is on meta.*CPU"):
        transport_cost(grid4, torch.tensor(a, device="meta"), torch.tensor(b))


def test_transport_cost_has_zero_gradient_where_the_endpoints_coincide(grid4):
    # sqrt's derivative at 0 is infinite, and inf * 0 would give nan; zero is
    # the subgradient at W's minimum, as torch.linalg.norm uses
    a, _ = _pair(grid4, 17)
    x = torch.tensor(a, requires_grad=True)
    W = transport_cost(grid4, x, torch.tensor(a))
    W.backward()
    assert W.item() == 0.0
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_second_order_gradients_raise_rather_than_return_wrong_values(grid4):
    a, b = _pair(grid4, 18)
    x = torch.tensor(a, requires_grad=True)
    (g,) = torch.autograd.grad(transport_cost(grid4, x, torch.tensor(b)), x, create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        g.sum().backward()


# ----- policies -----


def test_other_methods_refuse_inputs_that_require_grad(grid4):
    a, b = _pair(grid4, 11)
    with pytest.raises(TypeError, match="method='shooting' only.*'sinkhorn' is not differentiable"):
        transport_cost(
            grid4,
            torch.tensor(a, requires_grad=True),
            torch.tensor(b),
            method="sinkhorn",
            cost=ground_cost(grid4),
            epsilon=0.1,
        )


def test_other_methods_run_on_tensors_without_grad(grid4):
    a, b = _pair(grid4, 12)
    kw = dict(method="sinkhorn", cost=ground_cost(grid4), epsilon=0.1)
    W = transport_cost(grid4, torch.tensor(a), torch.tensor(b), **kw)
    assert isinstance(W, torch.Tensor) and W.item() == pytest.approx(transport_cost(grid4, a, b, **kw))


def _with_zeros(G, seed):
    a, b = _pair(G, seed)
    a[:4] = 0.0
    return a / (a @ G.pi), b


def test_boundary_data_with_grad_raises_instead_of_falling_back(grid4):
    a, b = _with_zeros(grid4, 13)
    with pytest.raises(ValueError, match="No fallback to method='socp': the inputs require grad"):
        geodesic(grid4, torch.tensor(a, requires_grad=True), torch.tensor(b))


def test_boundary_data_without_grad_falls_back_and_returns_tensors(grid4):
    pytest.importorskip("cvxpy")
    a, b = _with_zeros(grid4, 14)
    with pytest.warns(ShootingFallbackWarning):
        sol = geodesic(grid4, torch.tensor(a), torch.tensor(b))
    assert isinstance(sol.W2, torch.Tensor)
    assert sol.W2.item() == pytest.approx(geodesic(grid4, a, b, method="socp").W2)


def test_barycenter_and_analysis_do_not_take_tensors_yet(grid4):
    a, b = _pair(grid4, 15)
    with pytest.raises(TypeError, match="barycenter does not take torch tensors yet"):
        barycenter(grid4, [torch.tensor(a), b], [0.5, 0.5])
    with pytest.raises(TypeError, match="analysis does not take torch tensors yet"):
        analysis(grid4, torch.tensor(a), [a, b])


def test_a_numpy_only_mean_refuses_gradients_but_runs_without_them():
    # its torch versions evaluate numpy, so no gradient flows through theta
    from graphtransport import AdmissibleMean

    class NumpyOnly(AdmissibleMean):
        def __call__(self, s, t):
            return ((np.sqrt(s) + np.sqrt(t)) / 2) ** 2

        def partial_s(self, s, t):
            return (np.sqrt(s) + np.sqrt(t)) / (2 * np.sqrt(s))

    G = MarkovGraph(*grid_markov_chain(3), mean=NumpyOnly())
    a, b = _pair(G, 16)
    with pytest.raises(TypeError, match="gradients need a mean with torch versions"):
        transport_cost(G, torch.tensor(a, requires_grad=True), torch.tensor(b))
    W = transport_cost(G, torch.tensor(a), torch.tensor(b))
    assert W.item() == pytest.approx(transport_cost(G, a, b), rel=1e-10)
