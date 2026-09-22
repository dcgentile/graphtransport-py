"""method="shooting" in the unified API, and its standing as the default."""

import inspect
import warnings

import numpy as np
import pytest

from graphtransport import (
    GeodesicSolution,
    LogarithmicMean,
    MarkovGraph,
    analysis,
    barycenter,
    geodesic,
    grid_markov_chain,
    transport_cost,
    triangle_markov_chain,
)
from graphtransport.graph import graph_gradient
from graphtransport.shooting import ShootingError


def _density(G, rng, offset=0.5):
    v = rng.random(G.n) + offset
    return v / (v @ G.pi)


@pytest.fixture(scope="module")
def grid4():
    G = MarkovGraph(*grid_markov_chain(4))
    rng = np.random.default_rng(9)
    return G, _density(G, rng), _density(G, rng), _density(G, rng)


def test_shooting_is_the_default_everywhere():
    for f in (geodesic, transport_cost, barycenter, analysis):
        assert inspect.signature(f).parameters["method"].default == "shooting"


def test_geodesic_solution_shape_and_endpoints(grid4):
    G, A, B, _ = grid4
    sol = geodesic(G, A, B, nsteps=100)
    assert isinstance(sol, GeodesicSolution) and sol.status == "converged"
    assert sol.rho.shape == (G.n, 101) and sol.m.shape == (len(G.E), 100)
    assert np.array_equal(sol.m0, sol.m[:, 0])
    np.testing.assert_allclose(sol.rho[:, 0], A, atol=1e-12)
    np.testing.assert_allclose(sol.rho[:, -1], B, atol=1e-6)
    np.testing.assert_allclose(sol.rho.T @ G.pi, 1.0, atol=1e-10)


def test_geodesic_agrees_with_the_socp(grid4):
    # Julia's gates: W2 within 2e-2 at N=40 (measured 5e-7), m0 within 0.1
    # (measured 4e-3), and endpoint potentials within 5% of a fine SOCP --
    # shooting reports them in the SOCP's W2-gradient convention.
    pytest.importorskip("cvxpy")
    G, A, B, _ = grid4
    sh = geodesic(G, A, B)
    socp = geodesic(G, A, B, method="socp", N=40)
    assert abs(socp.W2 - sh.W2) < 2e-2 * sh.W2
    assert np.linalg.norm(socp.m0 - sh.m0) / np.linalg.norm(sh.m0) < 0.1
    fine = geodesic(G, A, B, method="socp", N=160)
    for k in ("phi0", "phi1"):
        np.testing.assert_allclose(
            graph_gradient(G, getattr(sh, k)), graph_gradient(G, getattr(fine, k)), rtol=0.05, err_msg=k
        )


def test_transport_cost_is_sqrt_w2_without_the_path(grid4):
    G, A, B, _ = grid4
    assert transport_cost(G, A, B) == pytest.approx(np.sqrt(geodesic(G, A, B).W2), rel=1e-12)


def test_barycenter_agrees_with_the_certified_socp_optimum(grid4):
    pytest.importorskip("cvxpy")
    G, A, B, C = grid4
    refs, lam = [A, B, C], np.array([0.5, 0.3, 0.2])
    nu_sh, J_sh, info = barycenter(G, refs, lam, tol=1e-6)
    assert info["status"] == "converged" and info["grad_hist"][-1] < 1e-6
    assert np.all(np.diff(info["J_hist"]) <= 0)  # steps accepted only on a decrease
    nu_fine, J_fine, _ = barycenter(G, refs, lam, method="socp", N=80)
    assert np.linalg.norm((nu_sh - nu_fine) * np.sqrt(G.pi)) < 5e-3
    assert abs(J_sh - J_fine) / J_fine < 5e-3
    # certificate: evaluated by one solver, shooting's point is no better than the SOCP's optimum
    nu_socp, _, _ = barycenter(G, refs, lam, method="socp", N=10)

    def J_at(nu):
        return sum(lam[i] * geodesic(G, refs[i], nu, method="socp", N=10).W2 for i in range(3))

    assert J_at(nu_sh) >= J_at(nu_socp) - 1e-6
    # exact in its own convention
    np.testing.assert_allclose(analysis(G, nu_sh, refs), lam, atol=1e-3)


def test_analysis_recovers_socp_synthesised_weights_to_discretisation_error(grid4):
    pytest.importorskip("cvxpy")
    G, A, B, C = grid4
    refs, lam = [A, B, C], np.array([0.5, 0.3, 0.2])
    nu, _, _ = barycenter(G, refs, lam, method="socp", N=10)
    np.testing.assert_allclose(analysis(G, nu, refs), lam, atol=5e-2)


def test_barycenter_on_a_cycle_where_the_tangent_kind_cannot_be_inferred():
    # On the triangle n == |E|, so exp_map cannot tell a potential from a
    # momentum by length. The descent passes kind explicitly; Julia's infers
    # it and would raise on any cycle graph.
    G = MarkovGraph(*triangle_markov_chain())
    refs = [np.array([2.0, 0.5, 0.5]), np.array([0.5, 2.0, 0.5]), np.array([0.5, 0.5, 2.0])]
    nu, _, info = barycenter(G, refs, [0.5, 0.3, 0.2])
    assert info["status"] == "converged"
    assert nu @ G.pi == pytest.approx(1.0)


def test_shooting_takes_the_exact_logarithmic_mean(grid4):
    # the SOCP can only approximate it with QuadLogMean
    G, A, B, _ = grid4
    GL = G.with_mean(LogarithmicMean())
    assert transport_cost(GL, A, B) > 0
    pytest.importorskip("cvxpy")
    with pytest.raises(ValueError, match="no conic representation"):
        transport_cost(GL, A, B, method="socp")


# ----- data shooting cannot take -----


def _with_zeros(G, rho, k=2):
    rho = rho.copy()
    rho[:k] = 0.0
    return rho / (rho @ G.pi)


@pytest.mark.parametrize(
    "call, name",
    [
        (lambda G, A, B, Z: geodesic(G, A, Z), "rhoB"),
        (lambda G, A, B, Z: transport_cost(G, Z, A), "rhoA"),
        (lambda G, A, B, Z: barycenter(G, [A, Z], [0.5, 0.5]), r"refs\[1\]"),
        (lambda G, A, B, Z: analysis(G, Z, [A, B]), "target"),
        (lambda G, A, B, Z: analysis(G, A, [Z, B]), r"refs\[0\]"),
    ],
)
def test_boundary_data_is_an_error_naming_the_argument_and_the_alternative(grid4, call, name):
    G, A, B, _ = grid4
    Z = _with_zeros(G, A)
    with pytest.raises(ValueError, match=f"{name} is not strictly positive.*method='socp'"):
        call(G, A, B, Z)


def test_the_socp_takes_the_same_boundary_data(grid4):
    pytest.importorskip("cvxpy")
    G, A, _, _ = grid4
    assert geodesic(G, A, _with_zeros(G, A), method="socp", N=10).W2 > 0


def test_a_zero_weight_reference_may_touch_the_boundary(grid4):
    # it is never log-mapped
    G, A, B, _ = grid4
    _, _, info = barycenter(G, [A, B, _with_zeros(G, A)], [0.5, 0.5, 0.0])
    assert info["status"] == "converged"


def test_near_boundary_failure_names_the_input_that_made_it_stiff():
    # Passes the positivity check but defeats shooting: the internal error is
    # re-raised with the argument and its smallest entry.
    G = MarkovGraph(*grid_markov_chain(5))
    rng = np.random.default_rng(1)
    A, B = _density(G, rng), _density(G, rng)
    B[:5] = 1e-5
    B /= B @ G.pi
    with pytest.raises(ShootingError, match=r"smallest density involved is 1\.\de-05, in rhoB.*method='socp'"):
        transport_cost(G, A, B)


def test_a_mass_error_the_api_admits_is_absorbed(grid4):
    # _check_density admits 1e-6; log_map on its own would demand 1e-8
    G, A, B, _ = grid4
    assert transport_cost(G, A * (1 + 5e-7), B) == pytest.approx(transport_cost(G, A, B), rel=1e-6)


# ----- keywords -----


@pytest.mark.parametrize(
    "kwargs, method, match",
    [
        ({"N": 20}, "shooting", r"does not take N .*nsteps="),
        ({"cost": 1, "epsilon": 0.1}, "shooting", "cost .*method='sinkhorn'"),
        ({"solver": "SCS"}, "shooting", "solver .*method='socp'"),
        ({"nsteps": 10}, "socp", "nsteps .*method='shooting'"),
        ({"tol": 1e-8}, "socp", "tol .*method='shooting' and method='sinkhorn'"),
        ({"maxiters": 5}, "sinkhorn", "maxiters .*method='shooting'"),
    ],
)
def test_a_keyword_owned_by_another_method_is_rejected_by_name(grid4, kwargs, method, match):
    G, A, B, _ = grid4
    with pytest.raises(TypeError, match=match):
        geodesic(G, A, B, method=method, **kwargs)


def test_a_keyword_two_methods_share_passes_for_both(grid4):
    G, A, B, _ = grid4
    assert transport_cost(G, A, B, tol=1e-10) == pytest.approx(transport_cost(G, A, B), rel=1e-8)


def test_maxiters_warns_and_reports_it(grid4):
    G, A, B, C = grid4
    with pytest.warns(UserWarning, match="reached maxiters=1"):
        _, _, info = barycenter(G, [A, B, C], [0.5, 0.3, 0.2], maxiters=1, tol=1e-14)
    assert info["status"] == "maxiters"


@pytest.mark.parametrize(
    "kwargs, match",
    [({"h": 0.0}, "h must"), ({"tol": -1.0}, "tol must"), ({"maxiters": 0}, "maxiters"), ({"log_tol": np.nan}, "log_tol")],
)
def test_barycenter_rejects_bad_descent_parameters(grid4, kwargs, match):
    G, A, B, _ = grid4
    with pytest.raises(ValueError, match=match):
        barycenter(G, [A, B], [0.5, 0.5], **kwargs)


def test_stalled_is_a_status_not_an_error(grid4):
    # An unreachable tol with a coarse ftol: the descent stops once a step no
    # longer decreases J by ftol relative, and reports that rather than raising.
    G, A, B, C = grid4
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # stalled must not warn; only maxiters does
        _, _, info = barycenter(G, [A, B, C], [0.5, 0.3, 0.2], tol=1e-15, ftol=1e-6)
    assert info["status"] == "stalled"
    # the accepted final point is recorded, so the histories have one more
    # entry than the iterations and stay the same length
    assert len(info["J_hist"]) == len(info["grad_hist"]) == info["iters"] + 1
    assert info["grad_hist"][-1] < info["grad_hist"][0]
