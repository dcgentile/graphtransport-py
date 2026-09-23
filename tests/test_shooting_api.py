"""method="shooting" in the unified API, and its standing as the default."""

import inspect
import warnings

import numpy as np
import pytest

from graphtransport import (
    GeodesicSolution,
    LogarithmicMean,
    MarkovGraph,
    ShootingFallbackWarning,
    analysis,
    barycenter,
    geodesic,
    grid_markov_chain,
    transport_cost,
    triangle_markov_chain,
)
from graphtransport.graph import graph_gradient
from graphtransport.shooting import ShootingError, analyze_shooting, barycenter_shooting


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


BOUNDARY_CALLS = [
    (lambda G, A, B, Z, **kw: geodesic(G, A, Z, **kw), "rhoB"),
    (lambda G, A, B, Z, **kw: transport_cost(G, Z, A, **kw), "rhoA"),
    (lambda G, A, B, Z, **kw: barycenter(G, [A, Z], [0.5, 0.5], **kw), r"refs\[1\]"),
    (lambda G, A, B, Z, **kw: analysis(G, Z, [A, B], **kw), "target"),
    (lambda G, A, B, Z, **kw: analysis(G, A, [Z, B], **kw), r"refs\[0\]"),
]
BOUNDARY_IDS = ["geodesic", "transport_cost", "barycenter", "analysis-target", "analysis-ref"]


@pytest.mark.parametrize("call, name", BOUNDARY_CALLS, ids=BOUNDARY_IDS)
def test_boundary_data_falls_back_to_the_socp_with_a_warning(grid4, call, name):
    pytest.importorskip("cvxpy")
    G, A, B, _ = grid4
    Z = _with_zeros(G, A)
    with pytest.warns(ShootingFallbackWarning, match=f"{name} is not strictly positive.*Falling back to method='socp'"):
        result = call(G, A, B, Z)
    assert result is not None


@pytest.mark.parametrize("call, name", BOUNDARY_CALLS, ids=BOUNDARY_IDS)
def test_fallback_false_raises_naming_the_argument_and_the_alternative(grid4, call, name):
    G, A, B, _ = grid4
    Z = _with_zeros(G, A)
    with pytest.raises(ValueError, match=f"{name} is not strictly positive.*method='socp'"):
        call(G, A, B, Z, fallback=False)


def test_the_fallback_gives_the_socp_answer(grid4):
    pytest.importorskip("cvxpy")
    G, A, _, _ = grid4
    Z = _with_zeros(G, A)
    with pytest.warns(ShootingFallbackWarning):
        sol = geodesic(G, A, Z)
    direct = geodesic(G, A, Z, method="socp")
    assert sol.W2 == pytest.approx(direct.W2) and sol.status == direct.status == "optimal"
    assert sol.rho.shape == direct.rho.shape  # the SOCP's N=10 path, not shooting's


def test_a_barycenter_says_which_method_produced_it(grid4):
    pytest.importorskip("cvxpy")
    G, A, B, _ = grid4
    assert barycenter(G, [A, B], [0.5, 0.5])[2]["method"] == "shooting"
    with pytest.warns(ShootingFallbackWarning):
        _, _, info = barycenter(G, [A, _with_zeros(G, B)], [0.5, 0.5])
    assert info["method"] == "socp" and "geodesics" in info
    assert barycenter(G, [A, B], [0.5, 0.5], method="socp")[2]["method"] == "socp"


def test_the_fallback_forwards_the_analysis_options_both_methods_share(grid4):
    pytest.importorskip("cvxpy")
    G, A, B, C = grid4
    with pytest.warns(ShootingFallbackWarning):
        lam_hat, A_gram = analysis(G, _with_zeros(G, A), [A, B, C], return_system=True, qp_method="scipy")
    assert A_gram.shape == (3, 3) and lam_hat.sum() == pytest.approx(1.0)


def test_the_fallback_can_be_escalated_to_an_error(grid4):
    pytest.importorskip("cvxpy")
    G, A, _, _ = grid4
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShootingFallbackWarning)
        with pytest.raises(ShootingFallbackWarning):
            geodesic(G, A, _with_zeros(G, A))


def test_without_cvxpy_the_shooting_error_is_raised_with_a_note(grid4, monkeypatch):
    # an ImportError from the fallback would hide the actual problem
    import graphtransport.solvers as solvers

    monkeypatch.setattr(solvers, "cvxpy_available", lambda: False)
    G, A, _, _ = grid4
    with pytest.raises(ValueError, match=r"rhoB is not strictly positive.*fallback.*graphtransport\[socp\]"):
        geodesic(G, A, _with_zeros(G, A))


def test_the_warning_points_at_the_callers_line(grid4):
    pytest.importorskip("cvxpy")
    G, A, _, _ = grid4
    for call in (lambda: geodesic(G, A, _with_zeros(G, A)), lambda: transport_cost(G, A, _with_zeros(G, A))):
        with pytest.warns(ShootingFallbackWarning) as record:
            call()
        assert record[0].filename == __file__


def test_floor_rtol_is_honoured_by_the_entry_point_check():
    # With the default floor at the entry point and the caller's inside
    # log_map, a density between the two passed one check and failed the
    # other under log_map's internal name ("target").
    G = MarkovGraph(*grid_markov_chain(3))
    rng = np.random.default_rng(2)
    A, B = _density(G, rng), _density(G, rng)
    B[0] = 1e-5
    B /= B @ G.pi
    with pytest.raises(ValueError, match="rhoB is not strictly positive"):
        geodesic(G, A, B, floor_rtol=1e-3, fallback=False)


def test_transport_cost_takes_verbose_like_geodesic(grid4):
    G, A, B, _ = grid4
    assert transport_cost(G, A, B, verbose=False) == pytest.approx(np.sqrt(geodesic(G, A, B, verbose=False).W2))


def test_the_socp_takes_the_same_boundary_data(grid4):
    pytest.importorskip("cvxpy")
    G, A, _, _ = grid4
    assert geodesic(G, A, _with_zeros(G, A), method="socp", N=10).W2 > 0


def test_a_zero_weight_reference_may_touch_the_boundary(grid4):
    # it is never log-mapped, so no fallback either
    G, A, B, _ = grid4
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShootingFallbackWarning)
        _, _, info = barycenter(G, [A, B, _with_zeros(G, A)], [0.5, 0.5, 0.0])
    assert info["status"] == "converged" and info["method"] == "shooting"


def _near_boundary_pair():
    # passes the positivity check, but a row at 1e-5 defeats shooting on a 5x5 grid
    G = MarkovGraph(*grid_markov_chain(5))
    rng = np.random.default_rng(1)
    A, B = _density(G, rng), _density(G, rng)
    B[:5] = 1e-5
    B /= B @ G.pi
    return G, A, B


def test_near_boundary_failure_names_the_input_that_made_it_stiff():
    G, A, B = _near_boundary_pair()
    with pytest.raises(ShootingError, match=r"smallest input density is 1\.\de-05, in rhoB\..*method='socp'"):
        transport_cost(G, A, B, fallback=False)


def test_near_boundary_failure_falls_back_too():
    pytest.importorskip("cvxpy")
    G, A, B = _near_boundary_pair()
    match = r"did not converge \(.*; smallest input density 1\.\de-05, in rhoB\)"
    with pytest.warns(ShootingFallbackWarning, match=match):
        w = transport_cost(G, A, B)
    assert w == pytest.approx(transport_cost(G, A, B, method="socp"))


def _long_transport_pair():
    # corner to corner on a 10x10 grid, smallest density 0.37; the SOCP's path
    # between them thins to 0.05
    k = 10
    G = MarkovGraph(*grid_markov_chain(k))
    xy = np.array([(i % k, i // k) for i in range(G.n)], dtype=float)

    def bump(center):
        rho = np.exp(-((xy - center) ** 2).sum(axis=1) / 8) + 0.05
        return rho / (rho @ G.pi)

    return G, bump((0, 0)), bump((k - 1, k - 1))


def test_a_long_transport_converges_without_falling_back():
    # The forward-difference Jacobian stalled here (line search failed at
    # iteration 6) and the default method fell back to the SOCP; with the
    # exact Jacobian shooting converges, as Julia does, in 19 iterations.
    G, A, B = _long_transport_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShootingFallbackWarning)
        W = transport_cost(G, A, B)
    assert W**2 == pytest.approx(138.60667081914244, rel=1e-10)  # Julia's log_map W2


def test_running_out_of_iterations_names_both_causes_not_the_boundary():
    G, A, B = _long_transport_pair()
    with pytest.raises(ShootingError, match=r"did not converge in 2 iterations.*smallest input density is "
                                            r"3\.7e-01, in rhoA\. .*maxiters=") as info:  # fmt: skip
        transport_cost(G, A, B, fallback=False, maxiters=2)
    assert "near the boundary" not in str(info.value)


def test_running_out_of_iterations_falls_back_with_the_real_error():
    pytest.importorskip("cvxpy")
    G, A, B = _long_transport_pair()
    match = (
        r"did not converge \(log_map: Newton did not converge in 2 iterations \(residual [^)]+\); "
        r"smallest input density 3\.7e-01, in rhoA\)"
    )
    with pytest.warns(ShootingFallbackWarning, match=match) as record:
        w = transport_cost(G, A, B, maxiters=2)
    assert w == pytest.approx(transport_cost(G, A, B, method="socp"))
    assert "raising maxiters= (default 50) may let shooting solve it exactly" in str(record[0].message)


# ----- each entry point's Newton budget -----


def test_analysis_takes_the_log_maps_newton_budget(grid4):
    # the error used to advise maxiters=, which analysis did not take
    G, A, B, C = grid4
    with pytest.raises(ShootingError, match=r"Newton did not converge in 0 iterations.*\(maxiters=, default 50\)"):
        analysis(G, A, [B, C], maxiters=0, fallback=False)


def test_barycenter_takes_the_log_maps_newton_budget_as_log_maxiters(grid4):
    # barycenter's maxiters is the descent's; the log maps' budget is log_maxiters,
    # and the unreachable-reference error now carries the log map's own failure
    G, A, B, _ = grid4
    with pytest.raises(ShootingError, match=r"not reachable.*The log map failed with: log_map: Newton did not "
                                            r"converge in 0 iterations.*\(log_maxiters=, default 50\)"):  # fmt: skip
        barycenter(G, [A, B], [0.5, 0.5], log_maxiters=0, fallback=False)


def test_the_fallback_warning_names_each_entry_points_budget(grid4):
    pytest.importorskip("cvxpy")
    G, A, B, _ = grid4
    with pytest.warns(ShootingFallbackWarning, match=r"raising log_maxiters= \(default 50\)"):
        barycenter(G, [A, B], [0.5, 0.5], log_maxiters=0)


@pytest.mark.parametrize(
    "call, match",
    [
        (
            lambda G, A, B: analysis(G, A, [A, B], log_maxiters=5),
            r"analysis\(method='shooting'\) does not take "
            r"log_maxiters; it takes .*maxiters",
        ),
        (lambda G, A, B: barycenter(G, [A, B], [0.5, 0.5], phi0_init=A), r"barycenter.*does not take phi0_init"),
        (lambda G, A, B: geodesic(G, A, B, h=1.0), r"geodesic.*does not take h; it takes .*phi0_init"),
        (lambda G, A, B: transport_cost(G, A, B, init=A), r"transport_cost.*does not take init"),
    ],
    ids=["analysis", "barycenter", "geodesic", "transport_cost"],
)
def test_a_keyword_of_another_shooting_entry_point_is_rejected_by_name(grid4, call, match):
    # METHOD_KEYWORDS is per method, so these used to reach the backend as a raw
    # "got an unexpected keyword argument"
    G, A, B, _ = grid4
    with pytest.raises(TypeError, match=match):
        call(G, A, B)


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
        ({"fallback": False}, "socp", "fallback .*method='shooting'"),
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
    [
        ({"h": 0.0}, "h must"),
        ({"tol": -1.0}, "tol must"),
        ({"maxiters": 0}, "maxiters"),
        ({"log_tol": np.nan}, "log_tol"),
    ],
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


# ----- the public shooting functions, called directly (no API wrapper) -----


@pytest.mark.parametrize(
    "call, name",
    [
        (lambda G, A, B, Z: barycenter_shooting(G, [A, Z], [0.5, 0.5]), r"refs\[1\] violates the positivity floor"),
        (lambda G, A, B, Z: analyze_shooting(G, Z, [A, B]), "target violates the positivity floor"),
        (lambda G, A, B, Z: analyze_shooting(G, A, [B, Z]), r"refs\[1\] violates the positivity floor"),
        (lambda G, A, B, Z: barycenter_shooting(G, [A, 2 * B], [0.5, 0.5]), r"refs\[1\] must be a probability density"),
    ],
    ids=["barycenter-ref", "analysis-target", "analysis-ref", "barycenter-mass"],
)
def test_direct_calls_name_the_callers_argument(grid4, call, name):
    # Both used to rely on log_map's check, which names its own parameters:
    # a bad reference came back as "target" and a bad base point as "nu".
    G, A, B, _ = grid4
    with pytest.raises(ValueError, match=name):
        call(G, A, B, _with_zeros(G, A))


def test_direct_barycenter_still_exempts_a_zero_weight_reference(grid4):
    G, A, B, _ = grid4
    _, _, info = barycenter_shooting(G, [A, B, _with_zeros(G, A)], [0.5, 0.5, 0.0])
    assert info["status"] == "converged"
