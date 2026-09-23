import numpy as np
import pytest

from graphtransport import (
    GeodesicSolution,
    MarkovGraph,
    analysis,
    barycenter,
    geodesic,
    markov_chain_from_edge_list,
    transport_cost,
)
from graphtransport.sinkhorn import bfs_hops, ground_cost, sinkhorn_barycenter


def _grid3():
    edges = []
    for i in range(9):
        if (i + 1) % 3 != 0:
            edges.append((i, i + 1))
        if i + 3 < 9:
            edges.append((i, i + 3))
    return MarkovGraph(*markov_chain_from_edge_list(edges))


@pytest.fixture(scope="module")
def setup():
    G = _grid3()
    cost = ground_cost(G, "shortest_path")
    hops = bfs_hops(G)
    mu = np.column_stack([np.exp(-2 * hops[k]) / np.exp(-2 * hops[k]).sum() for k in (0, 2, 7)])
    refs = [mu[:, s] / G.pi for s in range(3)]  # densities w.r.t. pi
    return G, cost, mu, refs


def test_barycenter_returns_density_matching_core(setup):
    G, cost, mu, refs = setup
    lam = [0.5, 0.3, 0.2]
    nu, J, info = barycenter(G, refs, lam, method="sinkhorn", cost=cost, epsilon=0.1, iters=256)
    assert np.dot(nu, G.pi) == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(nu * G.pi, sinkhorn_barycenter(lam, mu, cost, 0.1, iters=256), rtol=1e-12)
    assert J > 0
    assert info["epsilon"] == 0.1 and info["iters"] == 256
    assert len(info["marginal_errors"]) == 3
    assert all(e < 1e-6 for e in info["marginal_errors"])


def test_barycenter_zero_weight_drops_that_reference(setup):
    G, cost, _, refs = setup
    _, _, info = barycenter(G, refs, [0.7, 0.0, 0.3], method="sinkhorn", cost=cost, epsilon=0.1, iters=64)
    assert len(info["marginal_errors"]) == 2


def test_barycenter_requires_cost_and_epsilon(setup):
    G, cost, _, refs = setup
    with pytest.raises(ValueError, match="cost="):
        barycenter(G, refs, [0.5, 0.3, 0.2], method="sinkhorn", epsilon=0.1)
    with pytest.raises(ValueError, match="epsilon="):
        barycenter(G, refs, [0.5, 0.3, 0.2], method="sinkhorn", cost=cost)
    with pytest.raises(ValueError, match="9x9"):
        barycenter(G, refs, [0.5, 0.3, 0.2], method="sinkhorn", cost=cost[:3, :3], epsilon=0.1)


def test_unknown_method_raises(setup):
    G, cost, _, refs = setup
    with pytest.raises(ValueError, match="method must be one of"):
        barycenter(G, refs, [0.5, 0.3, 0.2], method="chambolle_pock", cost=cost, epsilon=0.1)
    with pytest.raises(ValueError, match="method must be one of"):
        geodesic(G, refs[0], refs[1], method="chambolle_pock", cost=cost, epsilon=0.1)
    with pytest.raises(ValueError, match="method must be one of"):
        analysis(G, refs[0], refs, method="entropic", cost=cost, epsilon=0.1)


def test_default_method_is_shooting():
    import inspect

    for f in (geodesic, transport_cost, barycenter, analysis):
        assert inspect.signature(f).parameters["method"].default == "shooting"


def test_geodesic_shape_endpoints_and_nans(setup):
    G, cost, _, refs = setup
    sol = geodesic(G, refs[0], refs[1], method="sinkhorn", cost=cost, epsilon=0.1, N=4, iters=256)
    assert isinstance(sol, GeodesicSolution)
    assert sol.rho.shape == (9, 5)
    assert sol.m.shape == (12, 4) and np.all(np.isnan(sol.m))
    assert np.all(np.isnan(sol.phi0)) and np.all(np.isnan(sol.phi1)) and np.all(np.isnan(sol.m0))
    assert sol.status == "converged" and sol.solvetime >= 0
    np.testing.assert_allclose((sol.rho * G.pi[:, None]).sum(axis=0), 1.0, atol=1e-8)
    # end columns are the blurred endpoints: the barycenter at weights (1,0) / (0,1)
    kw = dict(method="sinkhorn", cost=cost, epsilon=0.1, iters=256)
    np.testing.assert_allclose(sol.rho[:, 0], barycenter(G, refs[:2], [1, 0], **kw)[0])
    np.testing.assert_allclose(sol.rho[:, -1], barycenter(G, refs[:2], [0, 1], **kw)[0])
    assert sol.W2 > 0


def test_transport_cost_is_sqrt_w2(setup):
    G, cost, _, refs = setup
    kw = dict(method="sinkhorn", cost=cost, epsilon=0.1, N=2, iters=64)
    sol = geodesic(G, refs[0], refs[1], **kw)
    assert transport_cost(G, refs[0], refs[1], **kw) == pytest.approx(np.sqrt(sol.W2))


def test_analysis_recovers_synthesis_weights(setup):
    G, cost, _, refs = setup
    lam = np.array([0.5, 0.3, 0.2])
    kw = dict(method="sinkhorn", cost=cost, epsilon=0.1, iters=256)
    nu, _, _ = barycenter(G, refs, lam, **kw)
    lam_hat = analysis(G, nu, refs, **kw)
    np.testing.assert_allclose(lam_hat, lam, atol=1e-5)


def test_analysis_rejects_gram_matrix_options(setup):
    G, cost, _, refs = setup
    with pytest.raises(ValueError, match="not a Gram-matrix method"):
        analysis(G, refs[0], refs, method="sinkhorn", cost=cost, epsilon=0.1, return_system=True)
    with pytest.raises(ValueError, match="not a Gram-matrix method"):
        analysis(G, refs[0], refs, method="sinkhorn", cost=cost, epsilon=0.1, compute_condition=True)


# Input validation at the entry points (method-agnostic).

KW = {"method": "sinkhorn", "epsilon": 0.1}


@pytest.mark.parametrize(
    "lam, message",
    [
        ([1, 1, 1], "sum to 1"),
        ([0, 0, 0], "sum to 1"),
        ([2, -1, 0], "nonnegative"),
        ([0.5, np.nan, 0.5], "finite"),
        ([0.5, 0.5], "one weight per reference"),
    ],
)
def test_barycenter_rejects_weights_off_the_simplex(setup, lam, message):
    G, cost, _, refs = setup
    with pytest.raises(ValueError, match=message):
        barycenter(G, refs, lam, cost=cost, **KW)


def test_densities_must_integrate_to_one_against_pi(setup):
    G, cost, mu, refs = setup
    with pytest.raises(ValueError, match="rhoA is not a probability density"):
        geodesic(G, 2 * refs[0], refs[1], cost=cost, **KW)
    with pytest.raises(ValueError, match=r"refs\[1\] is not a probability density"):
        barycenter(G, [refs[0], 2 * refs[1], refs[2]], [0.5, 0.3, 0.2], cost=cost, **KW)
    # the classic mistake: a probability vector where a density is expected
    with pytest.raises(ValueError, match="mu / G.pi"):
        analysis(G, mu[:, 0], refs, cost=cost, **KW)
    with pytest.raises(ValueError, match="negative"):
        transport_cost(G, refs[0] - 2 * refs[0].max() * np.eye(9)[0], refs[1], cost=cost, **KW)
    with pytest.raises(ValueError, match=r"shape \(9,\)"):
        geodesic(G, refs[0][:5], refs[1], cost=cost, **KW)


def test_round_off_negatives_are_clipped(setup):
    G, cost, _, refs = setup
    mu = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0.0])
    noisy = mu / G.pi
    noisy[-1] = -1e-13  # what an upstream solver leaves in place of an exact zero
    nu, _, _ = barycenter(G, [noisy, refs[1]], [0.5, 0.5], cost=cost, **KW)
    assert np.all(np.isfinite(nu))


def test_refs_must_be_a_sequence_of_densities_not_a_matrix(setup):
    G, cost, _, refs = setup
    for matrix in (np.column_stack(refs), np.vstack(refs)):
        with pytest.raises(ValueError, match="sequence of densities"):
            barycenter(G, matrix, [0.5, 0.3, 0.2], cost=cost, **KW)
        with pytest.raises(ValueError, match="sequence of densities"):
            analysis(G, refs[0], matrix, cost=cost, **KW)
    with pytest.raises(ValueError, match="at least one"):
        barycenter(G, [], [], cost=cost, **KW)


@pytest.mark.parametrize("N", [0, -2, 4.0, True])
def test_geodesic_rejects_bad_n(setup, N):
    G, cost, _, refs = setup
    with pytest.raises(ValueError, match="N must be"):
        geodesic(G, refs[0], refs[1], N=N, cost=cost, **KW)


def test_geodesic_status_reports_whether_the_plan_converged(setup):
    G, cost, _, refs = setup
    assert geodesic(G, refs[0], refs[1], N=2, cost=cost, **KW).status == "converged"
    assert geodesic(G, refs[0], refs[1], N=2, cost=cost, iters=3, **KW).status == "iteration_limit"
    assert geodesic(G, refs[0], refs[1], N=2, cost=cost, iters=3, tol=1.0, **KW).status == "converged"


def test_transport_cost_solves_one_plan_and_warns_if_unconverged(setup, monkeypatch):
    from graphtransport import api

    G, cost, _, refs = setup
    barycenters = []
    monkeypatch.setattr(api, "build_geodesic", lambda *a, **k: barycenters.append(1))
    w = transport_cost(G, refs[0], refs[1], cost=cost, N=7, **KW)  # N is accepted and ignored
    assert not barycenters
    monkeypatch.undo()
    assert w == pytest.approx(np.sqrt(geodesic(G, refs[0], refs[1], cost=cost, **KW).W2), rel=1e-14)
    with pytest.warns(UserWarning, match="marginal error"):
        transport_cost(G, refs[0], refs[1], cost=cost, iters=3, **KW)


def test_geodesic_path_matches_columnwise_barycenters(setup):
    G, cost, _, refs = setup
    sol = geodesic(G, refs[0], refs[1], N=4, cost=cost, **KW)
    for k, t in enumerate(np.linspace(0, 1, 5)):
        nu, _, _ = barycenter(G, refs[:2], [1 - t, t], cost=cost, **KW)
        np.testing.assert_allclose(sol.rho[:, k], nu, rtol=1e-12)


def test_socp_method_dispatches_to_socp_backend():
    pytest.importorskip("cvxpy")
    from graphtransport import GeodesicSolution as GS
    from graphtransport import triangle_markov_chain
    from graphtransport.socp import geodesic_socp

    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    refs = [np.array([2.0, 0.5, 0.5]), np.array([0.5, 2.0, 0.5]), np.array([0.5, 0.5, 2.0])]
    sol = geodesic(G, refs[0], refs[1], method="socp", N=4)
    assert isinstance(sol, GS)
    assert sol.W2 == pytest.approx(geodesic_socp(G, refs[0], refs[1], N=4).W2, rel=1e-8)
    assert transport_cost(G, refs[0], refs[1], method="socp", N=4) == pytest.approx(np.sqrt(sol.W2))
    lam = np.array([0.5, 0.3, 0.2])
    nu, J, info = barycenter(G, refs, lam, method="socp", N=4)
    assert len(info["geodesics"]) == 3 and J == pytest.approx(
        sum(w * g.W2 for w, g in zip(lam, info["geodesics"], strict=True)), rel=1e-8
    )
    np.testing.assert_allclose(analysis(G, nu, refs, method="socp", N=4), lam, atol=1e-3)
    lam_hat, A = analysis(G, nu, refs, method="socp", N=4, return_system=True)
    assert A.shape == (3, 3)


@pytest.mark.parametrize(
    "call",
    [
        lambda G, cost, refs: geodesic(G, refs[0], refs[1], method="socp", cost=cost, epsilon=0.1, N=4),
        lambda G, cost, refs: transport_cost(G, refs[0], refs[1], method="socp", tol=1e-8),
        lambda G, cost, refs: barycenter(G, refs, [0.5, 0.3, 0.2], method="socp", cost=cost, epsilon=0.1),
        lambda G, cost, refs: analysis(G, refs[0], refs, method="socp", iters=64),
    ],
)
def test_sinkhorn_keywords_without_the_method_say_so(setup, call):
    # Before the guard these fell through to the conic solver and surfaced as
    # "Clarabel: unrecognized solver setting 'cost'", naming everything except
    # the keyword to change.
    G, cost, _, refs = setup
    with pytest.raises(TypeError, match="method='sinkhorn'"):
        call(G, cost, refs)


def test_sinkhorn_keywords_are_fine_with_the_sinkhorn_method(setup):
    G, cost, _, refs = setup
    assert transport_cost(G, refs[0], refs[1], method="sinkhorn", cost=cost, epsilon=0.1) > 0


def test_missing_cvxpy_names_the_extra_and_the_alternative(setup, monkeypatch):
    # The SOCP modules used to `import cvxpy as cp` directly, so the install
    # hint in solvers.import_cvxpy could not fire on the default path.
    import builtins

    G, cost, _, refs = setup
    real_import = builtins.__import__

    def no_cvxpy(name, *args, **kwargs):
        if name == "cvxpy" or name.startswith("cvxpy."):
            raise ImportError("No module named 'cvxpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cvxpy)
    with pytest.raises(ImportError, match=r"graphtransport\[socp\]"):
        geodesic(G, refs[0], refs[1], method="socp", N=4)
    with pytest.raises(ImportError, match="sinkhorn"):
        barycenter(G, refs, [0.5, 0.3, 0.2], method="socp", N=4)
