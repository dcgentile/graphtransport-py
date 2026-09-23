import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from graphtransport import (
    MarkovGraph,
    grid_markov_chain,
    triangle_markov_chain,
    weighted_hypercube_markov_chain,
)
from graphtransport.graph import graph_gradient, metric_tensor
from graphtransport.shooting import (
    PositivityFloorError,
    ShootingError,
    analyze_shooting,
    exp_map,
    hamiltonian_flow,
    integrate_hamiltonian,
    log_map,
    log_map_mollified,
    momentum_to_potential,
    solve_weighted_laplacian,
    weighted_laplacian,
)


def _hypercube():
    # seeded: the builder draws fresh random weights on every call
    return MarkovGraph(*weighted_hypercube_markov_chain(rng=0))


def _grid5():
    return MarkovGraph(*grid_markov_chain(5))


def _density(G, rng, offset=0.5):
    v = rng.random(G.n) + offset
    return v / (v @ G.pi)


def _pi_norm(G, v):
    return float(np.linalg.norm(v * np.sqrt(G.pi)))


@pytest.fixture
def state():
    G = _hypercube()
    rng = np.random.default_rng(2)
    nu = _density(G, rng)
    phi = 0.1 * rng.standard_normal(G.n)
    phi -= phi @ G.pi
    return G, nu, phi


# ----- weighted Laplacian and exp_map -----


def test_laplacian_is_the_flow_and_kills_constants(state):
    G, nu, phi = state
    rho_dot, _ = hamiltonian_flow(G, nu, phi)
    L = weighted_laplacian(G, nu)
    np.testing.assert_allclose(L @ phi, G.pi * rho_dot, rtol=1e-12)
    np.testing.assert_allclose(L @ np.ones(G.n), 0.0, atol=1e-14)
    np.testing.assert_allclose((L - L.T).toarray(), 0.0, atol=0)


def test_momentum_to_potential_inverts_m_equals_theta_grad_phi(state):
    G, nu, phi = state
    m = metric_tensor(G, nu) * graph_gradient(G, phi)
    phi_rec = momentum_to_potential(G, nu, m)
    np.testing.assert_allclose(phi_rec, phi, rtol=1e-10)
    assert abs(phi_rec @ G.pi) < 1e-12  # gauge


def test_solve_weighted_laplacian_rejects_a_rhs_with_a_constant_component(state):
    G, nu, _ = state
    with pytest.raises(ValueError, match="orthogonal to the constants"):
        solve_weighted_laplacian(G, nu, np.ones(G.n))


def test_exp_map_agrees_with_the_flow_for_both_tangent_kinds(state):
    G, nu, phi = state
    rho_path, _ = integrate_hamiltonian(G, nu, phi, nsteps=100)
    m = metric_tensor(G, nu) * graph_gradient(G, phi)
    np.testing.assert_allclose(exp_map(G, nu, phi, nsteps=100), rho_path[:, -1], rtol=1e-12)
    np.testing.assert_allclose(exp_map(G, nu, phi + 3.0, nsteps=100), rho_path[:, -1], rtol=1e-12)  # gauge
    np.testing.assert_allclose(exp_map(G, nu, m, nsteps=100), rho_path[:, -1], rtol=1e-8)
    # same step size as the 100-step path, half the time
    np.testing.assert_allclose(exp_map(G, nu, phi, nsteps=50, t=0.5), rho_path[:, 50], rtol=1e-12)
    assert exp_map(G, nu, phi, nsteps=100) @ G.pi == pytest.approx(1.0, abs=1e-10)


def test_exp_map_refuses_to_guess_the_kind_when_n_equals_the_edge_count():
    G = MarkovGraph(*triangle_markov_chain())  # 3 nodes, 3 edges
    nu = np.array([1.2, 0.9, 0.9])
    nu /= nu @ G.pi
    phi = np.array([0.05, -0.02, -0.03])
    with pytest.raises(ValueError, match="cannot be inferred"):
        exp_map(G, nu, phi)
    expected = integrate_hamiltonian(G, nu, phi - phi @ G.pi)[0][:, -1]
    np.testing.assert_allclose(exp_map(G, nu, phi, kind="potential"), expected)


def test_exp_map_rejects_a_tangent_of_neither_length(state):
    G, nu, _ = state
    with pytest.raises(ValueError, match="expected"):
        exp_map(G, nu, np.ones(5))
    with pytest.raises(ValueError, match="kind must be"):
        exp_map(G, nu, np.ones(G.n), kind="velocity")


# ----- log_map -----


@pytest.mark.parametrize("builder", [_hypercube, _grid5], ids=["weighted_hypercube", "grid5"])
def test_log_map_round_trip_and_newton_counts(builder):
    G = builder()
    rng = np.random.default_rng(4)
    for _ in range(3):
        nu, mu = _density(G, rng), _density(G, rng)
        r = log_map(G, nu, mu)
        assert r.iters <= 8  # Julia's spec: 3-8 cold; 2-4 typical
        assert _pi_norm(G, exp_map(G, nu, r.phi0) - mu) < 1e-6
        assert _pi_norm(G, exp_map(G, nu, r.m0) - mu) < 1e-6  # through the momentum too
        assert abs(r.phi0 @ G.pi) < 1e-12  # gauge
        assert log_map(G, nu, mu, phi0_init=r.phi0).iters == 0  # warm start from the solution
        mu2 = 0.98 * mu + 0.02  # a nearby target: warm start costs no more than cold
        assert log_map(G, nu, mu2, phi0_init=r.phi0).iters <= log_map(G, nu, mu2).iters


@pytest.mark.parametrize("builder", [_hypercube, _grid5], ids=["weighted_hypercube", "grid5"])
def test_log_map_against_the_socp(builder):
    # m0 and W2 converge to the SOCP's at O(h) in its h = 1/N. The potential
    # comparison is what pins the flow's global *sign*: the conservation laws
    # hold equally for the time-reversed flow, and the two-node closed form
    # takes absolute values.
    pytest.importorskip("cvxpy")
    from graphtransport.socp import geodesic_socp

    G = builder()
    rng = np.random.default_rng(4)
    nu, mu = _density(G, rng), _density(G, rng)
    r = log_map(G, nu, mu)
    prev = np.inf
    for N in (10, 40, 160):
        sol = geodesic_socp(G, nu, mu, N=N)
        m_err = np.linalg.norm(r.m0 - sol.m0) / np.linalg.norm(sol.m0)
        assert m_err < 5.0 / N
        assert m_err < prev + 1e-6
        assert abs(r.W2 - sol.W2) < 1.0 / N
        prev = m_err
    # The SOCP's endpoint potential is the gradient of W2; the flow's phi0 is the
    # Hamiltonian velocity potential. In the continuum limit phi_socp = -2 phi0.
    np.testing.assert_allclose(
        graph_gradient(G, sol.phi0), -2 * graph_gradient(G, r.phi0), rtol=0.05
    )


def test_log_map_two_node_closed_form():
    G = MarkovGraph(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.5, 0.5]))
    s, t = -0.6, 0.7
    r = log_map(G, np.array([1 - s, 1 + s]), np.array([1 - t, 1 + t]), nsteps=400)
    w_ref = quad(lambda x: (1 - x**2) ** -0.25, s, t)[0] / np.sqrt(2)
    assert np.sqrt(r.W2) == pytest.approx(w_ref, abs=1e-8)


def test_log_map_damps_an_initial_guess_that_overshoots_the_floor():
    G = _grid5()
    A = G.Q.toarray() > 0

    def concentrated(c):
        m = np.ones(G.n)
        m[c] *= 10
        m[A[c]] *= 10
        return m / (m @ G.pi)

    nu, mu = concentrated(0), concentrated(24)  # opposite corners
    # Premise: the undamped linearized guess really does hit the floor here;
    # without it this test would stop exercising the damping loop silently.
    phi_lin = solve_weighted_laplacian(G, nu, G.pi * (mu - nu))
    with pytest.raises(PositivityFloorError):
        integrate_hamiltonian(G, nu, phi_lin)
    r = log_map(G, nu, mu)
    assert r.residual < 1e-9
    assert _pi_norm(G, exp_map(G, nu, r.phi0) - mu) < 1e-6


def test_log_map_reaches_tol_despite_a_solver_sized_mass_gap():
    # The flow conserves mass exactly, so a mass gap between the endpoints is a
    # residual component Newton cannot touch. An endpoint off by 1.6e-9 -- the
    # size of a SOCP barycenter's error -- used to stall Newton at exactly that
    # floor, above the default tol of 1e-9.
    G = _hypercube()
    rng = np.random.default_rng(9)
    nu, mu = _density(G, rng), _density(G, rng)
    nu_off = nu * (1 - 5e-9)
    r = log_map(G, nu_off, mu)
    assert r.residual < 1e-9


@pytest.mark.parametrize(
    "nu, target, match",
    [
        ([1.0, 1.0, 1.0], [0.0, 1.5, 1.5], "positivity floor"),
        ([1.0, 1.0, 1.0], [2.0, 1.0, 1.0], "probability density"),
        ([1.0, 1.0], [1.0, 1.0, 1.0], "shape"),
    ],
)
def test_log_map_guards(nu, target, match):
    # Real errors, not asserts: python -O would strip those.
    G = MarkovGraph(*triangle_markov_chain())
    with pytest.raises(ValueError, match=match):
        log_map(G, np.array(nu), np.array(target))


@pytest.mark.parametrize(
    "kwargs, match",
    [({"tol": 0.0}, "tol"), ({"tol": np.nan}, "tol"), ({"maxiters": -1}, "maxiters"), ({"nsteps": 0}, "nsteps")],
)
def test_log_map_rejects_bad_parameters(kwargs, match):
    G = MarkovGraph(*triangle_markov_chain())
    with pytest.raises(ValueError, match=match):
        log_map(G, np.ones(3), np.array([1.2, 0.9, 0.9]), **kwargs)


def test_log_map_reports_non_convergence_as_a_shooting_error():
    G = _hypercube()
    rng = np.random.default_rng(4)
    with pytest.raises(ShootingError, match="did not converge"):
        log_map(G, _density(G, rng), _density(G, rng), maxiters=0)


def test_the_jacobian_is_exact():
    # The Jacobian is torch's forward-mode derivative of the flow; check it
    # against an independent central difference of the same map, whose error
    # is O(h^2) ~ 1e-10 at h = 1e-5.
    from graphtransport.shooting import explog
    from graphtransport.shooting.hamiltonian import _as_tensor, rho_floor

    G = MarkovGraph(*grid_markov_chain(3))
    rng = np.random.default_rng(12)
    nu, mu = _density(G, rng), _density(G, rng)
    z = 0.05 * rng.standard_normal(G.n - 1)
    nu_t = _as_tensor(nu)

    def F(zz):
        return explog._shoot(G, nu_t, zz, 150, rho_floor(G))[0].numpy()[: G.n - 1] - mu[: G.n - 1]

    _, schedule = explog._shoot(G, nu_t, z, 150, rho_floor(G))
    exact = explog._shooting_jacobian(G, nu_t, z, schedule)
    eye = np.eye(G.n - 1)
    central = np.column_stack([(F(z + 1e-5 * eye[j]) - F(z - 1e-5 * eye[j])) / 2e-5 for j in range(G.n - 1)])
    np.testing.assert_allclose(exact, central, atol=1e-8 * np.abs(central).max())


def test_the_jacobian_is_exact_on_a_shot_that_halved_a_step():
    # The replay of a recorded schedule exists for shots that bisect a step
    # near the floor. Seed 1272 gives one: a node at ~1e-4 and a strong
    # potential, one of 150 steps halved, the same schedule under every
    # perturbation below (checked, so the central differences stay on the
    # shot's branch of the piecewise-smooth flow map).
    from graphtransport.shooting import explog
    from graphtransport.shooting.hamiltonian import _as_tensor, rho_floor

    G = MarkovGraph(*grid_markov_chain(3))
    n, floor = G.n, rho_floor(G)
    rng = np.random.default_rng(1272)
    nu = rng.uniform(0.5, 1.5, n)
    nu[rng.integers(n)] = 10 ** rng.uniform(-6, -3)
    nu /= nu @ G.pi
    z = rng.standard_normal(n - 1) * 10 ** rng.uniform(-1, 1.5)
    nu_t = _as_tensor(nu)

    _, schedule = explog._shoot(G, nu_t, z, 150, floor)
    assert max(len(steps) for steps in schedule) > 1  # a step was halved
    exact = explog._shooting_jacobian(G, nu_t, z, schedule)
    h, eye = 1e-6, np.eye(n - 1)
    for j in range(n - 1):
        (up, s_up), (down, s_down) = (explog._shoot(G, nu_t, z + h * eye[j], 150, floor),
                                      explog._shoot(G, nu_t, z - h * eye[j], 150, floor))  # fmt: skip
        assert s_up == schedule and s_down == schedule
        central = (up.numpy() - down.numpy())[: n - 1] / (2 * h)
        np.testing.assert_allclose(exact[:, j], central, atol=1e-7 * np.abs(exact).max())


# ----- analysis and the mollified fallback -----


def test_analyze_shooting_recovers_socp_synthesised_weights():
    # The SOCP synthesizes in a different discretization, so expect O(h)
    # agreement, not solver tolerance. The genuinely O(h) quantity that
    # involves no QP is the true lam's Gram-form residual lam^T A lam: Julia
    # measured a factor of 100-300 between N=2 and N=10 and requires 10.
    pytest.importorskip("cvxpy")
    from graphtransport.socp import analyze_socp, barycenter_socp

    G = _hypercube()
    rng = np.random.default_rng(6)
    refs = [_density(G, rng, offset=0.3) for _ in range(3)]
    lam = np.array([0.5, 0.3, 0.2])
    residuals = []
    for N in (2, 10):
        nu, _, _ = barycenter_socp(G, refs, lam, N=N)
        lam_hat, A = analyze_shooting(G, nu, refs, return_system=True)
        residuals.append(lam @ A @ lam / np.max(np.diag(A)))
    assert residuals[1] < residuals[0] / 10
    np.testing.assert_allclose(lam_hat, lam, atol=1e-2)
    assert lam_hat.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(lam_hat, analyze_socp(G, nu, refs, N=80), atol=1e-2)

    inits = [log_map(G, nu, r).phi0 for r in refs]
    np.testing.assert_allclose(analyze_shooting(G, nu, refs, phi0_inits=inits), lam_hat, atol=1e-8)


def test_analyze_shooting_checks_the_warm_start_count():
    G = _hypercube()
    rng = np.random.default_rng(6)
    refs = [_density(G, rng) for _ in range(3)]
    with pytest.raises(ValueError, match="one entry per reference"):
        analyze_shooting(G, refs[0], refs, phi0_inits=[np.zeros(G.n)])


def test_log_map_mollified_approximates_the_socp_on_boundary_data():
    # A target supported on two columns of a 5x5 grid: shooting cannot run
    # directly, the SOCP handles it natively and is the reference.
    pytest.importorskip("cvxpy")
    from graphtransport.socp import geodesic_socp

    G = _grid5()
    rng = np.random.default_rng(7)
    nu = _density(G, rng)
    target = np.where(np.arange(G.n) % 5 < 2, 1.0 + rng.random(G.n), 0.0)
    target /= target @ G.pi
    with pytest.raises(ValueError, match="positivity floor"):
        log_map(G, nu, target)

    # Julia uses N=400 for the reference; N=100 moves it by 9e-4 relative, a
    # fifth of the tighter tolerance below, and takes 2.6 s instead of 20 s.
    w_ref = np.sqrt(geodesic_socp(G, nu, target, N=100).W2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a stiff level may be skipped, with a warning
        r = log_map_mollified(G, nu, target)
    assert r.approximate
    assert len(r.Ws) >= 2
    assert np.all(np.diff(r.Ws) >= 0)  # W grows as epsilon -> 0: less smoothing
    assert abs(r.W - w_ref) / w_ref < 1e-2
    assert abs(r.Ws[-1] - w_ref) / w_ref < 5e-3
    assert r.W2 == pytest.approx(r.W**2)


@pytest.mark.parametrize("epsilons, match", [((1e-2,), "at least two"), ((0.5, 1.5), r"\(0, 1\)"), ((0.1, 0.0), r"\(0, 1\)")])
def test_log_map_mollified_rejects_bad_levels(epsilons, match):
    G = MarkovGraph(*triangle_markov_chain())
    with pytest.raises(ValueError, match=match):
        log_map_mollified(G, np.ones(3), np.array([1.2, 0.9, 0.9]), epsilons=epsilons)


# ----- review fixes -----


def _grid3_pair(seed=3):
    G = MarkovGraph(*grid_markov_chain(3))
    rng = np.random.default_rng(seed)
    return G, _density(G, rng), _density(G, rng)


def test_log_map_mollified_rejects_a_negative_density_before_mollifying():
    # (1 - eps) rho + eps lifts -1e-4 above the floor at eps = 1e-3, so every
    # level's log_map used to accept it and a distance came back.
    G, nu, mu = _grid3_pair()
    bad = mu.copy()
    bad[0] = -1e-4
    bad /= bad @ G.pi
    with pytest.raises(ValueError, match="negative entries"):
        log_map_mollified(G, nu, bad, epsilons=(1e-2, 1e-3))


@pytest.mark.parametrize(
    "target, match",
    [(np.full(9, np.nan), "non-finite"), (np.ones(4), r"shape \(9,\)"), (2 * np.ones(9), "probability density")],
)
def test_log_map_mollified_validates_its_raw_inputs(target, match):
    G, nu, _ = _grid3_pair()
    with pytest.raises(ValueError, match=match):
        log_map_mollified(G, nu, target)


def test_log_map_mollified_accepts_zeros():
    # the reason the function exists
    G, nu, mu = _grid3_pair()
    target = mu.copy()
    target[:3] = 0.0
    target /= target @ G.pi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert log_map_mollified(G, nu, target, epsilons=(1e-2, 1e-3)).W > 0


@pytest.mark.parametrize("epsilons", [(1e-2, 1e-2), (1e-2, 1e-3, 1e-2)])
def test_log_map_mollified_requires_distinct_levels(epsilons):
    # Coincident sqrt(eps) rows make the fit rank-deficient; lstsq returned a
    # minimum-norm split of one distance between W0 and a, reported as W.
    G, nu, mu = _grid3_pair()
    with pytest.raises(ValueError, match="distinct"):
        log_map_mollified(G, nu, mu, epsilons=epsilons)


def test_exp_map_names_a_boundary_base_point_for_either_tangent_kind():
    # The momentum branch used to reach the Laplacian solve first and report
    # "the graph is disconnected".
    G, nu, _ = _grid3_pair()
    nu = nu.copy()
    nu[0] = 0.0
    for tangent in (np.full(len(G.E), 0.01), np.zeros(G.n)):
        with pytest.raises(ValueError, match="positivity floor"):
            exp_map(G, nu, tangent)


def test_log_map_rejects_a_non_finite_warm_start():
    # used to be halved thirteen times and reported as "too far apart"
    G, nu, mu = _grid3_pair()
    with pytest.raises(ValueError, match="phi0_init has non-finite entries"):
        log_map(G, nu, mu, phi0_init=np.full(G.n, np.nan))


def test_analyze_shooting_names_the_2d_refs_trap():
    G, nu, mu = _grid3_pair()
    with pytest.raises(ValueError, match="sequence of densities"):
        analyze_shooting(G, nu, np.column_stack([mu, nu]))


def test_verbose_log_map_goes_through_logging(caplog):
    G, nu, mu = _grid3_pair()
    with caplog.at_level("INFO", logger="graphtransport.shooting.explog"):
        log_map(G, nu, mu, verbose=True)
    assert any("residual" in rec.getMessage() for rec in caplog.records)
