import numpy as np
import pytest
from scipy.integrate import quad

cp = pytest.importorskip("cvxpy")

from graphtransport import (  # noqa: E402
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    MarkovGraph,
    QuadLogMean,
    triangle_markov_chain,
)
from graphtransport.graph import graph_gradient, metric_tensor  # noqa: E402
from graphtransport.socp import geodesic_block, geodesic_socp  # noqa: E402

TWO_NODE_Q = np.array([[0.0, 1.0], [1.0, 0.0]])
TWO_NODE_PI = np.array([0.5, 0.5])


def _two_node(mean=None):
    return MarkovGraph(TWO_NODE_Q, TWO_NODE_PI, mean=mean)


def _rho(r):
    return np.array([1.0 - r, 1.0 + r])


def _w_ref(theta, a, b):
    # generalized two-node closed form: W = (1/sqrt 2) int_a^b theta(1-r, 1+r)^(-1/2) dr
    return quad(lambda r: float(theta(1 - r, 1 + r)) ** -0.5, a, b)[0] / np.sqrt(2)


def test_two_node_closed_form_and_ode_path_converge_at_rate_h():
    # Mirrors "geodesic_socp vs two-node closed form": both the SOCP and the
    # explicit-Euler reference are O(h); errors must be < 2/N and not grow.
    G = _two_node()
    eps = 1e-4
    s, t = -1.0 + eps, 1.0 - eps
    rhoA, rhoB = _rho(s), _rho(t)
    C = _w_ref(GeometricMean(), s, t)

    n_ode = 2000
    gamma = np.empty(n_ode + 1)
    gamma[0] = s
    for i in range(n_ode):
        x = gamma[i]
        gamma[i + 1] = np.clip(x + (1 / n_ode) * C * np.sqrt(2) * max(0.0, (1 - x) * (1 + x)) ** 0.25, s, t)

    def ode_gamma(tau):
        raw = tau * n_ode
        lo = int(np.clip(np.floor(raw), 0, n_ode - 1))
        frac = raw - lo
        return gamma[lo] * (1 - frac) + gamma[lo + 1] * frac

    prev_w, prev_rho = np.inf, np.inf
    for N in (10, 20, 50):
        sol = geodesic_socp(G, rhoA, rhoB, N=N)
        assert sol.status == cp.OPTIMAL
        w_err = abs(np.sqrt(sol.W2) - C)
        assert w_err < 2.0 / N
        ts = np.linspace(0, 1, N + 1)
        rho_err = max(abs(sol.rho[1, i] - (1.0 + ode_gamma(tau))) for i, tau in enumerate(ts))
        assert rho_err < 2.0 / N
        assert w_err < prev_w + 1e-9 and rho_err < prev_rho + 1e-9
        prev_w, prev_rho = w_err, rho_err


def test_endpoint_potentials_are_gradients_of_w2():
    # Mirrors "endpoint potentials: dual sign/scale calibration": phi must match
    # a central finite difference of the SOCP's own W2 (tight) and the closed
    # form's derivative (O(h)).
    G = _two_node()
    s, t = -0.6, 0.7
    C = _w_ref(GeometricMean(), s, t)
    dW2_ds = -np.sqrt(2) * C * (1 - s**2) ** -0.25
    dW2_dt = np.sqrt(2) * C * (1 - t**2) ** -0.25
    eps = 1e-4

    def pair(phi):  # d rho / dr = [-1, 1] so <phi, d rho>_pi = (phi[1] - phi[0]) / 2
        return (phi[1] - phi[0]) / 2

    for N in (5, 20):
        sol = geodesic_socp(G, _rho(s), _rho(t), N=N)
        fd_s = (geodesic_socp(G, _rho(s + eps), _rho(t), N=N).W2 - geodesic_socp(G, _rho(s - eps), _rho(t), N=N).W2) / (
            2 * eps
        )
        fd_t = (geodesic_socp(G, _rho(s), _rho(t + eps), N=N).W2 - geodesic_socp(G, _rho(s), _rho(t - eps), N=N).W2) / (
            2 * eps
        )
        assert pair(sol.phi0) == pytest.approx(fd_s, rel=1e-3)
        assert pair(sol.phi1) == pytest.approx(fd_t, rel=1e-3)
        assert abs(pair(sol.phi0) - dW2_ds) < 1.0 / N
        assert abs(pair(sol.phi1) - dW2_dt) < 1.0 / N


def test_continuity_duals_relate_to_momenta_through_midpoint_density():
    # m_t = -(1/2h) theta(rhobar_t) grad(psi_t / pi), psi_t the continuity duals.
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    N = 4
    h = 1.0 / N
    blk = geodesic_block(G, N, h, np.array([2.0, 0.5, 0.5]), np.array([0.4, 0.4, 2.2]))
    problem = cp.Problem(cp.Minimize(h * blk["action"]), blk["constraints"])
    problem.solve(solver="CLARABEL")
    assert problem.status == cp.OPTIMAL
    rho_path = np.asarray(blk["rho"].value)
    m_path = np.asarray(blk["m"].value)
    duals = np.asarray(blk["c_cont"].dual_value)
    sign = None
    for t in range(N):
        psi = duals[:, t] / G.pi
        rbar = (rho_path[:, t] + rho_path[:, t + 1]) / 2
        predicted = -(1 / (2 * h)) * metric_tensor(G, rbar) * graph_gradient(G, psi)
        # cvxpy's dual sign convention is the opposite of JuMP's; the relation
        # must hold with one consistent sign across all time steps
        if sign is None:
            sign = 1.0 if np.allclose(predicted, m_path[:, t], rtol=1e-3, atol=1e-8) else -1.0
        np.testing.assert_allclose(sign * predicted, m_path[:, t], rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("theta", [GeometricMean(), ArithmeticMean(), HarmonicMean(), QuadLogMean(8)], ids=repr)
def test_two_node_closed_form_per_mean(theta):
    s, t = -0.6, 0.7
    C = _w_ref(theta, s, t)
    prev = np.inf
    for N in (10, 40):
        sol = geodesic_socp(_two_node(theta), _rho(s), _rho(t), N=N)
        assert sol.status == cp.OPTIMAL
        err = abs(np.sqrt(sol.W2) - C)
        assert err < 2.0 / N
        assert err < prev + 1e-6
        prev = err


def test_arithmetic_mean_two_node_is_exact():
    s, t = -0.6, 0.7
    assert ArithmeticMean()(1 - 0.3, 1 + 0.3) == 1.0  # constant mobility => W = (t - s)/sqrt 2
    sol = geodesic_socp(_two_node(ArithmeticMean()), _rho(s), _rho(t), N=5)
    assert np.sqrt(sol.W2) == pytest.approx((t - s) / np.sqrt(2), abs=1e-6)


def test_logarithmic_mean_has_no_conic_form():
    with pytest.raises(ValueError, match="no conic representation"):
        geodesic_socp(_two_node(LogarithmicMean()), _rho(-0.6), _rho(0.7), N=5)


def test_distance_ordering_across_means():
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    a, b = np.array([2.0, 0.5, 0.5]), np.array([0.5, 0.5, 2.0])
    W2 = {
        name: geodesic_socp(G.with_mean(theta), a, b, N=20).W2
        for name, theta in (
            ("H", HarmonicMean()),
            ("G", GeometricMean()),
            ("L", QuadLogMean(8)),
            ("A", ArithmeticMean()),
        )
    }
    assert W2["H"] > W2["G"] > W2["L"] > W2["A"]
    assert geodesic_socp(G.with_mean(QuadLogMean(12)), a, b, N=20).W2 == pytest.approx(W2["L"], rel=1e-6)


def test_solution_structure():
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    sol = geodesic_socp(G, [2.0, 0.5, 0.5], [0.5, 0.5, 2.0], N=6)
    assert sol.rho.shape == (3, 7) and sol.m.shape == (3, 6)
    np.testing.assert_allclose(sol.m0, sol.m[:, 0])
    np.testing.assert_allclose(sol.rho[:, 0], [2.0, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(sol.rho[:, -1], [0.5, 0.5, 2.0], atol=1e-6)
    np.testing.assert_allclose(sol.rho.T @ G.pi, 1.0, atol=1e-6)  # mass conserved along the path
    assert sol.solvetime >= 0


@pytest.mark.parametrize("N", [0, -3, 2.5, True, "4", None])
def test_invalid_N_is_rejected_before_the_solver(N):
    # N=0 used to be a ZeroDivisionError and N=2.5 a cvxpy dimension error.
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    with pytest.raises(ValueError, match="N must be an integer >= 1"):
        geodesic_socp(G, [2.0, 0.5, 0.5], [0.5, 0.5, 2.0], N=N)


def test_failed_solve_with_check_false_returns_nan_of_the_right_shape():
    # An infeasible solve leaves every variable's .value at None, and
    # np.asarray(None, dtype=float) is the 0-d array nan rather than an error,
    # so the solution's fields used to come back 0-d (and m[:, 0] raised).
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    bad = np.array([2.0, 0.5, 1.5])  # wrong mass: no feasible path
    with pytest.raises(RuntimeError, match="not a solution"):
        geodesic_socp(G, bad, [0.5, 0.5, 2.0], N=4)

    sol = geodesic_socp(G, bad, [0.5, 0.5, 2.0], N=4, check=False)
    assert sol.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert sol.rho.shape == (3, 5) and sol.m.shape == (3, 4) and sol.m0.shape == (3,)
    assert np.isnan(sol.rho).all() and np.isnan(sol.m).all()


def test_failed_solve_error_carries_a_hint():
    Q, pi = triangle_markov_chain()
    G = MarkovGraph(Q, pi)
    with pytest.raises(RuntimeError, match="check=False"):
        geodesic_socp(G, [2.0, 0.5, 1.5], [0.5, 0.5, 2.0], N=4)
