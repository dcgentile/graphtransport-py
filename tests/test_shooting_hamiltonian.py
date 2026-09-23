import importlib
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from graphtransport import (
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    LogarithmicMean,
    MarkovGraph,
    QuadLogMean,
    triangle_markov_chain,
    weighted_hypercube_markov_chain,
)
from graphtransport.graph import graph_divergence, graph_gradient
from graphtransport.shooting import (
    PositivityFloorError,
    hamiltonian,
    hamiltonian_flow,
    integrate_hamiltonian,
    rho_floor,
)

MEANS = [GeometricMean(), ArithmeticMean(), HarmonicMean(), LogarithmicMean(), QuadLogMean(8)]


def _seeded_hypercube():
    # weighted_hypercube_markov_chain draws fresh weights on every call (as in
    # Julia); unseeded, each test run would check a different graph.
    return weighted_hypercube_markov_chain(rng=0)


GRAPHS = [triangle_markov_chain, _seeded_hypercube]


def _two_node(mean=None):
    return MarkovGraph(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.5, 0.5]), mean=mean)


def _interior_state(G, rng, scale=0.1):
    rho = rng.random(G.n) + 0.5
    rho /= rho @ G.pi
    phi = scale * rng.standard_normal(G.n)
    phi -= phi @ G.pi  # gauge: <phi, 1>_pi = 0
    return rho, phi


def _w_two_node(a, b):
    # closed form on the two-node graph: rho(r) = [1-r, 1+r],
    # W = (1/sqrt 2) int_a^b (1 - r^2)^(-1/4) dr
    return abs(quad(lambda r: (1 - r**2) ** -0.25, a, b)[0] / np.sqrt(2))


@pytest.mark.parametrize("builder", GRAPHS, ids=["triangle", "weighted_hypercube"])
def test_conservation_laws(builder):
    # Mass and H are conserved, and 2H is the squared transport distance
    # between the flow's own endpoints -- i.e. the flow traces a genuine
    # geodesic, not merely a curve that conserves H by construction.
    G = MarkovGraph(*builder())
    rng = np.random.default_rng(1)
    for _ in range(3):
        rho0, phi0 = _interior_state(G, rng)
        H0 = hamiltonian(G, rho0, phi0)
        rho_path, phi_path = integrate_hamiltonian(G, rho0, phi0, nsteps=200)

        assert rho_path[:, -1] @ G.pi == pytest.approx(1.0, abs=1e-10)
        drift = max(abs(hamiltonian(G, rho_path[:, i], phi_path[:, i]) - H0) for i in range(rho_path.shape[1]))
        assert drift < 1e-4  # RK4 truncation error


@pytest.mark.parametrize("builder", GRAPHS, ids=["triangle", "weighted_hypercube"])
def test_twice_the_hamiltonian_is_the_squared_distance_to_the_endpoint(builder):
    pytest.importorskip("cvxpy")
    from graphtransport import geodesic

    G = MarkovGraph(*builder())
    rng = np.random.default_rng(1)
    for _ in range(3):
        rho0, phi0 = _interior_state(G, rng)
        H0 = hamiltonian(G, rho0, phi0)
        rho_end = integrate_hamiltonian(G, rho0, phi0, nsteps=200)[0][:, -1]
        # phi0 is deliberately small, so 2H is small and the residual is the
        # solver's noise floor rather than a shrinking O(1/N) error: check the
        # absolute magnitude at both N rather than demanding improvement.
        for N in (20, 100):
            assert abs(2 * H0 - geodesic(G, rho0, rho_end, method="socp", N=N).W2) < 1e-4


@pytest.mark.parametrize("c0, atol", [(0.1, 1e-9), (0.3, 1e-8)])
def test_two_node_closed_form(c0, atol):
    # Fully independent of the SOCP: on two nodes the gauge-fixed potential is
    # the scalar (c0, -c0), so sqrt(2 H) can be compared against the quadrature
    # distance between rho0 and wherever the flow actually lands.
    G = _two_node()
    r0 = -0.6
    rho0 = np.array([1 - r0, 1 + r0])
    phi0 = np.array([c0, -c0])
    rho_path, _ = integrate_hamiltonian(G, rho0, phi0, nsteps=400)
    r_end = 1 - rho_path[0, -1]
    assert np.sqrt(2 * hamiltonian(G, rho0, phi0)) == pytest.approx(_w_two_node(r0, r_end), abs=atol)


def test_positivity_floor_is_raised_not_silently_crossed():
    # c0 large enough to drive r past the boundary within [0, 1]. numpy gives
    # nan rather than Julia's DomainError, so silence is the failure mode here.
    G = _two_node()
    rho0 = np.array([1.6, 0.4])
    with pytest.raises(PositivityFloorError, match="method='socp'"):
        integrate_hamiltonian(G, rho0, np.array([0.6, -0.6]), nsteps=400)


@pytest.mark.parametrize("mean", MEANS, ids=lambda m: type(m).__name__)
def test_every_mean_conserves_the_hamiltonian(mean):
    # The shooting maps take every AdmissibleMean, including the exact
    # LogarithmicMean that the SOCP can only reach through QuadLogMean.
    G = MarkovGraph(*triangle_markov_chain(), mean=mean)
    rng = np.random.default_rng(3)
    rho0, phi0 = _interior_state(G, rng)
    H0 = hamiltonian(G, rho0, phi0)
    rho_path, phi_path = integrate_hamiltonian(G, rho0, phi0, nsteps=200)
    drift = max(abs(hamiltonian(G, rho_path[:, i], phi_path[:, i]) - H0) for i in range(201))
    assert drift < 1e-6
    assert rho_path[:, -1] @ G.pi == pytest.approx(1.0, abs=1e-10)


def test_quadlog_approximates_the_exact_logarithmic_flow():
    # QuadLogMean(8) is accurate to ~1e-10 as a mean, so the flows it drives
    # should agree with the exact logarithmic one to about that.
    rho0 = np.array([1.6, 0.4])
    phi0 = np.array([0.15, -0.15])
    exact = integrate_hamiltonian(_two_node(LogarithmicMean()), rho0, phi0, nsteps=200)[0]
    quad_log = integrate_hamiltonian(_two_node(QuadLogMean(8)), rho0, phi0, nsteps=200)[0]
    np.testing.assert_allclose(quad_log, exact, atol=1e-9)


def test_flow_matches_the_divergence_identity_and_preserves_mass_infinitesimally():
    G = MarkovGraph(*triangle_markov_chain())
    rng = np.random.default_rng(5)
    rho, phi = _interior_state(G, rng, scale=0.5)
    rho_dot, _ = hamiltonian_flow(G, rho, phi)
    theta = G.mean(rho[G.E[:, 0]], rho[G.E[:, 1]])
    np.testing.assert_allclose(rho_dot, -graph_divergence(G, theta * graph_gradient(G, phi)), rtol=1e-12)
    assert rho_dot @ G.pi == pytest.approx(0.0, abs=1e-14)


def test_flow_is_gauge_invariant():
    # phi is defined up to an additive constant: only grad phi enters the
    # equations of motion. A sign slip in graph_gradient would break this
    # while leaving mass and H conservation intact.
    G = MarkovGraph(*triangle_markov_chain())
    rng = np.random.default_rng(11)
    rho0, phi0 = _interior_state(G, rng)
    rho_a, phi_a = integrate_hamiltonian(G, rho0, phi0, nsteps=100)
    rho_b, phi_b = integrate_hamiltonian(G, rho0, phi0 + 3.7, nsteps=100)
    np.testing.assert_allclose(rho_b, rho_a, atol=1e-12)
    np.testing.assert_allclose(phi_b - phi_a, 3.7, atol=1e-12)


def test_scaling_phi_reparametrises_time():
    # theta is 1-homogeneous, so scaling the potential rescales time:
    # rho(T; a phi) == rho(aT; phi). Breaks if a mean loses its homogeneity.
    G = MarkovGraph(*triangle_markov_chain())
    rng = np.random.default_rng(11)
    rho0, phi0 = _interior_state(G, rng)
    a = 2.0
    fast = integrate_hamiltonian(G, rho0, a * phi0, nsteps=200, T=1.0)[0][:, -1]
    slow = integrate_hamiltonian(G, rho0, phi0, nsteps=200, T=a)[0][:, -1]
    np.testing.assert_allclose(fast, slow, atol=1e-12)


@pytest.mark.parametrize("rho", [np.array([0.0, 1.0, 2.0]), np.array([1e-14, 1.0, 2.0]), np.array([-1.0, 1.5, 2.5])])
def test_public_flow_and_hamiltonian_reject_non_interior_densities(rho):
    # At the boundary phi_dot is -inf and just inside it the result is finite
    # but meaningless; neither announces itself downstream, so both public
    # entry points check rather than trusting the caller.
    G = MarkovGraph(*triangle_markov_chain())
    phi = np.array([0.1, -0.05, -0.05])
    with pytest.raises(ValueError, match="positivity floor"):
        hamiltonian_flow(G, rho, phi)
    with pytest.raises(ValueError, match="positivity floor"):
        hamiltonian(G, rho, phi)


def test_the_rk4_stages_do_not_pay_for_the_domain_check(monkeypatch):
    # The integrator validates once per call, not four times per step: the
    # stages go through the unchecked routine. (importlib because the package
    # re-exports the function `hamiltonian`, which shadows the module of the
    # same name on `graphtransport.shooting`.)
    module = importlib.import_module("graphtransport.shooting.hamiltonian")

    calls = []
    original = module._check_interior
    monkeypatch.setattr(module, "_check_interior", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    integrate_hamiltonian(
        MarkovGraph(*triangle_markov_chain()), np.array([1.5, 0.9, 0.6]), np.array([0.1, -0.05, -0.05]), nsteps=10
    )
    assert len(calls) == 1


@pytest.mark.parametrize("max_halvings", [-5, -1, 2.5, True, None])
def test_invalid_max_halvings(max_halvings):
    # A negative value would disable bisection with no symptom: the first
    # floor crossing raises instead of being retried.
    with pytest.raises(ValueError, match="max_halvings must be an integer >= 0"):
        integrate_hamiltonian(_two_node(), np.array([1.6, 0.4]), np.array([0.1, -0.1]), max_halvings=max_halvings)


def test_rho_floor_scales_with_the_stationary_distribution():
    G = MarkovGraph(*_seeded_hypercube())
    assert rho_floor(G) == pytest.approx(1e-6 * G.pi.min())
    assert rho_floor(G, rtol=1e-3) == pytest.approx(1e-3 * G.pi.min())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"nsteps": 0}, "nsteps must be an integer >= 1"),
        ({"nsteps": 2.5}, "nsteps must be an integer >= 1"),
        ({"nsteps": True}, "nsteps must be an integer >= 1"),
        ({"T": 0.0}, "T must be a positive"),
        ({"T": -1.0}, "T must be a positive"),
        ({"T": np.inf}, "T must be a positive"),
    ],
)
def test_invalid_integration_parameters(kwargs, match):
    G = _two_node()
    with pytest.raises(ValueError, match=match):
        integrate_hamiltonian(G, np.array([1.6, 0.4]), np.array([0.1, -0.1]), **kwargs)


@pytest.mark.parametrize(
    "rho0, match",
    [
        (np.array([0.0, 2.0]), "violates the positivity floor"),
        (np.array([-0.5, 2.5]), "violates the positivity floor"),
        (np.array([1.0, 1.0, 1.0]), r"shape \(2,\)"),
        (np.array([np.nan, 2.0]), "non-finite"),
    ],
)
def test_invalid_initial_density(rho0, match):
    # An assert would be stripped by `python -O`, and the flow's output for a
    # boundary density is silently nan rather than an error.
    with pytest.raises(ValueError, match=match):
        integrate_hamiltonian(_two_node(), rho0, np.array([0.1, -0.1]))


@pytest.mark.parametrize("phi0, match", [(np.zeros(3), r"shape \(2,\)"), (np.array([np.nan, 0.0]), "non-finite")])
def test_invalid_initial_potential(phi0, match):
    with pytest.raises(ValueError, match=match):
        integrate_hamiltonian(_two_node(), np.array([1.6, 0.4]), phi0)


def test_a_constant_potential_does_not_move_the_density():
    # grad phi == 0, so H == 0 and the flow is stationary.
    G = MarkovGraph(*triangle_markov_chain())
    rho0 = np.array([1.5, 0.9, 0.6])
    phi0 = np.full(G.n, 2.5)
    assert hamiltonian(G, rho0, phi0) == 0.0
    rho_path, phi_path = integrate_hamiltonian(G, rho0, phi0, nsteps=20)
    np.testing.assert_allclose(rho_path, np.repeat(rho0[:, None], 21, axis=1), atol=1e-14)
    np.testing.assert_allclose(phi_path, np.repeat(phi0[:, None], 21, axis=1), atol=1e-14)


def test_time_reversal_returns_to_the_start():
    # (rho, phi) -> (rho, -phi) reverses the flow, so running forward then
    # backward is the identity up to truncation error.
    G = MarkovGraph(*triangle_markov_chain())
    rng = np.random.default_rng(7)
    rho0, phi0 = _interior_state(G, rng)
    rho_mid, phi_mid = (p[:, -1] for p in integrate_hamiltonian(G, rho0, phi0, nsteps=200))
    rho_back, phi_back = (p[:, -1] for p in integrate_hamiltonian(G, rho_mid, -phi_mid, nsteps=200))
    np.testing.assert_allclose(rho_back, rho0, atol=1e-10)
    np.testing.assert_allclose(-phi_back, phi0, atol=1e-10)


def test_more_than_one_torch_thread_warns_once():
    import torch

    from graphtransport.shooting import TorchThreadsWarning

    # the module, not the function graphtransport.shooting exports under the same name
    ham = importlib.import_module("graphtransport.shooting.hamiltonian")

    G = _two_node()
    rho0, phi0 = np.array([1.2, 0.8]), np.array([0.1, -0.1])
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        ham._threads_warned = False
        with pytest.warns(TorchThreadsWarning, match="torch is using 2 threads.*set_num_threads") as record:
            integrate_hamiltonian(G, rho0, phi0, nsteps=5)
        assert record[0].filename == __file__  # the caller's line, not the solver's
        with warnings.catch_warnings():
            warnings.simplefilter("error", TorchThreadsWarning)
            integrate_hamiltonian(G, rho0, phi0, nsteps=5)  # once per process
        torch.set_num_threads(1)
        ham._threads_warned = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", TorchThreadsWarning)
            integrate_hamiltonian(G, rho0, phi0, nsteps=5)  # one thread: nothing to say
    finally:
        torch.set_num_threads(threads)


def test_warnings_skip_package_frames_only():
    ham = importlib.import_module("graphtransport.shooting.hamiltonian")
    assert ham._in_package("graphtransport")
    assert ham._in_package("graphtransport.shooting.hamiltonian")
    # a user's module that merely shares the prefix is the caller, not the library
    assert not ham._in_package("graphtransport_experiments")
    assert not ham._in_package("__main__")
