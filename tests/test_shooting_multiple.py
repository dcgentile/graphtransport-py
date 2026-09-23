"""Multiple shooting (segments > 1) for the log map and the geodesic."""

import numpy as np
import pytest
import torch

from graphtransport import MarkovGraph, analysis, barycenter, geodesic, grid_markov_chain, transport_cost
from graphtransport.shooting import hamiltonian, log_map
from graphtransport.shooting.geodesic import geodesic_shooting
from graphtransport.shooting.multiple import segment_steps


def _corner_pair(k, shift):
    G = MarkovGraph(*grid_markov_chain(k))
    xy = np.array([(i % k, i // k) for i in range(G.n)], dtype=float)

    def bump(center):
        rho = np.exp(-((xy - center) ** 2).sum(axis=1) / 8) + 0.05
        return rho / (rho @ G.pi)

    return G, bump((0, 0)), bump((shift, shift))


@pytest.fixture(scope="module")
def short_pair():
    # corner to corner on a 5x5 grid: single shooting converges
    return _corner_pair(5, 4)


@pytest.fixture(scope="module")
def long_pair():
    # corner to corner on a 10x10 grid: 19 Newton iterations for single shooting
    return _corner_pair(10, 9)


@pytest.fixture(scope="module")
def long_solves(long_pair):
    # each 10x10 solve takes seconds; solved once and shared
    G, A, B = long_pair
    return log_map(G, A, B, segments=1), log_map(G, A, B, segments=4)


def test_segment_steps_cover_the_step_grid():
    assert segment_steps(150, 4) == [38, 38, 37, 37]
    assert segment_steps(7, 7) == [1] * 7
    assert sum(segment_steps(151, 6)) == 151


@pytest.mark.parametrize("segments, nsteps", [(2, 150), (3, 150), (3, 50), (7, 150)])
def test_multiple_shooting_solves_the_same_discrete_problem(short_pair, segments, nsteps):
    G, A, B = short_pair
    single = geodesic_shooting(G, A, B, nsteps=nsteps)
    multi = geodesic_shooting(G, A, B, nsteps=nsteps, segments=segments)
    assert multi.rho.shape == single.rho.shape == (G.n, nsteps + 1)
    assert multi.W2 == pytest.approx(single.W2, rel=1e-10)
    for field in ("rho", "m", "phi0", "phi1"):
        np.testing.assert_allclose(getattr(multi, field), getattr(single, field), atol=1e-8, err_msg=field)


def test_segments_one_is_single_shooting(short_pair):
    G, A, B = short_pair
    r = log_map(G, A, B, segments=1)
    assert r.starts is None
    assert r.W2 == log_map(G, A, B).W2


def test_a_long_transport_takes_fewer_newton_steps_and_agrees_with_julia(long_pair, long_solves):
    # Single shooting (exact Jacobian) takes 19 iterations here, as Julia does;
    # four short segments bend less, and Newton needs about half as many.
    G, A, B = long_pair
    single, r = long_solves
    assert r.iters < single.iters
    assert r.W2 == pytest.approx(138.60667081914244, rel=1e-12)  # Julia's log_map
    rho_s, phi_s = r.starts
    assert rho_s.shape == phi_s.shape == (G.n, 4)
    # the Hamiltonian is conserved along the geodesic, so every segment carries the same W2
    for k in range(4):
        assert 2 * hamiltonian(G, rho_s[:, k], phi_s[:, k]) == pytest.approx(r.W2, rel=1e-8)

    sol = geodesic(G, A, B, segments=4, fallback=False)
    assert sol.W2 == pytest.approx(r.W2, rel=1e-10)
    np.testing.assert_allclose(sol.rho[:, 0], A, atol=1e-12)
    np.testing.assert_allclose(sol.rho[:, -1], B, atol=1e-8)
    assert transport_cost(G, A, B, segments=2, fallback=False) == pytest.approx(np.sqrt(r.W2), rel=1e-8)


def test_the_jacobian_is_exact(short_pair):
    # each segment's block is the tangent model along its schedule; check the
    # whole sparse Jacobian against central differences of the residual at a
    # state away from the solution, on the same schedules
    from graphtransport.shooting.hamiltonian import rho_floor
    from graphtransport.shooting.multiple import _System

    G, A, B = short_pair
    system = _System(G, A, B, 3, 150, rho_floor(G))
    rng = np.random.default_rng(3)
    rho_s = np.column_stack([A, (A + B) / 2, B]) * (1 + 0.05 * rng.standard_normal((G.n, 3)))
    phi_s = 0.1 * rng.standard_normal((G.n, 3))
    x = system.pack(rho_s, phi_s)
    R, schedules = system.residual(x)
    J = system.jacobian(x, schedules).toarray()
    h = 1e-6
    for j in range(0, x.size, 3):  # every third column keeps it quick
        e = np.zeros(x.size)
        e[j] = h
        (up, s_up), (down, s_down) = system.residual(x + e), system.residual(x - e)
        assert s_up == schedules and s_down == schedules
        central = (system.reduced(up) - system.reduced(down)) / (2 * h)
        np.testing.assert_allclose(J[:, j], central, atol=1e-7 * np.abs(J).max())


def test_a_single_shooting_potential_warm_starts_multiple_shooting(long_pair, long_solves):
    G, A, B = long_pair
    r = log_map(G, A, B, segments=4, phi0_init=long_solves[0].phi0)
    assert r.iters <= 1
    assert r.W2 == pytest.approx(138.60667081914244, rel=1e-12)


def test_the_long_transport_agrees_with_the_socp(long_pair, long_solves):
    pytest.importorskip("cvxpy")
    G, A, B = long_pair
    W2 = long_solves[1].W2
    socp10, socp40 = (geodesic(G, A, B, method="socp", N=N).W2 for N in (10, 40))
    # the SOCP's time-discretization error shrinks towards the shooting value
    assert abs(socp40 - W2) < abs(socp10 - W2) < 1.0
    assert socp40 == pytest.approx(W2, rel=1e-3)


@pytest.mark.parametrize("segments", [0, -1, 151, 2.0, True, "2", "AUTO", None])
def test_segments_is_validated(short_pair, segments):
    G, A, B = short_pair
    with pytest.raises(ValueError, match="segments must be 'auto' or an integer between 1 and nsteps"):
        log_map(G, A, B, segments=segments)


def test_segments_is_a_shooting_keyword(short_pair):
    G, A, B = short_pair
    with pytest.raises(TypeError, match="segments .*method='shooting'"):
        geodesic(G, A, B, method="socp", segments=2)


def test_barycenter_and_analysis_take_segments():
    # wiring and agreement, which the grid size does not change: 3x3, two references
    G, A, B = _corner_pair(3, 2)
    refs, lam = [A, B], [0.6, 0.4]
    nu1, J1, _ = barycenter(G, refs, lam, fallback=False)
    nu3, J3, info = barycenter(G, refs, lam, segments=3, fallback=False)
    assert info["method"] == "shooting"
    assert J3 == pytest.approx(J1, rel=1e-8)
    np.testing.assert_allclose(nu3, nu1, atol=1e-6)
    np.testing.assert_allclose(analysis(G, nu3, refs, segments=3, fallback=False),
                               analysis(G, nu3, refs, fallback=False), atol=1e-8)  # fmt: skip


def test_gradients_through_multiple_shooting_are_exact():
    G = MarkovGraph(*grid_markov_chain(3))
    pi = torch.tensor(G.pi)
    rng = np.random.default_rng(0)
    a, b = (x / (x @ G.pi) for x in (rng.uniform(0.5, 1.5, G.n) for _ in range(2)))
    b = torch.tensor(b)

    def f(x):  # the start density's gradient is the one through the replayed flow's VJP
        return geodesic(G, x / (x @ pi), b, tol=1e-13, segments=3).W2

    assert torch.autograd.gradcheck(f, (torch.tensor(a, requires_grad=True),), eps=1e-6, atol=1e-6, rtol=1e-5)


# ----- segments="auto", the default -----


def test_auto_keeps_a_short_transport_on_single_shooting():
    # near-uniform densities: Newton's first step is a full one, so no switch
    G = MarkovGraph(*grid_markov_chain(5))
    rng = np.random.default_rng(0)
    a, b = (x / (x @ G.pi) for x in (rng.uniform(0.5, 1.5, G.n) for _ in range(2)))
    auto, single = log_map(G, a, b), log_map(G, a, b, segments=1)
    assert auto.starts is None
    assert auto.iters == single.iters and auto.W2 == single.W2


def test_auto_switches_a_long_transport_to_multiple_shooting(long_pair, long_solves):
    # corner to corner: the first step is cut to 1/16, so auto continues with K=8
    G, A, B = long_pair
    auto = log_map(G, A, B)
    assert auto.starts is not None and auto.starts[0].shape == (G.n, 8)
    assert auto.iters < long_solves[0].iters  # fewer than single shooting's 19
    assert auto.W2 == pytest.approx(138.60667081914244, rel=1e-11)  # Julia's log_map


def test_auto_carries_on_with_single_shooting_if_the_switch_fails(long_pair, long_solves, monkeypatch):
    from graphtransport.shooting import ShootingError, explog

    def failing(*args, **kwargs):
        raise ShootingError("multiple shooting failed (test)")

    monkeypatch.setattr(explog, "_log_map_multiple", failing)
    G, A, B = long_pair
    r = log_map(G, A, B)
    assert r.starts is None
    assert r.W2 == pytest.approx(long_solves[0].W2, rel=1e-12)
    assert r.iters == long_solves[0].iters  # single shooting's own path, from its first step on


def test_auto_gives_up_quickly_where_multiple_shooting_stalls():
    # 7x7 corner bumps: the first step is cut to 1/8, so auto switches, but K=8
    # from its default start stalls (a first step of 1/16, residual 41 -> 39
    # over 8 steps) and would fail after 50. auto abandons it after one step and
    # carries on by single shooting -- same answer, and 2.3 s rather than 12.6 s.
    G, A, B = _corner_pair(7, 6)
    xy = np.array([(i % 7, i // 7) for i in range(G.n)], dtype=float)
    a = np.exp(-(xy**2).sum(axis=1) / (2 * 1.4**2)) + 0.05
    b = a[::-1].copy()
    a, b = a / (a @ G.pi), b / (b @ G.pi)
    single = log_map(G, a, b, segments=1)
    auto = log_map(G, a, b)
    assert auto.starts is None  # ended on single shooting
    assert auto.iters == single.iters and auto.W2 == pytest.approx(single.W2, rel=1e-12)
