import numpy as np
import pytest

from graphtransport import MarkovGraph, markov_chain_from_edge_list
from graphtransport.sinkhorn import bfs_hops, graph_diameter, ground_cost


def _graph(edges):
    return MarkovGraph(*markov_chain_from_edge_list(edges))


def _grid(n):
    edges = []
    for i in range(n * n):
        if (i + 1) % n != 0:
            edges.append((i, i + 1))
        if i + n < n * n:
            edges.append((i, i + n))
    return _graph(edges)


TRIANGLE = [(0, 1), (1, 2), (0, 2)]
DISCONNECTED = [(0, 1), (2, 3)]


def test_bfs_hops_grid():
    H = bfs_hops(_grid(3))
    assert H[0, 0] == 0
    assert H[0, 1] == 1
    assert H[0, 4] == 2  # corner to centre
    assert H[0, 8] == 4  # corner to opposite corner
    np.testing.assert_allclose(H, H.T)


def test_graph_diameter():
    assert graph_diameter(_graph(TRIANGLE)) == 1
    assert graph_diameter(_grid(3)) == 4
    assert graph_diameter(_grid(4)) == 6


def test_graph_diameter_disconnected_raises():
    with pytest.raises(ValueError, match="disconnected"):
        graph_diameter(_graph(DISCONNECTED))


def test_shortest_path_cost_is_normalized_squared_hops():
    G = _grid(3)
    C = ground_cost(G, "shortest_path")
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(np.diag(C), 0.0)
    assert C.max() == pytest.approx(1.0)
    assert C[0, 8] == pytest.approx(1.0)  # 4 hops -> 16/16
    assert C[0, 1] == pytest.approx(1 / 16)
    raw = ground_cost(G, "shortest_path", normalize=False)
    assert raw[0, 8] == pytest.approx(16.0)
    assert raw[0, 1] == pytest.approx(1.0)


def test_diffusion_cost_properties():
    G = _grid(3)
    C = ground_cost(G, "diffusion")
    assert C.shape == (9, 9)
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(np.diag(C), 0.0)
    assert np.all(np.isfinite(C))
    assert C.max() == pytest.approx(1.0)
    off_diag = C[~np.eye(9, dtype=bool)]
    assert np.all(off_diag > 0)


def test_diffusion_cost_matches_definition_at_explicit_t():
    G = _grid(3)
    t = 2
    Qt = np.linalg.matrix_power((np.eye(9) + G.Q.toarray()) / 2, t)  # default laziness 1/2
    expected = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            expected[i, j] = np.sum((Qt[i] - Qt[j]) ** 2 / G.pi)
    C = ground_cost(G, "diffusion", t=t, normalize=False)
    np.testing.assert_allclose(C, expected)


def test_diffusion_cost_triangle_is_symmetric_across_pairs():
    C = ground_cost(_graph(TRIANGLE), "diffusion")
    off = C[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off, off[0])


def test_ground_cost_disconnected_raises():
    with pytest.raises(ValueError, match="disconnected"):
        ground_cost(_graph(DISCONNECTED), "shortest_path")


def test_ground_cost_bad_rule_raises():
    with pytest.raises(ValueError, match="rule"):
        ground_cost(_graph(TRIANGLE), "euclidean")


PATH3 = [(0, 1), (1, 2)]
CYCLE4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
STAR = [(0, 1), (0, 2), (0, 3)]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("edges", [PATH3, CYCLE4, STAR], ids=["path3", "cycle4", "star"])
def test_lazy_diffusion_cost_separates_nodes_with_the_same_neighbourhood(edges):
    G = _graph(edges)
    for t in (1, 2, 5):
        C = ground_cost(G, "diffusion", t=t)
        assert np.all(C[~np.eye(G.n, dtype=bool)] > 0)


@pytest.mark.parametrize("edges", [PATH3, CYCLE4, STAR], ids=["path3", "cycle4", "star"])
def test_plain_walk_diffusion_cost_warns_when_it_collapses_nodes(edges):
    with pytest.warns(UserWarning, match="zero between"):
        C = ground_cost(_graph(edges), "diffusion", laziness=0.0)
    assert C[1 if edges is STAR else 0, 3 if edges is STAR else 2] == 0.0


def test_laziness_zero_is_the_plain_walk():
    G = _grid(3)
    Qt = np.linalg.matrix_power(G.Q.toarray(), 3)
    expected = (((Qt[:, None, :] - Qt[None, :, :]) ** 2) / G.pi).sum(axis=-1)
    np.testing.assert_allclose(ground_cost(G, "diffusion", t=3, laziness=0.0, normalize=False), expected)


def test_identically_zero_diffusion_cost_raises():
    from graphtransport import markov_chain_from_weight_matrix

    G = MarkovGraph(*markov_chain_from_weight_matrix(np.ones((4, 4))))  # Q = J/4, rank one
    with pytest.raises(ValueError, match="identically zero"):
        ground_cost(G, "diffusion", laziness=0.0)
    with pytest.raises(ValueError, match="identically zero"):
        ground_cost(_graph([(0, 1)]), "diffusion")  # two nodes: lambda = -1 -> 0 at laziness 1/2
    assert ground_cost(_graph([(0, 1)]), "diffusion", laziness=0.75)[0, 1] == pytest.approx(1.0)


def test_diffusion_cost_with_explicit_t_rejects_disconnected_graph():
    with pytest.raises(ValueError, match="disconnected"):
        ground_cost(_graph(DISCONNECTED), "diffusion", t=2)


@pytest.mark.parametrize("t", [0, -1, 2.0, True])
def test_diffusion_cost_rejects_bad_t(t):
    with pytest.raises(ValueError, match="integer >= 1"):
        ground_cost(_grid(3), "diffusion", t=t)


@pytest.mark.parametrize("laziness", [-0.1, 1.0])
def test_diffusion_cost_rejects_bad_laziness(laziness):
    with pytest.raises(ValueError, match="laziness"):
        ground_cost(_grid(3), "diffusion", laziness=laziness)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("rule", ["shortest_path", "diffusion"])
def test_single_node_graph_has_zero_cost(rule):
    G = MarkovGraph(np.array([[1.0]]), np.array([1.0]))
    np.testing.assert_array_equal(ground_cost(G, rule, t=1), [[0.0]])


def test_default_rule_is_shortest_path():
    G = _grid(3)
    np.testing.assert_array_equal(ground_cost(G), ground_cost(G, "shortest_path"))
