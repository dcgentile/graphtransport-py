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
    assert np.all(off_diag > 0)  # t = diameter: no pair at zero distance


def test_diffusion_cost_matches_definition_at_explicit_t():
    G = _grid(3)
    t = 2
    Qt = np.linalg.matrix_power(G.Q.toarray(), t)
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
