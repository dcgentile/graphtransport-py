"""MarkovGraph: the compact edge-based graph representation used throughout
GraphTransportation.jl (ported from core/MarkovGraph.jl).

Stores an undirected graph together with a reversible Markov chain on it,
using a per-edge (rather than per-node-pair-matrix) representation: momenta,
potential gradients, and other edge quantities are each a single length-|E|
array, with an arbitrary but fixed orientation per edge. Antisymmetry of edge
fields is therefore hard-coded into the representation, following the
convention of Erbar, Rumpf, Schmitzer & Simon.

Node and edge indices are 0-indexed, unlike the Julia original.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from graphtransport.means import AdmissibleMean, GeometricMean


def _check_mean(mean) -> AdmissibleMean:
    if not isinstance(mean, AdmissibleMean):
        hint = f"; did you mean {mean.__name__}()?" if isinstance(mean, type) and issubclass(mean, AdmissibleMean) else ""
        raise TypeError(f"mean must be an AdmissibleMean instance, got {mean!r}{hint}")
    return mean


class MarkovGraph:
    """Graph primitive: an undirected graph plus a reversible Markov chain on it.

    Attributes:
        n: number of vertices.
        E: (|E|, 2) int array of oriented edges; E[e] = (x, y) means the edge
            is oriented from x to y (the orientation is the one with x < y).
        pi: (n,) stationary distribution.
        Q: (n, n) sparse transition rate matrix.
        kappa: (|E|,) edge weights kappa[e] = Q[x, y] * pi[x], which by
            reversibility equals Q[y, x] * pi[y].
        D: (n, |E|) sparse incidence matrix with D @ m == graph_divergence(G, m)
            for any edge field m. D[x, e] = -Q[x, y], D[y, e] = Q[y, x] for
            edge e = (x, y).
        mean: the mobility theta(s, t) of the transport metric on this graph
            (GeometricMean() by default). The mean is part of the geometry,
            like the graph itself, so it lives here rather than as a per-call
            option: every geodesic, barycenter and analysis computed from the
            same MarkovGraph uses the same metric, and a target synthesized on
            one graph is analyzed with the mean it was made with.
    """

    def __init__(self, Q, pi, *, rtol: float = 1e-12, mean: AdmissibleMean | None = None):
        self.mean: AdmissibleMean = GeometricMean() if mean is None else _check_mean(mean)
        Q_dense = np.asarray(Q, dtype=float)
        pi = np.asarray(pi, dtype=float)
        if Q_dense.ndim != 2 or Q_dense.shape[0] != Q_dense.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q_dense.shape}")
        n = Q_dense.shape[0]
        Q = sparse.csr_matrix(Q_dense)
        if pi.shape[0] != n:
            raise ValueError(f"pi has length {pi.shape[0]}, expected {n}")

        # Edge order matches Julia's findnz on a SparseMatrixCSC: column-major,
        # i.e. by target node then source, so per-edge quantities (kappa, m,
        # theta) line up with the Julia package's without a permutation.
        Qcoo = Q.tocsc().tocoo()
        edges: list[tuple[int, int]] = []
        kappas: list[float] = []
        for i, j, q_ij in sorted(zip(Qcoo.row, Qcoo.col, Qcoo.data), key=lambda t: (t[1], t[0])):
            if i >= j:
                continue
            q_ji = Q[j, i]
            kappa_ij = q_ij * pi[i]
            kappa_ji = q_ji * pi[j]
            scale = max(abs(kappa_ij), abs(kappa_ji), 1e-300)
            if abs(kappa_ij - kappa_ji) / scale > rtol:
                raise ValueError(
                    f"reversibility violated on edge ({i},{j}): "
                    f"Q[i,j]*pi[i]={kappa_ij}, Q[j,i]*pi[j]={kappa_ji}"
                )
            edges.append((i, j))
            kappas.append(kappa_ij)

        self.n = n
        self.E = np.array(edges, dtype=int).reshape(-1, 2)
        self.pi = pi
        self.Q = Q
        self.kappa = np.array(kappas, dtype=float)

        num_edges = len(edges)
        D_rows = np.empty(2 * num_edges, dtype=int)
        D_cols = np.empty(2 * num_edges, dtype=int)
        D_vals = np.empty(2 * num_edges, dtype=float)
        for e, (x, y) in enumerate(edges):
            D_rows[2 * e], D_cols[2 * e], D_vals[2 * e] = x, e, -Q[x, y]
            D_rows[2 * e + 1], D_cols[2 * e + 1], D_vals[2 * e + 1] = y, e, Q[y, x]
        self.D = sparse.csr_matrix((D_vals, (D_rows, D_cols)), shape=(n, num_edges))

    def with_mean(self, mean: AdmissibleMean) -> "MarkovGraph":
        """The same graph with a different mean; shares the cached matrices."""
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.mean = _check_mean(mean)
        return other

    def __repr__(self) -> str:
        return f"MarkovGraph(n={self.n}, |E|={self.E.shape[0]}, mean={self.mean!r})"


def graph_gradient(G: MarkovGraph, phi) -> np.ndarray:
    """(grad phi)[e] = phi[x] - phi[y] for the oriented edge e = (x, y)."""
    phi = np.asarray(phi, dtype=float)
    if phi.shape[0] != G.n:
        raise ValueError(f"phi has length {phi.shape[0]}, expected {G.n}")
    return phi[G.E[:, 0]] - phi[G.E[:, 1]]


def graph_divergence(G: MarkovGraph, m) -> np.ndarray:
    """Adjoint of graph_gradient with respect to <.,.>_pi and <.,.>_Q.

    Satisfies <phi, div m>_pi == -<grad phi, m>_Q for any node field phi and
    edge field m, where <m, w>_Q := sum_e kappa[e] * m[e] * w[e].
    """
    m = np.asarray(m, dtype=float)
    if m.shape[0] != G.E.shape[0]:
        raise ValueError(f"m has length {m.shape[0]}, expected {G.E.shape[0]}")
    return G.D @ m


def dense_metric_tensor(rho, mean) -> np.ndarray:
    """The n x n metric tensor of GraphTransportation.jl's dense graph calculus:
    entry (i, j) is mean(rho[i], rho[j]) for i != j and zero on the diagonal.
    Used with dense antisymmetric edge fields (the momentum convention of the
    SOCP analysis); the per-edge `metric_tensor` is the compact counterpart."""
    rho = np.asarray(rho, dtype=float)
    g = mean(rho[:, np.newaxis], rho[np.newaxis, :])
    g = np.array(g, dtype=float)
    np.fill_diagonal(g, 0.0)
    return g


def metric_tensor(G: MarkovGraph, rho, mean=None) -> np.ndarray:
    """theta[e] = mean(rho[x], rho[y]) for the oriented edge e = (x, y).

    `mean` defaults to the graph's own G.mean; any callable (s, t) -> theta
    can be passed explicitly. Does not include the edge weight kappa; the
    Riemannian inner product of two potential gradients at rho is
    <grad phi, grad psi>_rho = sum_e kappa[e] * theta[e] * (grad phi)[e] * (grad psi)[e].
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape[0] != G.n:
        raise ValueError(f"rho has length {rho.shape[0]}, expected {G.n}")
    if mean is None:
        mean = G.mean
    return mean(rho[G.E[:, 0]], rho[G.E[:, 1]])
