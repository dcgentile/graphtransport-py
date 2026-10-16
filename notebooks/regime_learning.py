"""Learning the regimes of a barycentric coding model from snapshots: the machinery behind
notebooks/regime_learning.ipynb.

The atoms (regimes) rho_1..rho_k are fitted to frames nu_1..nu_T by descent on the mean squared
barycentric defect,

    loss(rho) = mean_t  min_{lam in simplex}  || sum_i lam_i log_{nu_t} rho_i ||^2_{nu_t},

which is zero exactly when every frame is a barycenter of the atoms. The inner minimizer lam_hat is the
BCM's analysis step; by the envelope theorem the gradient may hold it fixed, so the gradient with respect
to each atom flows only through the log maps, which graphtransport differentiates exactly (shooting, by
the implicit function theorem). Atoms are parametrized as rho = ((1 - floor) softmax(theta) + floor pi)
/ pi, positive densities of unit mass.

Frames are independent, so a process pool computes one frame's term per task. The pool forks, which
Linux and macOS support; workers=1 runs everything in the calling process instead.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

import graphtransport as gt
from graphtransport.gram import simplex_qp
from graphtransport.shooting import ShootingError
from graphtransport.shooting.explog import weighted_laplacian

_G: gt.MarkovGraph | None = None


def _init_worker(Q, pi) -> None:
    global _G
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
    _G = gt.MarkovGraph(Q, pi)


def frame_term(args):
    """One frame's loss term at the atoms R: (loss, lam_hat, gradient wrt the atoms (k, n), log-map
    potentials (k, n)), or None if shooting fails on any of its log maps."""
    nu, R, warm = args
    G = _G
    if G is None:
        raise RuntimeError("frame_term runs in a worker started by _init_worker")
    rhos = [torch.tensor(r, requires_grad=True) for r in R]
    vs = []
    try:
        for i, rho in enumerate(rhos):
            sol = gt.geodesic(G, torch.tensor(nu), rho, fallback=False, phi0_init=None if warm is None else warm[i])
            vs.append(-sol.phi0 / 2)  # sol.phi0 is the gradient of W^2, -2 times the log map
    except ShootingError:
        return None
    L = torch.tensor(weighted_laplacian(G, nu).toarray())
    V = torch.stack(vs)
    lam = simplex_qp((V @ L @ V.T).detach().numpy())
    g = torch.tensor(lam) @ V
    loss = g @ L @ g
    loss.backward()
    grads = np.stack([r.grad.numpy() for r in rhos if r.grad is not None])
    return float(loss.detach()), lam, grads, V.detach().numpy()


@dataclass
class Run:
    """A training run's record: one entry per optimizer step (Adam) or accepted iterate (L-BFGS-B)."""

    loss: list = field(default_factory=list)
    atoms: list = field(default_factory=list)  # (k, n) per entry
    lams: list = field(default_factory=list)  # (T, k) per entry; NaN rows where shooting failed
    terms: list = field(default_factory=list)  # (T,) per entry; NaN where shooting failed
    evals: list = field(default_factory=list)  # loss-and-gradient evaluations so far

    def as_arrays(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": np.asarray(getattr(self, k)) for k in ("loss", "atoms", "lams", "terms", "evals")}

    @classmethod
    def from_arrays(cls, data, prefix: str) -> Run:
        return cls(**{k: list(data[f"{prefix}_{k}"]) for k in ("loss", "atoms", "lams", "terms", "evals")})


class Trainer:
    def __init__(self, G: gt.MarkovGraph, frames, init_atoms, *, floor: float = 0.05, workers: int = 15):
        self.G, self.frames, self.floor = G, [np.asarray(f, dtype=float) for f in frames], floor
        self.pi = torch.tensor(G.pi)
        # invert the parametrization at the initial atoms
        p = np.stack([np.asarray(r) * G.pi for r in init_atoms])
        q = np.clip((p - floor * G.pi) / (1 - floor), 1e-6, None)
        self.theta = torch.tensor(np.log(q / q.sum(1, keepdims=True)), requires_grad=True)
        self.warm: list = [None] * len(self.frames)
        self.evals = 0
        self.run = Run()
        if workers > 1:
            self.pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(G.Q.toarray(), G.pi))
            self._map = self.pool.map
        else:
            self.pool = None
            _init_worker(G.Q.toarray(), G.pi)
            self._map = lambda f, xs: list(map(f, xs))

    def atoms(self, theta=None) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        return ((1 - self.floor) * torch.softmax(theta, 1) + self.floor * self.pi) / self.pi

    def evaluate(self, R):
        """Mean loss, mean gradient wrt the atoms (k, n), and the per-frame outputs at atoms R."""
        out = self._map(frame_term, [(f, R, w) for f, w in zip(self.frames, self.warm, strict=True)])
        for t, o in enumerate(out):
            if o is not None:
                self.warm[t] = list(o[3])
        ok = [o for o in out if o is not None]
        self.evals += 1
        return float(np.mean([o[0] for o in ok])), np.mean([o[2] for o in ok], axis=0), out

    def _loss_and_grad(self, theta):
        th = theta.detach().clone().requires_grad_(True)
        rho = self.atoms(th)
        R = rho.detach().numpy()
        loss, grad, out = self.evaluate(R)
        rho.backward(torch.tensor(grad))
        if th.grad is None:
            raise RuntimeError("no gradient reached the logits")
        return loss, th.grad, R, out

    def _record(self, loss, R, out) -> None:
        k = len(R)
        self.run.loss.append(loss)
        self.run.atoms.append(R)
        self.run.lams.append(np.array([o[1] if o is not None else np.full(k, np.nan) for o in out]))
        self.run.terms.append(np.array([o[0] if o is not None else np.nan for o in out]))
        self.run.evals.append(self.evals)

    def adam(self, steps: int, *, lr: float = 0.1, callback=None) -> Run:
        opt = torch.optim.Adam([self.theta], lr=lr)
        for s in range(steps):
            loss, grad, R, out = self._loss_and_grad(self.theta)
            self._record(loss, R, out)  # the atoms this step's gradient was taken at
            opt.zero_grad()
            self.theta.grad = grad
            opt.step()
            if callback:
                callback(s, self)
        return self.run

    def lbfgs(self, iters: int, *, callback=None):
        """scipy's L-BFGS-B on the flattened logits with the exact gradient. One record per accepted
        iterate, never a line-search trial point. Returns scipy's OptimizeResult."""
        from scipy.optimize import minimize

        shape = tuple(self.theta.shape)
        cache = {}

        def fun(x):
            loss, grad, R, out = self._loss_and_grad(torch.tensor(x.reshape(shape)))
            cache[x.tobytes()] = (loss, R, out)
            return loss, grad.numpy().ravel().copy()

        def accept(x):
            self._record(*cache[x.tobytes()])
            with torch.no_grad():
                self.theta.copy_(torch.tensor(x.reshape(shape)))
            if callback:
                callback(len(self.run.loss) - 1, self)

        x0 = self.theta.detach().numpy().ravel().copy()
        fun(x0)
        accept(x0)
        return minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            callback=accept,
            options={"maxiter": iters, "maxcor": 20, "ftol": 1e-14, "gtol": 1e-10},
        )

    def loss_at(self, R) -> float:
        return self.evaluate(np.asarray(R))[0]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _socp_distance(pair):
    G, a, b = pair
    return gt.transport_cost(G, a, b, method="socp", N=20)


def recovery(G: gt.MarkovGraph, atoms_by_step, planted, *, workers: int = 15):
    """W(atom_i, matched planted regime) / D at every step, where D is the smallest distance between two
    planted regimes and the matching is the permutation that fits the last step best. Returns
    (errors (steps, k), perm, D)."""
    k = len(planted)
    with mp.get_context("fork").Pool(workers) as pool:
        D = min(pool.map(_socp_distance, [(G, planted[i], planted[j]) for i, j in itertools.combinations(range(k), 2)]))
        last = atoms_by_step[-1]
        C = np.array(pool.map(_socp_distance, [(G, a, r) for a in last for r in planted])).reshape(k, k)
        perm = min(itertools.permutations(range(k)), key=lambda p: sum(C[i, p[i]] for i in range(k)))
        errs = pool.map(_socp_distance, [(G, A[i], planted[perm[i]]) for A in atoms_by_step for i in range(k)])
    return np.array(errs).reshape(len(atoms_by_step), k) / D, perm, D


CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def animate(run: Run, planted, errs, perm, D, LAM, side: int, pi, path: str, *, xlabel: str, style: dict) -> None:
    """Write a GIF of a run: the atoms against the planted regimes (outlined), the frames' coordinates in
    the atoms' simplex (red: outside the atoms' hull), the loss and the recovery error, one animation
    frame per record. ``style`` holds the notebook's colors and colormap."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ink, ink2, muted, surface, alert, cmap = (style[k] for k in ("ink", "ink2", "muted", "surface", "alert", "mass"))
    k, steps = len(planted), len(run.loss)
    pure = [t for t in range(len(LAM)) if LAM[t].max() > 0.999]
    mixed = [t for t in range(len(LAM)) if t not in pure]
    vmax = max((r * pi).max() for r in planted) * 1.15
    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32, left=0.05, right=0.97, top=0.88, bottom=0.09)
    title = fig.suptitle("", x=0.05, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.05, 0.925, "Atoms (blue, mass per node) with the planted regimes outlined", fontsize=9.5, color=ink2)
    images, atom_titles = [], []
    for i in range(k):
        ax = fig.add_subplot(gs[0, i])
        images.append(
            ax.imshow((run.atoms[0][i] * pi).reshape(side, side), cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        )
        target = (planted[perm[i]] * pi).reshape(side, side)
        ax.contour(target, levels=[0.5 * target.max()], colors=ink, linewidths=1.6, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        atom_titles.append(ax.set_title("", fontsize=10.5))
    axs = fig.add_subplot(gs[1, 0])
    axs.add_patch(plt.Polygon(CORNERS, closed=True, fill=False, edgecolor=muted, lw=1))
    for i in range(k):
        c = CORNERS[i]
        axs.text(
            *(c + (c - CORNERS.mean(0)) * 0.14), f"atom {i + 1}", ha="center", va="center", fontsize=9.5, color=ink2
        )
    dots = axs.scatter(np.zeros(len(mixed)), np.zeros(len(mixed)), s=22, color=ink2, edgecolors=surface, lw=0.5)
    pdots = axs.scatter(
        np.zeros(len(pure)), np.zeros(len(pure)), s=70, marker="D", facecolors="none", edgecolors=ink, lw=1.4
    )
    if pure:
        axs.text(
            0.5,
            -0.1,
            "◇ frames that are a planted regime    • mixed frames",
            transform=axs.transAxes,
            ha="center",
            fontsize=8.5,
            color=ink2,
        )
    axs.set_aspect("equal")
    axs.axis("off")
    simplex_title = axs.set_title("", fontsize=10, pad=22)
    axl = fig.add_subplot(gs[1, 1])
    axl.semilogy(run.loss, color=ink, lw=1.8)
    lmark = axl.scatter([0], [run.loss[0]], s=40, color=ink, zorder=3)
    axl.set_xlabel(xlabel)
    axl.set_title("loss: mean squared defect", fontsize=10)
    axr = fig.add_subplot(gs[1, 2])
    rmarks = []
    for i, ls in zip(range(k), ["-", "--", ":"], strict=False):
        axr.plot(errs[:, i], color=ink2, ls=ls, lw=1.6, label=f"atom {i + 1}")
        rmarks.append(axr.scatter([0], [errs[0, i]], s=30, color=ink, zorder=3))
    axr.legend(loc="upper right", fontsize=9, frameon=False)
    axr.set_ylim(0, errs.max() * 1.12)
    axr.set_xlabel(xlabel)
    axr.set_title("distance to the planted regime (W / D)", fontsize=10)

    def update(s):
        title.set_text(f"Learning the regimes: {xlabel} {s} of {steps - 1}")
        for i in range(k):
            images[i].set_data((run.atoms[s][i] * pi).reshape(side, side))
            atom_titles[i].set_text(f"atom {i + 1}   W to planted = {errs[s, i]:.2f} D")
        P = np.asarray(run.lams[s]) @ CORNERS
        outside = np.sqrt(np.asarray(run.terms[s])) / D > 0.01
        dots.set_offsets(P[mixed])
        dots.set_color([alert if outside[t] else ink2 for t in mixed])
        if pure:
            pdots.set_offsets(P[pure])
            pdots.set_edgecolor([alert if outside[t] else ink for t in pure])
        simplex_title.set_text(
            f"frames as barycenters of the atoms\n{outside.sum()} of {len(LAM)} outside their hull (red)"
        )
        lmark.set_offsets([[s, run.loss[s]]])
        for i in range(k):
            rmarks[i].set_offsets([[s, errs[s, i]]])
        return []

    FuncAnimation(fig, update, frames=steps, blit=False).save(path, writer=PillowWriter(fps=12), dpi=72)
    plt.close(fig)
