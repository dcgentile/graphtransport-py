"""Generate the documentation site's figures into docs/assets/.

    python docs/figures/make_figures.py            # every figure
    python docs/figures/make_figures.py speed      # one (or several) by name

Each figure's data is computed once and cached in docs/figures/.cache (not
committed), so restyling only re-renders; delete the cache to recompute. Every
chart is rendered twice, for the site's light and dark themes (the pages pick
one with Material's #only-light / #only-dark). Colors come from a validated
palette: one blue ramp for densities (magnitude), and three categorical hues
for the three ways of shooting. Needs the "socp" and "docs" extras; about ten
minutes from an empty cache, most of it the digit barycenters.
"""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import graphtransport as gt  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
CACHE = HERE / ".cache"

torch.set_num_threads(1)  # the solver's small torch ops gain nothing from threads
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

# ----- palette -----

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"],
                  density=LinearSegmentedColormap.from_list("density", ["#fcfcfb", *BLUE_RAMP])),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70"],
                 density=LinearSegmentedColormap.from_list("density", ["#1a1a19", *BLUE_RAMP[::-1]])),
}  # fmt: skip


def cached(name, compute):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t0 = time.perf_counter()
    data = compute()
    print(f"  computed {name} in {time.perf_counter() - t0:.0f} s")
    path.write_bytes(pickle.dumps(data))
    return data


def grid(k):
    G = gt.MarkovGraph(*gt.grid_markov_chain(k))
    return G, np.array([(i % k, i // k) for i in range(G.n)], dtype=float)


def density(G, values):
    return values / (values @ G.pi)


def density_axes(ax, image, k, cmap, vmax, surface, smooth=False):
    # one square per node, except where a figure is purely illustrative (the hero):
    # smoothing would invent structure between nodes, and in the methods figure it
    # would look like Sinkhorn's genuine blur
    ax.imshow(image.reshape(k, k), origin="lower", cmap=cmap, vmin=0, vmax=vmax,
              interpolation="bicubic" if smooth else "nearest")  # fmt: skip
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(surface)


def save(fig, name, theme):
    fig.savefig(ASSETS / f"{name}-{theme}.png", dpi=160, facecolor=THEMES[theme]["surface"], bbox_inches="tight")
    plt.close(fig)


# ----- hero: a geodesic, animated -----

K_HERO = 16


def hero_data():
    G, xy = grid(K_HERO)
    blob = density(G, np.exp(-((xy - (3, 3)) ** 2).sum(axis=1) / (2 * 1.6**2)) + 0.02)
    radius = np.sqrt(((xy - (9.5, 9.5)) ** 2).sum(axis=1))
    ring = density(G, np.exp(-((radius - 4.2) ** 2) / (2 * 0.9**2)) + 0.02)
    sol = gt.geodesic(G, blob, ring, fallback=False, maxiters=200)  # shooting, the default
    return dict(rho=sol.rho, W2=sol.W2)


def hero():
    data = cached("hero", hero_data)
    rho = data["rho"]
    frames = list(np.linspace(0, rho.shape[1] - 1, 48).round().astype(int))
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 12  # hold each end
    for theme, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        fig.patch.set_facecolor(t["surface"])
        vmax = float(np.percentile(rho, 99.5))
        density_axes(ax, rho[:, 0], K_HERO, t["density"], vmax, t["surface"], smooth=True)
        image = ax.images[0]
        fig.subplots_adjust(0, 0, 1, 1)

        def draw(j):
            image.set_data(rho[:, j].reshape(K_HERO, K_HERO))
            return (image,)

        FuncAnimation(fig, draw, frames=frames, blit=True).save(
            ASSETS / f"hero-{theme}.gif", writer=PillowWriter(fps=16), savefig_kwargs={"facecolor": t["surface"]}
        )
        plt.close(fig)


# ----- the three methods at time 1/2 -----


def methods_data():
    G, xy = grid(K_HERO)
    rho = cached("hero", hero_data)["rho"]
    blob, ring = rho[:, 0], rho[:, -1]
    socp = gt.geodesic(G, blob, ring, method="socp", N=10)
    sinkhorn = gt.geodesic(G, blob, ring, method="sinkhorn", cost=gt.ground_cost(G), epsilon=0.01, N=10)
    return {"shooting": rho[:, rho.shape[1] // 2], "SOCP (N=10)": socp.rho[:, 5], "Sinkhorn": sinkhorn.rho[:, 5]}


def methods():
    mids = cached("methods", methods_data)
    vmax = max(float(np.percentile(m, 99.5)) for m in mids.values())
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
        fig.patch.set_facecolor(t["surface"])
        for ax, (name, mid) in zip(axes, mids.items()):
            density_axes(ax, mid, K_HERO, t["density"], vmax, t["surface"])
            ax.set_title(name, color=t["ink"], fontsize=10)
        save(fig, "methods", theme)


# ----- digit barycenters -----


def digits_data():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    k = 16
    G, _ = grid(k)

    def glyph(ch):
        fig = Figure(figsize=(1, 1), dpi=k)
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.42, ch, ha="center", va="center", fontsize=80, weight="bold")
        fig.canvas.draw()
        ink = 1 - np.asarray(fig.canvas.buffer_rgba())[..., 0] / 255
        return density(G, ink[::-1].ravel())

    digits = [glyph(c) for c in "0137"]
    square = {}
    for i in range(4):
        for j in range(4):
            s, t = i / 3, j / 3
            lam = np.array([(1 - s) * (1 - t), (1 - s) * t, s * (1 - t), s * t])
            used = lam > 0
            if used.sum() == 1:
                square[i, j] = digits[int(np.argmax(lam))]
            else:
                square[i, j] = gt.barycenter(G, [d for d, u in zip(digits, used) if u], lam[used], method="socp",
                                             N=12)[0]  # fmt: skip
    return square


def digits():
    square = cached("digits", digits_data)
    vmax = max(float(np.percentile(rho, 99.5)) for rho in square.values())
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(4, 4, figsize=(5, 5))
        fig.patch.set_facecolor(t["surface"])
        for (i, j), rho in square.items():
            density_axes(axes[i, j], rho, 16, t["density"], vmax, t["surface"])
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.98, 0.08, 0.08)
        save(fig, "digits", theme)


# ----- multiple shooting: seconds per geodesic -----

SPEED_SIZES = [8, 10, 12, 16]
SPEED_VARIANTS = {"single shooting": 1, "auto (default)": "auto", "8 segments": 8}


def speed_pair(k, kind):
    G, xy = grid(k)
    if kind == "near-uniform":
        rng = np.random.default_rng(0)
        a, b = rng.uniform(0.5, 1.5, G.n), rng.uniform(0.5, 1.5, G.n)
    else:
        a = np.exp(-(xy**2).sum(axis=1) / (2 * (k / 5) ** 2)) + 0.05
        b = a[::-1].copy()
    return G, density(G, a), density(G, b)


def speed_data():
    gt.geodesic(*speed_pair(3, "near-uniform"))  # warm-up
    out = {}
    for kind in ["near-uniform", "corner bumps"]:
        for name, segments in SPEED_VARIANTS.items():
            times = []
            for k in SPEED_SIZES:
                G, a, b = speed_pair(k, kind)
                t0 = time.perf_counter()
                gt.geodesic(G, a, b, fallback=False, maxiters=200, segments=segments)
                times.append(time.perf_counter() - t0)
            out[kind, name] = times
    return out


def speed():
    data = cached("speed", speed_data)
    nodes = [k * k for k in SPEED_SIZES]
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
        fig.patch.set_facecolor(t["surface"])
        for ax, kind in zip(axes, ["near-uniform", "corner bumps"]):
            ax.set_facecolor(t["surface"])
            for color, name in zip(t["series"], SPEED_VARIANTS):
                seconds = data[kind, name]
                ax.plot(nodes, seconds, "-o", color=color, lw=2, ms=6, label=name,
                        markeredgecolor=t["surface"], markeredgewidth=1.5)  # fmt: skip
            ax.set_yscale("log")
            ax.set_title(f"{kind} ({'short' if kind == 'near-uniform' else 'long'} transport)", color=t["ink"],
                         fontsize=10, loc="left")  # fmt: skip
            ax.set_xlabel("graph nodes", color=t["ink2"], fontsize=9)
            ax.grid(True, which="major", axis="y", color=t["grid"], lw=0.8)
            ax.tick_params(colors=t["muted"], labelsize=8, which="both")
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(t["axis"])
            ax.set_xlim(40, 300)
            # direct labels at the right end, nudged apart where lines meet
            ends = sorted((data[kind, name][-1], name, color) for color, name in zip(t["series"], SPEED_VARIANTS))
            placed = []
            for value, name, color in ends:
                y = value
                if placed and y < placed[-1] * 1.35:
                    y = placed[-1] * 1.35
                placed.append(y)
                ax.annotate(name, (nodes[-1], value), xytext=(nodes[-1] + 6, y), color=t["ink2"], fontsize=8,
                            va="center", annotation_clip=False)  # fmt: skip
        axes[0].set_ylabel("seconds per geodesic", color=t["ink2"], fontsize=9)
        legend = axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(t["ink2"])
        fig.tight_layout()
        save(fig, "speed", theme)


# ----- gradient descent on a density -----

GRAD_STEPS = [0, 3, 10, 30]


def gradient_data():
    G, xy = grid(5)
    target = density(G, np.exp(-((xy - (4, 4)) ** 2).sum(axis=1) / 2) + 0.05)
    pi = torch.tensor(G.pi)
    logits = torch.zeros(G.n, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.3)
    snapshots, history = {0: np.ones(G.n)}, []
    for step in range(1, GRAD_STEPS[-1] + 1):
        optimizer.zero_grad()
        W = gt.transport_cost(G, torch.softmax(logits, 0) / pi, torch.tensor(target))
        W.backward()
        optimizer.step()
        history.append(W.item())
        if step in GRAD_STEPS:
            snapshots[step] = (torch.softmax(logits, 0) / pi).detach().numpy()
    return dict(snapshots=snapshots, target=target, history=history)


def gradient():
    data = cached("gradient", gradient_data)
    panels = [(data["snapshots"][s], f"step {s}") for s in GRAD_STEPS] + [(data["target"], "target")]
    vmax = max(float(p.max()) for p, _ in panels)
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(8, 1.9))
        fig.patch.set_facecolor(t["surface"])
        for ax, (image, title) in zip(axes, panels):
            density_axes(ax, image, 5, t["density"], vmax, t["surface"])
            ax.set_title(title, color=t["ink"], fontsize=9)
        save(fig, "gradient", theme)

        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        ax.plot(range(1, len(data["history"]) + 1), data["history"], color=t["series"][0], lw=2)
        ax.set_xlabel("Adam step", color=t["ink2"], fontsize=9)
        ax.set_ylabel("W(ρ, target)", color=t["ink2"], fontsize=9)
        ax.set_ylim(0, None)
        ax.grid(True, axis="y", color=t["grid"], lw=0.8)
        ax.tick_params(colors=t["muted"], labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        fig.tight_layout()
        save(fig, "gradient-curve", theme)


FIGURES = {"hero": hero, "methods": methods, "digits": digits, "speed": speed, "gradient": gradient}

if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for name in sys.argv[1:] or FIGURES:
        print(name)
        FIGURES[name]()
