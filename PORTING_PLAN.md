# Porting plan

Re-derived from a direct read of `GraphTransportation.jl` as of commit `c4f28a2`
(2026-09-17), cross-checked against `test/runtests.jl`'s test sets. The Julia
package has grown substantially since Step 2 of this port landed: it now has
a full admissible-means type hierarchy threaded through every backend, and a
new shooting/Hamiltonian geodesic method. This plan supersedes the informal
"Step 3: ground_cost, Sinkhorn" note from earlier.

**Revision (2026-09-17):** reordered so SOCP lands before shooting (shooting
needs *something* to validate against besides the two-node closed form --
without SOCP first we'd be building shooting with a weaker test target than
the Julia suite actually uses). The autodiff-native Sinkhorn layer is now two
steps (torch and jax), both pluggable behind one interface, rather than a
single either/or choice.

**Rule for every step below:** small commit, its own tests, cross-checked
against a fixed numeric example against the Julia original where practical,
reviewed via PR before the next step starts. Same discipline as Steps 1-2.

## Already done (Steps 1-2, need rework -- see Phase 1)

- Package skeleton.
- `chains.py`: Markov chain constructors. **Still valid, no changes needed.**
- `graph.py`: `MarkovGraph`. **Needs rework** -- the Julia struct now carries
  a `mean::AdmissibleMean` field (see Phase 1).
- `means.py`: `geomean`, `logmean` + partials as bare functions. **Superseded**
  -- Julia replaced this with a proper type hierarchy (`core/Means.jl`) with
  different near-diagonal handling (a 5th-order Taylor series with a correct
  derivative, not the old "arithmetic mean" approximation) and no
  negative-argument branch (the new types are only ever evaluated on
  `(0,∞)²`). Phase 1 replaces this file's contents; it's a breaking change to
  the current public API, which is fine since nothing downstream depends on
  it yet.

## Phase 1 -- Admissible means + MarkovGraph rework

Source: `src/core/Means.jl`, `src/core/MarkovGraph.jl`. Test target:
`@testset "Admissible means (types, derivatives, admissibility)"`.

1. **`means.py` rewrite: the `AdmissibleMean` hierarchy.** An abstract base
   (Python: `abc.ABC` or a `Protocol`) with `__call__(s, t)` and
   `partial_s(s, t)` / `partial_t(s, t)` (the latter derived generically as
   `partial_s(t, s)`, exactly like the Julia default). Concrete classes:
   `GeometricMean`, `ArithmeticMean`, `HarmonicMean`, `LogarithmicMean`
   (closed form away from the diagonal, 5th-order series within `|s/t - 1| <
   1e-3`, both value and derivative), `QuadLogMean(K=8)` (Gauss-Legendre
   quadrature approximation of the logarithmic mean). One deliberate
   implementation change: use `numpy.polynomial.legendre.leggauss(K)` for
   the quadrature nodes/weights instead of hand-rolling the Golub-Welsch
   eigendecomposition Julia uses -- same mathematical object, standard
   library instead of reimplementing numerical linear algebra we don't need
   to own.
   Test: value + finite-difference-checked derivative for each mean; the
   ordering `harmonic <= geometric <= logarithmic <= arithmetic` on random
   `(s,t)` pairs; `LogarithmicMean`'s series branch agrees with the closed
   form near the switch point; `QuadLogMean(K)` converges to `LogarithmicMean`
   as `K` grows (check `K=4,6,8,12` against the Julia docstring's own error
   figures: `8.5e-4, 6e-7, 1.2e-10, 2e-15`).

2. **`graph.py` rework: `MarkovGraph.mean`.** Add a `mean: AdmissibleMean`
   field, defaulting to `GeometricMean()`; a `with_mean(new_mean)` method
   returning a new `MarkovGraph` sharing the cached `Q`/`D`/`kappa` (mirrors
   Julia's `MarkovGraph(G; mean)`); update `metric_tensor` to default to
   `G.mean` rather than requiring an explicit mean argument.
   Test: default mean is geometric; `with_mean` shares underlying arrays
   (no recompute) but has independent `.mean`; `metric_tensor` uses `G.mean`
   by default and accepts an override.

## Phase 2 -- Sinkhorn / entropic OT (highest priority: no heavy solver deps, autodiff-friendly, most directly useful for ML)

Source: `src/sinkhorn/Sinkhorn.jl`. Test targets: `@testset "Sinkhorn
barycentric coordinates..."`, `@testset "barycenter(method=:sinkhorn) and
ground_cost"`.

3. **`sinkhorn/cost.py`: `graph_diameter`, `ground_cost`.** Pure BFS (no
   external deps beyond numpy) for `graph_diameter` and the
   `:shortest_path` cost rule; matrix-power diffusion distance for
   `:diffusion`. Both raise on a disconnected graph, both normalize to
   `[0,1]` by default.
   Test: diameter of known small graphs (triangle=1, 3x3 grid=4); shortest-
   path cost is symmetric with zero diagonal; disconnected graph raises.

4. **`sinkhorn/core.py`: forward Sinkhorn barycenter.** `regularize_cost`,
   the forward half of `sinkhorn_differentiate` (drop the `target`/gradient
   branch for this commit), `sinkhorn_barycenter`, `build_geodesic`.
   Test: barycenter of identical measures reproduces that measure at small
   epsilon; barycenter is a valid probability vector; permutation-
   equivariance; marginal error against the two-marginal Sinkhorn plan
   (`_sinkhorn_plan`, port alongside as a private helper) is below solver
   tolerance.

5. **`sinkhorn/core.py` extension: the hand-derived backward pass.**
   `sinkhorn_differentiate`'s gradient branch (Algorithm 1 of Bonneel, Peyré
   & Cuturi 2016), `barycentric_loss`, `loss_gradient` (softmax-Jacobian
   chain rule through `logarithmic_change_of_variable`), `simplex_regression`
   (L-BFGS -- `scipy.optimize.minimize(method="L-BFGS-B")` with the analytic
   `jac=`, matching Julia's `Optim.LBFGS()`).
   Test: `loss_gradient` matches central finite differences (same check the
   Julia suite does); `sinkhorn_differentiate`'s `w` matches finite
   differences in `lambda` directly (not just through the softmax), at both
   a small and large iteration budget `L`; `simplex_regression` recovers
   known synthesis weights on a random small graph.

6. **`sinkhorn/backends/torch_backend.py`: autograd-native variant (torch).**
   A thin `torch` re-expression of the *forward* Sinkhorn barycenter --
   once you have autograd, the hand-derived backward pass from Step 5 is
   redundant for anything built with that framework. `torch` is an optional
   dependency (`pip install graphtransport[torch]`); importing this module
   without `torch` installed raises a clear `ImportError` with an install
   hint, not a bare `ModuleNotFoundError`.
   Test: forward output matches the numpy version to numerical tolerance;
   `torch.autograd.gradcheck` passes.

7. **`sinkhorn/backends/jax_backend.py`: autograd-native variant (jax).**
   Same forward math, re-expressed for `jax` (also optional,
   `pip install graphtransport[jax]`). Both backends implement the same
   small interface (e.g. a `sinkhorn_barycenter(coords, measures, cost,
   epsilon, iters)` free function per module) so callers pick a backend by
   which module they import, not by an internal branch -- keeps both
   optional dependencies genuinely optional and avoids a runtime "which
   framework is installed" dispatch layer neither of us needs yet. Revisit
   this if a real caller wants framework-agnostic code.
   Test: forward output matches the numpy version; `jax.test_util.check_grads`
   passes; a shared test module parametrized over both backends (skipped if
   that backend isn't installed) checks the two agree with each other.

8. **API dispatch: `geodesic`/`barycenter`/`analysis` with `method="sinkhorn"`.**
   Port the unified-entry-point wrappers from `API.jl` (`_geodesic_sinkhorn`,
   the `:sinkhorn` branches of `barycenter`/`analysis`) as the first method
   registered in a dispatcher. Establishes the multi-method API shape
   (`method=` keyword, `GeodesicSolution`-equivalent return type) that later
   phases (shooting, SOCP) plug additional methods into.
   Test: mirrors `@testset "unified API..."`'s sinkhorn-specific assertions.

## Phase 3 -- Predefined graphs (cheap, dataset-adjacent utility)

Source: `src/core/CommonGraphs.jl`.

9. **`graphs.py`: constructors.** `triangle`, `triangle_with_tail`, `square`,
   `t_graph`, `double_t`, `cube`, `triangular_prism`, `hypercube`,
   `weighted_hypercube`, `grid(n)`. All pure edge-list constructions -- no
   new numerical machinery, just data. Skip `ma_house` (needs the bundled
   shapefile + `LibGEOS`/`Shapefile`; Python equivalent would be
   `geopandas`/`shapely` -- worth doing only if you actually want the MA
   redistricting demo ported, since it needs the shapefile data file too).
   Test: node/edge counts match each graph's known structure; `grid(n)` for
   small `n` matches a hand-checked adjacency.

## Phase 4 -- SOCP (moved ahead of shooting: gives shooting a real validation target, not just the two-node closed form)

Source: `src/socp/*.jl`, plus the shared Gram-QP core in `src/core/Analysis.jl`
(needed here first, then reused by shooting's `analyze_shooting` in Phase 5).
Dependency: `cvxpy`, with `clarabel` as the conic backend (matches what the
Julia side now uses) and `SCS` as a fallback (bundled with `cvxpy` by
default, so it's a good "just works" path if `clarabel`'s Python package
has friction). Test targets: `@testset "geodesic_socp vs two-node closed
form"`, `@testset "SOCP with each admissible mean"`, `@testset
"barycenter_socp..."`, `@testset "analyze_socp..."`.

10. **`analysis.py`: shared Gram-matrix simplex QP.** `potential_gram_qp`
    and its underlying simplex-constrained QP solve
    (`solve_barycentric_coordinates_qp`). Small (`p x p`, `p` = number of
    references, typically <= 5) -- now that `cvxpy` is a dependency anyway
    (Step 11), solve it there too rather than mixing in a second QP
    approach, matching the Julia original's use of Convex.jl/SCS for this
    exact piece.
    Test: mirrors `@testset "potential_gram_qp: Gram matrix is the
    Riemannian inner product at the target"`.

11. **`socp/geodesic.py`: the cone program.** `geodesic_socp` via `cvxpy`.
    Each mean's `_mean_cone!` needs translating to a `cvxpy` constraint;
    `QuadLogMean`'s terms are power-cone constraints, which `cvxpy` supports
    natively via `cp.PowCone3D`.
    Test: mirrors `@testset "geodesic_socp vs two-node closed form"` and
    `@testset "SOCP with each admissible mean"` (per-mean two-node closed
    form, distance ordering `harmonic >= geometric >= logarithmic >=
    arithmetic`).

12. **`socp/barycenter.py`, `socp/analysis.py`.** `barycenter_socp`
    (joint SOCP over all references), `analyze_socp` (reuses Step 10's
    Gram-QP core with SOCP-sourced tangent vectors).
    Test: mirrors `@testset "barycenter_socp..."` and `@testset
    "analyze_socp: recovers barycentric coordinates"`.

13. **API dispatch extension: `method="socp"`.** Wire `geodesic_socp`/
    `barycenter_socp`/`analyze_socp` into the Phase 2 Step 8 dispatcher.

## Phase 5 -- Shooting / Hamiltonian (self-contained, every mean, no solver dependency; now validated against Phase 4's SOCP)

Source: `src/shooting/Hamiltonian.jl`, `src/shooting/ExpLog.jl`. Test
targets: `@testset "Hamiltonian shooting..."`, `@testset "log_map by
shooting"` (its real target: round-trip `W2`/`m0` against `geodesic_socp`,
now available), `@testset "analyze_shooting..."`.

14. **`shooting/hamiltonian.py`: the flow.** `hamiltonian`,
    `hamiltonian_flow`, `integrate_hamiltonian` (RK4 with adaptive interval
    halving on a positivity floor -- port the `PositivityFloorError`
    exception and the halving/retry logic faithfully). Pure numpy ODE
    integration, needs the mean's `partial_s`/`partial_t` from Phase 1.
    Test: mirrors `@testset "Hamiltonian shooting: conservation laws"` and
    the per-mean conservation-law testset.

15. **`shooting/explog.py`: exp/log maps.** `weighted_laplacian`,
    `solve_weighted_laplacian`, `momentum_to_potential`, `exp_map`, `log_map`
    (Newton's method with warm/cold retry), `log_map_mollified` (boundary
    fallback), `analyze_shooting`. The most numerically delicate piece in
    this plan -- Newton convergence and the positivity floor need care.
    Test: mirrors `@testset "log_map by shooting"`'s actual target now that
    Phase 4 exists (round-trip `W2`/`m0` vs. `geodesic_socp`, Newton
    iteration counts) and `@testset "log_map_mollified..."`.

16. **API dispatch extension: `method="shooting"`.** The Riemannian
    gradient-descent barycenter loop from `API.jl`'s `_barycenter_shooting`
    (log-map to every reference, descend, exp-map, halve/double step size
    on rejection/acceptance).
    Test: mirrors the shooting sections of `@testset "unified API..."` and
    `@testset "SOCP with each admissible mean"`'s "barycenter + analysis
    round trip per mean" (both methods, same graph, cross-checked).

## Phase 6 -- Chambolle-Pock: recommend skipping

Source: `src/chambolle_pock/*.jl` (the largest single subsystem: custom
proximal operators, a Newton-based cone projection, a continuity-enforcing
linear system). This is the paper's slow reference implementation,
restricted to `GeometricMean` only, and functionally superseded by SOCP
(exact) and shooting (fast, every mean) for everything the unified API
needs. Recommend not porting it unless you specifically want the reference
algorithm available for pedagogical/validation reasons -- flag if you want
it reconsidered, otherwise it's left out of this plan.

## Out of scope

`src/experiments/*` (one-off research scripts: Massachusetts geodesics,
parameter sweeps, benchmarks) -- these aren't library code and don't belong
in a reusable package. If any specific experiment's *result* is worth a
blog post or a demo, that's a `research-site` content question, not a
porting one.
