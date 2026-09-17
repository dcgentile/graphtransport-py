# Porting plan

Re-derived from a direct read of `GraphTransportation.jl` as of commit `c4f28a2`
(2026-09-17), cross-checked against `test/runtests.jl`'s test sets. The Julia
package has grown substantially since Step 2 of this port landed: it now has
a full admissible-means type hierarchy threaded through every backend, and a
new shooting/Hamiltonian geodesic method. This plan supersedes the informal
"Step 3: ground_cost, Sinkhorn" note from earlier.

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

6. **`sinkhorn/torch.py` (optional, ML-facing): autograd-native variant.**
   A thin `torch` (or `jax`) re-expression of the *forward* Sinkhorn
   barycenter only -- once you have autograd, the hand-derived backward
   pass from Step 5 is redundant for anything built with that framework.
   This is the piece meant to be dropped into an actual training loop
   (`loss.backward()` works directly) rather than the closed-form gradient,
   which exists mainly for parity/testing and for callers who don't want a
   torch dependency. Needs a decision from you: torch or jax (see "Open
   questions" below).
   Test: forward output matches the numpy version to numerical tolerance;
   `torch.autograd.gradcheck` (or `jax.test_util.check_grads`) passes.

7. **API dispatch: `geodesic`/`barycenter`/`analysis` with `method="sinkhorn"`.**
   Port the unified-entry-point wrappers from `API.jl` (`_geodesic_sinkhorn`,
   the `:sinkhorn` branches of `barycenter`/`analysis`) as the first method
   registered in a dispatcher. Establishes the multi-method API shape
   (`method=` keyword, `GeodesicSolution`-equivalent return type) that later
   phases (shooting, SOCP) plug additional methods into, without committing
   to those now.
   Test: mirrors `@testset "unified API..."`'s sinkhorn-specific assertions.

## Phase 3 -- Predefined graphs (cheap, dataset-adjacent utility)

Source: `src/core/CommonGraphs.jl`.

8. **`graphs.py`: constructors.** `triangle`, `triangle_with_tail`, `square`,
   `t_graph`, `double_t`, `cube`, `triangular_prism`, `hypercube`,
   `weighted_hypercube`, `grid(n)`. All pure edge-list constructions -- no
   new numerical machinery, just data. Skip `ma_house` (needs the bundled
   shapefile + `LibGEOS`/`Shapefile`; Python equivalent would be
   `geopandas`/`shapely` -- worth doing only if you actually want the MA
   redistricting demo ported, since it needs the shapefile data file too).
   Test: node/edge counts match each graph's known structure; `grid(n)` for
   small `n` matches a hand-checked adjacency.

## Phase 4 -- Shooting / Hamiltonian (self-contained, every mean, no external solver -- second-highest ML relevance after Sinkhorn)

Source: `src/shooting/Hamiltonian.jl`, `src/shooting/ExpLog.jl`, the shared
Gram-QP core in `src/core/Analysis.jl`. Test targets: `@testset "Hamiltonian
shooting..."`, `@testset "log_map by shooting"`, `@testset
"analyze_shooting..."`.

9. **`analysis.py`: shared Gram-matrix simplex QP.** `potential_gram_qp` and
   its underlying simplex-constrained QP solve. Julia uses Convex.jl/SCS for
   this, but the QP here is tiny (`p x p`, `p` = number of references,
   typically <= 5) -- `scipy.optimize.minimize(method="SLSQP")` or a simple
   projected-gradient solve is enough; no need for a convex-programming
   dependency for this piece. This is needed by both `analyze_shooting`
   (this phase) and, later, `analyze_socp` (Phase 5) -- porting it once here
   avoids duplicating it when/if Phase 5 happens.
   Test: mirrors `@testset "potential_gram_qp: Gram matrix is the Riemannian
   inner product at the target"`.

10. **`shooting/hamiltonian.py`: the flow.** `hamiltonian`, `hamiltonian_flow`,
    `integrate_hamiltonian` (RK4 with adaptive interval halving on a
    positivity floor -- port the `PositivityFloorError` exception and the
    halving/retry logic faithfully). Pure numpy ODE integration, needs the
    mean's `partial_s`/`partial_t` from Phase 1.
    Test: mirrors `@testset "Hamiltonian shooting: conservation laws"` and
    the per-mean conservation-law testset.

11. **`shooting/explog.py`: exp/log maps.** `weighted_laplacian`,
    `solve_weighted_laplacian`, `momentum_to_potential`, `exp_map`, `log_map`
    (Newton's method with warm/cold retry), `log_map_mollified` (boundary
    fallback), `analyze_shooting`. The most numerically delicate piece in
    this phase -- Newton convergence and the positivity floor need care.
    Test: mirrors `@testset "log_map by shooting"` (round-trip against
    `geodesic_socp`'s `W2`/`m0` -- this test target requires Phase 5 to be
    done first, or falls back to the two-node closed form and shooting-vs-
    shooting consistency checks only) and `@testset "log_map_mollified..."`.

12. **`api.py` extension: `barycenter(method="shooting")`.** The Riemannian
    gradient-descent loop from `API.jl`'s `_barycenter_shooting` (log-map to
    every reference, descend, exp-map, halve/double step size on
    rejection/acceptance).
    Test: mirrors the shooting barycenter section of `@testset "unified
    API..."` and the per-mean barycenter+analysis round trip.

## Phase 5 -- SOCP (optional: exact certificate, but needs a conic solver dependency)

Source: `src/socp/*.jl`. Lower priority than Phases 2-4 for the stated
ML-usability goal -- shooting already covers every admissible mean without a
solver dependency, just without SOCP's global-optimum guarantee. Worth doing
only if you want the exact certificate (e.g. to validate shooting's output,
as the Julia tests do) or need boundary-supported measures (shooting
requires strictly positive densities; SOCP doesn't).

13. **`socp/geodesic.py`: the cone program.** `cvxpy` + a conic backend
    (`clarabel` has a Python package and matches what Julia now uses, or
    `SCS`). Each mean's `_mean_cone!` needs translating to a `cvxpy`
    constraint; `QuadLogMean`'s terms are power-cone constraints, which
    `cvxpy` supports natively via `cp.PowCone3D`.

14. **`socp/barycenter.py`, `socp/analysis.py`.** `barycenter_socp`,
    `analyze_socp` (reuses Phase 4's Gram-QP core).

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

## Open questions for you

- **Phase 2, Step 6 (torch vs. jax):** which framework should the
  autodiff-native Sinkhorn layer target? Torch is the more common choice
  for "drop into someone else's training loop"; jax composes better with
  `vmap`/`jit` if you want to batch over many graphs/weight-settings at
  once (relevant for, e.g., the site's demo computing many barycenters).
- **Phase 4 vs. Phase 5 ordering:** this plan sequences shooting before
  SOCP because it has no solver dependency, but Phase 4 Step 11's test
  target ideally cross-checks against SOCP's `W2`. Confirm you're fine
  validating shooting against the two-node closed form and internal
  consistency only until/unless Phase 5 happens, rather than reordering.
- **Scope confirmation:** confirm Phases 1-4 (means, MarkovGraph rework,
  Sinkhorn, predefined graphs, shooting) as the target scope for now, with
  Phases 5-6 (SOCP, Chambolle-Pock) explicitly deferred/optional. This plan
  is written assuming that's the right cut, given the "useful for ML
  projects" goal, but it's your call.
