# Choosing a method

All three methods take the same call and differ in `method=` and its keywords.

- **`"shooting"`** (default) solves Newton's method on the Hamiltonian flow of
  the geodesic equations. It is exact in time, needs no conic solver, works with
  every admissible mean, and is [differentiable](differentiable.md). Every
  density must be strictly positive.
- **`"socp"`** discretises time into `N` steps and solves one second-order cone
  program (cvxpy + Clarabel, `graphtransport[socp]`). It takes densities that are
  zero on part of the graph, and its barycenter is a global optimum, the
  certificate for the others. Its error is \(O(1/N)\).
- **`"sinkhorn"`** is entropically regularised transport for a ground cost you
  pass in (`cost=`, `epsilon=`). It computes a different object, and it blurs.

<figure markdown>
![The three methods at time 1/2](assets/methods-light.png#only-light)
![The three methods at time 1/2](assets/methods-dark.png#only-dark)
<figcaption>The blob-to-ring geodesic at time ½, one square per node. Shooting and
the SOCP compute the same geodesic; Sinkhorn's is the entropic interpolation.</figcaption>
</figure>

## Cost

The SOCP at its default `N=10` was the fastest method on every problem measured,
on n×n grids of 9 to 256 nodes. What shooting offers is exactness in time, no
conic solver, and gradients, not speed. Seconds per geodesic, single shooting
against the SOCP:

| nodes | near-uniform: shooting | SOCP N=10 | corner bumps: shooting | SOCP N=10 |
|---|---|---|---|---|
| 64 | 0.56 | 0.30 | 2.9 | 0.38 |
| 144 | 1.5 | 0.84 | 12 | 1.3 |
| 256 | 6.4 | 1.6 | 71 | 1.9 |

## Multiple shooting

On a long transport, Newton needs many damped steps. `segments=K` splits the time
interval into K pieces and solves for the state at every junction together: the
same discrete problem, in fewer steps, but with twice the work per step. The
default, `segments="auto"`, takes one ordinary step and switches to 8 segments
if the line search had to cut that step to a quarter or less.

<figure markdown>
![Seconds per geodesic: single shooting, auto and 8 segments](assets/speed-light.png#only-light)
![Seconds per geodesic: single shooting, auto and 8 segments](assets/speed-dark.png#only-dark)
<figcaption>Auto matches single shooting on short transports and multiple shooting
on long ones. Pass <code>segments=1</code> for single shooting throughout.</figcaption>
</figure>

## Densities that touch zero

Shooting divides by the density. Given a density that is zero somewhere, the
default method warns (`ShootingFallbackWarning`) and returns the SOCP's answer;
`fallback=False` raises instead. Near zero the flow gets stiff: on a 5×5 grid a
row of nodes at 1e-4 solves by shooting and one at 1e-5 does not. For such data,
pass `method="socp"`.

## The mean

The metric weights each edge by a mean \(\theta(\rho_x, \rho_y)\) of the densities
at its ends: `GeometricMean()` by default, or `ArithmeticMean`, `HarmonicMean`,
`LogarithmicMean`, `QuadLogMean`. The mean belongs to the graph,
`MarkovGraph(Q, pi, mean=...)` or `G.with_mean(...)`, so every computation on a
graph uses the same metric.
