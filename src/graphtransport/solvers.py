"""Conic-solver plumbing shared by the cvxpy-based backends (the analysis
QP and the SOCP geodesic/barycenter programs).

cvxpy and a conic solver are optional dependencies
(`pip install "graphtransport[socp]"`). Clarabel is preferred, matching the
Julia package; SCS (bundled with cvxpy) is the fallback.
"""

from __future__ import annotations

PREFERRED_SOLVERS = ("CLARABEL", "SCS")


def default_solver() -> str:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "this function needs cvxpy and a conic solver; install them with "
            "`pip install 'graphtransport[socp]'`"
        ) from exc
    installed = cp.installed_solvers()
    for name in PREFERRED_SOLVERS:
        if name in installed:
            return name
    raise RuntimeError(f"no supported conic solver installed (need one of {PREFERRED_SOLVERS})")


def solve_conic(problem, solver, what: str, *, check: bool = True, **solver_kwargs):
    """Solve a cvxpy problem and, with check=True, raise unless the solver
    reports an optimal (or inaccurate-optimal) solution, mirroring the Julia
    package's _check_solved: an iterate left behind by an iteration limit or
    slow progress is not a solution and need not even have unit mass."""
    import cvxpy as cp

    solver = default_solver() if solver is None else solver
    problem.solve(solver=solver, **solver_kwargs)
    status = problem.status
    if check and status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"{what}: solver {solver} stopped with status {status!r} (not optimal); the iterate is "
            "not a solution. Try a smaller N (or fewer quadrature nodes for QuadLogMean), pass "
            "solver options to raise the iteration limit, or pass check=False to get the "
            "partial iterate anyway."
        )
    return status
