"""Conic-solver plumbing shared by the cvxpy-based backends (the analysis
QP and the SOCP geodesic/barycenter programs).

cvxpy and a conic solver are optional dependencies
(`pip install "graphtransport[socp]"`). Clarabel is preferred, matching the
Julia package; SCS (bundled with cvxpy) is the fallback.
"""

from __future__ import annotations

PREFERRED_SOLVERS = ("CLARABEL", "SCS")


def import_cvxpy():
    """cvxpy, or an ImportError that says how to install it."""
    try:
        import cvxpy
    except ImportError as exc:
        raise ImportError(
            "this function needs cvxpy and a conic solver; install them with "
            "`pip install 'graphtransport[socp]'`. The SOCP is the default method of the "
            "unified API, so geodesic/barycenter/analysis need it too; method='sinkhorn' "
            "is a different algorithm that needs no optional dependency."
        ) from exc
    return cvxpy


def cvxpy_available() -> bool:
    try:
        import_cvxpy()
    except ImportError:
        return False
    return True


def default_solver() -> str:
    cp = import_cvxpy()
    installed = cp.installed_solvers()
    for name in PREFERRED_SOLVERS:
        if name in installed:
            return name
    raise RuntimeError(f"no supported conic solver installed (need one of {PREFERRED_SOLVERS})")


def solve_conic(problem, solver, what: str, *, check: bool = True, hint: str = "", **solver_kwargs):
    """Solve a cvxpy problem and, with check=True, raise unless the solver
    reports an optimal solution: an iterate left behind by an iteration limit
    or slow progress is not a solution and need not even have unit mass.

    This is stricter than the Julia package's _check_solved, which also
    accepts ALMOST_OPTIMAL (cvxpy's OPTIMAL_INACCURATE). Such an iterate can be
    far from optimal: a four-reference barycenter of 16x16 digit images at
    N=10 came back optimal_inaccurate with J = 9.31, where N=8 and N=12 reach
    OPTIMAL at J = 6.94 and 6.95, and analysis recovered its weights wrong by
    up to 0.09. check=False still returns it, with its status.

    `hint` is appended to the error; each caller supplies advice that applies
    to its own arguments."""
    cp = import_cvxpy()
    solver = default_solver() if solver is None else solver
    problem.solve(solver=solver, **solver_kwargs)
    status = problem.status
    if check and status != cp.OPTIMAL:
        if status == cp.OPTIMAL_INACCURATE:
            reason = (
                "reached only reduced accuracy; such an iterate can be far from optimal, so it is not accepted. "
                "A different N often solves to optimality."
            )
        else:
            reason = "the iterate is not a solution."
        raise RuntimeError(
            f"{what}: solver {solver} stopped with status {status!r} (not optimal): {reason}"
            f"{' ' + hint if hint else ''}"
        )
    return status
