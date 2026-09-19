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
            "`pip install 'graphtransport[socp]'`"
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
    reports an optimal (or inaccurate-optimal) solution, mirroring the Julia
    package's _check_solved: an iterate left behind by an iteration limit or
    slow progress is not a solution and need not even have unit mass.

    `hint` is appended to the error; each caller supplies advice that applies
    to its own arguments."""
    cp = import_cvxpy()
    solver = default_solver() if solver is None else solver
    problem.solve(solver=solver, **solver_kwargs)
    status = problem.status
    if check and status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"{what}: solver {solver} stopped with status {status!r} (not optimal); the iterate is "
            f"not a solution.{' ' + hint if hint else ''}"
        )
    return status
