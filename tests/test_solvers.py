import pytest

cp = pytest.importorskip("cvxpy")

from graphtransport.solvers import solve_conic  # noqa: E402


class _StubProblem:
    """Stands in for a cvxpy Problem whose solve ends in a given status. The
    real case -- a 16x16 digit barycenter at N=10 -- takes ~18 s to reach it."""

    def __init__(self, status):
        self.status = None
        self._status = status

    def solve(self, **kwargs):
        self.status = self._status


def test_optimal_is_accepted():
    assert solve_conic(_StubProblem(cp.OPTIMAL), "CLARABEL", "stub") == cp.OPTIMAL


def test_optimal_inaccurate_is_rejected_with_its_own_reason():
    # the Julia package accepts ALMOST_OPTIMAL; such an iterate was observed
    # with its objective 35% above the optimum
    with pytest.raises(RuntimeError, match=r"status 'optimal_inaccurate' \(not optimal\): reached only reduced "
                                           r"accuracy.*A different N.*try this"):
        solve_conic(_StubProblem(cp.OPTIMAL_INACCURATE), "CLARABEL", "stub", hint="try this")


def test_optimal_inaccurate_is_returned_with_check_false():
    assert solve_conic(_StubProblem(cp.OPTIMAL_INACCURATE), "CLARABEL", "stub", check=False) == cp.OPTIMAL_INACCURATE


@pytest.mark.parametrize("status", ["infeasible", "unbounded", "user_limit", "solver_error"])
def test_other_statuses_are_not_solutions(status):
    with pytest.raises(RuntimeError, match="the iterate is not a solution"):
        solve_conic(_StubProblem(status), "CLARABEL", "stub")
