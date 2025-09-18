import numpy as np
from solver import AdaptiveSolver
from baseline import FixedMixedBaseline
from games import rock_paper_scissors


def test_solver_policy_is_distribution():
    s = AdaptiveSolver(3)
    p = s.policy()
    assert p.shape == (3,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_nash_value_rps():
    A = rock_paper_scissors()
    b = FixedMixedBaseline(3, A)
    assert abs(b.value) < 1e-6
    assert abs(b.strategy.sum() - 1.0) < 1e-9
