"""FixedMixedBaseline returns valid distribution"""

import numpy as np
from baseline import FixedMixedBaseline
from games import rock_paper_scissors


def test_nash_strategy_is_distribution():
    A = rock_paper_scissors()
    b = FixedMixedBaseline(3, A)
    s = b.policy()
    assert s.shape == (3,)
    assert abs(s.sum() - 1.0) < 1e-9

