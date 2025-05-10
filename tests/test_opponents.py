import numpy as np
from games import rock_paper_scissors
from opponents import build_opponent_suite


def test_opponent_suite_has_five():
    A = rock_paper_scissors()
    suite = build_opponent_suite(A, lambda: np.random.default_rng(0))
    assert len(suite) == 5


def test_opponents_return_valid_actions():
    A = rock_paper_scissors()
    suite = build_opponent_suite(A, lambda: np.random.default_rng(0))
    for name, factory in suite.items():
        opp = factory()
        a = opp.act()
        assert 0 <= a < A.shape[1], f"{name} returned invalid action {a}"
