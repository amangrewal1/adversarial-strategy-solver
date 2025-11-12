"""FixedHedge weights evolve smoothly"""

import numpy as np
from baseline import FixedHedge


def test_weights_sum_to_one_after_updates():
    h = FixedHedge(3, eta=0.05)
    for _ in range(100):
        h.update(np.array([0.0, 1.0, 0.5]))
        assert abs(h.w.sum() - 1.0) < 1e-9

