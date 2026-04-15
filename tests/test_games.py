import numpy as np
from games import rock_paper_scissors, matching_pennies


def test_rps_shape():
    A = rock_paper_scissors()
    assert A.shape == (3, 3)
    assert np.allclose(A + A.T, 0)


def test_pennies_shape():
    A = matching_pennies()
    assert A.shape == (2, 2)
    # Matching pennies is a zero-sum game with value 0 on the main diagonal
    # sum is zero (each row sums to 0).
    assert np.allclose(A.sum(axis=1), 0)
