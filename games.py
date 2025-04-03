import numpy as np


def rock_paper_scissors():
    return np.array([[0, -1, 1],
                     [1, 0, -1],
                     [-1, 1, 0]], dtype=float)


def matching_pennies():
    return np.array([[1, -1],
                     [-1, 1]], dtype=float)


def shapley_game():
    return np.array([[0, -1, 1],
                     [1, 0, -1],
                     [-1, 1, 0]], dtype=float)


def random_zero_sum(n, rng):
    return rng.uniform(-1, 1, size=(n, n))


GAMES = {
    "rps": rock_paper_scissors(),
    "pennies": matching_pennies(),
}
