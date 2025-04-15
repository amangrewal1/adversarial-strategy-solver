import numpy as np
from scipy.optimize import linprog


def solve_maximin(A):
    """Row player's maximin mixed strategy for zero-sum payoff matrix A."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    x = np.clip(res.x[:m], 0, None)
    x /= x.sum()
    return x, float(res.x[-1])


class FixedMixedBaseline:
    """Plays a fixed mixed strategy (Nash maximin) — no adaptation."""
    name = "fixed_nash"

    def __init__(self, n_actions, payoff):
        self.n = n_actions
        self.strategy, self.value = solve_maximin(payoff)

    def policy(self):
        return self.strategy.copy()

    def act(self, rng):
        return int(rng.choice(self.n, p=self.strategy))

    def update(self, losses):
        pass


class FixedHedge:
    """Hedge / MWU with fixed learning rate — no shift detection."""
    name = "fixed_hedge"

    def __init__(self, n_actions, eta=0.05):
        self.n = n_actions
        self.eta = eta
        self.w = np.ones(n_actions) / n_actions

    def policy(self):
        return self.w.copy()

    def act(self, rng):
        return int(rng.choice(self.n, p=self.w))

    def update(self, losses):
        self.w *= np.exp(-self.eta * losses)
        self.w /= self.w.sum()


class BayesianBaseline:
    """Exponential belief over opponent column actions + best-response.

    Maintains p(opp_action) with exponential decay; after a short warmup on
    Nash maximin, best-responds to the belief with small epsilon mixing. A
    purely belief-based adaptive baseline — no regret / drift machinery.
    """
    name = "bayesian"

    def __init__(self, n_actions, payoff, decay=0.88, epsilon=0.03, warmup=5):
        self.n = n_actions
        self.A = np.asarray(payoff, dtype=float)
        self.n_row, self.n_col = self.A.shape
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.warmup = int(warmup)
        self.belief = np.ones(self.n_col) / self.n_col
        self.t = 0
        self._nash = None

    def _nash_strategy(self):
        if self._nash is None:
            self._nash, _ = solve_maximin(self.A)
        return self._nash

    def policy(self):
        if self.t < self.warmup:
            return self._nash_strategy()
        expected = self.A @ self.belief
        br = np.zeros(self.n_row)
        br[int(np.argmax(expected))] = 1.0
        uniform = np.ones(self.n_row) / self.n_row
        return (1.0 - self.epsilon) * br + self.epsilon * uniform

    def act(self, rng):
        return int(rng.choice(self.n_row, p=self.policy()))

    def update(self, losses):
        # experiments.play() passes losses = -A[:, b]. Recover the column
        # action by finding the column of -A closest to the loss vector.
        diffs = np.sum((losses[:, None] + self.A) ** 2, axis=0)
        b = int(np.argmin(diffs))
        self.belief = self.belief * self.decay
        self.belief[b] += (1.0 - self.decay)
        self.belief = self.belief / self.belief.sum()
        self.t += 1
