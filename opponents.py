import numpy as np


class StationaryOpponent:
    name = "stationary"

    def __init__(self, n_actions, rng, strategy=None):
        self.n = n_actions
        self.rng = rng
        self.strategy = strategy if strategy is not None else rng.dirichlet(np.ones(n_actions) * 0.7)

    def act(self):
        return int(self.rng.choice(self.n, p=self.strategy))

    def observe(self, player_action):
        pass


class PeriodicShiftOpponent:
    name = "periodic_shift"

    def __init__(self, n_actions, rng, period=200, n_regimes=3):
        self.n = n_actions
        self.rng = rng
        self.period = period
        self.regimes = [rng.dirichlet(np.ones(n_actions) * 0.3) for _ in range(n_regimes)]
        self.t = 0

    def act(self):
        idx = (self.t // self.period) % len(self.regimes)
        a = int(self.rng.choice(self.n, p=self.regimes[idx]))
        self.t += 1
        return a

    def observe(self, player_action):
        pass


class BestResponseOpponent:
    name = "best_response"

    def __init__(self, n_actions, rng, opp_payoff, noise=0.05):
        self.n = n_actions
        self.rng = rng
        self.payoff = opp_payoff
        self.noise = noise
        self.counts = np.ones(n_actions)

    def act(self):
        p = self.counts / self.counts.sum()
        expected = self.payoff @ p
        if self.rng.random() < self.noise:
            return int(self.rng.integers(self.n))
        return int(np.argmax(expected))

    def observe(self, player_action):
        self.counts[player_action] += 1


class FictitiousPlayOpponent:
    name = "fictitious_play"

    def __init__(self, n_actions, rng, opp_payoff):
        self.n = n_actions
        self.rng = rng
        self.payoff = opp_payoff
        self.counts = np.ones(n_actions)

    def act(self):
        p = self.counts / self.counts.sum()
        expected = self.payoff @ p
        best = np.flatnonzero(expected == expected.max())
        return int(self.rng.choice(best))

    def observe(self, player_action):
        self.counts[player_action] += 1


class AdversarialExploiter:
    name = "adversarial_exploiter"

    def __init__(self, n_actions, rng, opp_payoff, window=40, shift_period=250, noise=0.08):
        self.n = n_actions
        self.rng = rng
        self.payoff = opp_payoff
        self.window = window
        self.shift_period = shift_period
        self.noise = noise
        self.history = []
        self.t = 0

    def act(self):
        self.t += 1
        if self.t % self.shift_period == 0:
            self.history = []
        if len(self.history) < 3:
            return int(self.rng.integers(self.n))
        recent = self.history[-self.window:]
        emp = np.bincount(recent, minlength=self.n).astype(float)
        emp /= emp.sum()
        expected = self.payoff @ emp
        if self.rng.random() < self.noise:
            return int(self.rng.integers(self.n))
        return int(np.argmax(expected))

    def observe(self, player_action):
        self.history.append(int(player_action))


def build_opponent_suite(A, rng_factory):
    n = A.shape[1]
    opp_payoff = -A.T
    return {
        "stationary":           lambda: StationaryOpponent(n, rng_factory()),
        "periodic_shift":       lambda: PeriodicShiftOpponent(n, rng_factory(), period=200, n_regimes=3),
        "best_response":        lambda: BestResponseOpponent(n, rng_factory(), opp_payoff),
        "fictitious_play":      lambda: FictitiousPlayOpponent(n, rng_factory(), opp_payoff),
        "adversarial_exploiter":lambda: AdversarialExploiter(n, rng_factory(), opp_payoff,
                                                             window=40, shift_period=250),
    }
