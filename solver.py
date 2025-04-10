import numpy as np
from collections import deque


class AdaptiveSolver:
    """Multiplicative Weights Update with Welch-style drift detection."""
    name = "adaptive_mwu"

    def __init__(self, n_actions, eta=0.25, window=80, shift_stat=3.0,
                 boost=4.0, decay=0.993, reset_mix=0.45, cooldown=40):
        self.n = n_actions
        self.eta_base = eta
        self.eta = eta
        self.w = np.ones(n_actions) / n_actions
        self.window = window
        self.shift_stat = shift_stat
        self.boost = boost
        self.decay = decay
        self.reset_mix = reset_mix
        self.cooldown = cooldown
        self.loss_hist = deque(maxlen=window)
        self.since_shift = cooldown
        self.shifts = []
        self.t = 0

    def policy(self):
        return self.w.copy()

    def act(self, rng):
        return int(rng.choice(self.n, p=self.w))

    def update(self, losses):
        self.t += 1
        self.w *= np.exp(-self.eta * losses)
        self.w /= self.w.sum()

        cur_loss = float(self.w @ losses)
        self.loss_hist.append(cur_loss)
        self.since_shift += 1

        if (self.since_shift >= self.cooldown
                and len(self.loss_hist) == self.window):
            half = self.window // 2
            arr = np.fromiter(self.loss_hist, dtype=float, count=self.window)
            old, new = arr[:half], arr[half:]
            se = np.sqrt(old.var(ddof=1) / half + new.var(ddof=1) / half) + 1e-6
            stat = (new.mean() - old.mean()) / se
            if stat > self.shift_stat:
                self._on_shift()

        self.eta = max(self.eta_base, self.eta * self.decay)

    def _on_shift(self):
        self.w = (1 - self.reset_mix) * self.w + self.reset_mix * np.ones(self.n) / self.n
        self.w /= self.w.sum()
        self.eta = min(1.0, self.eta_base * self.boost)
        self.loss_hist.clear()
        self.shifts.append(self.t)
        self.since_shift = 0
