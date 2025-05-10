# MWU + Drift Detection

The solver combines standard Multiplicative Weights Update (MWU) with a Welch-style
drift detector.

**MWU update.** On each round, play sampled from `w`, observe column `b`, then
`w *= exp(-eta * loss)` where `loss[i] = -A[i, b]`. Normalize.

**Drift test.** Maintain a rolling window of per-round losses. Split into two
halves; compute a pooled-variance Welch statistic between the halves. If the
statistic exceeds `shift_stat`, declare a shift.

**On shift.** Mix current `w` toward uniform by `reset_mix`, boost `eta` by
`boost`, decay back to `eta_base` at rate `decay`. Enter a short cooldown to
prevent repeated detection on the same shift.
