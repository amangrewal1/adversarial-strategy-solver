import argparse
import json
import numpy as np

from games import rock_paper_scissors, matching_pennies, random_zero_sum
from opponents import build_opponent_suite
from solver import AdaptiveSolver
from baseline import FixedMixedBaseline, FixedHedge, BayesianBaseline


def play(player, opponent, A, T, rng):
    actions_o = np.empty(T, dtype=int)
    payoffs = np.empty(T)
    for t in range(T):
        a = player.act(rng)
        b = opponent.act()
        actions_o[t] = b
        payoffs[t] = A[a, b]
        player.update(-A[:, b])
        opponent.observe(a)
    return payoffs, actions_o


def static_regret(payoffs, opp_actions, A):
    per_action_cum = np.cumsum(A[:, opp_actions], axis=1)
    best_cum = per_action_cum.max(axis=0)
    return best_cum - np.cumsum(payoffs)


def convergence_time(regret_rate, eps, stay=50):
    run = 0
    for t in range(len(regret_rate)):
        run = run + 1 if regret_rate[t] <= eps else 0
        if run >= stay:
            return t
    return len(regret_rate)


def run_opponent(opp_name, opp_factory, A, T, seeds, eps=0.15):
    rows = {"adaptive": [], "hedge": [], "nash": [], "bayes": []}
    regret_rates = {"adaptive": [], "hedge": [], "nash": [], "bayes": []}
    shift_counts = []

    for s in range(seeds):
        for key, player_ctor, rng_seed in [
            ("adaptive", lambda: AdaptiveSolver(A.shape[0]), 1000 + s),
            ("hedge",    lambda: FixedHedge(A.shape[0], eta=0.018), 2000 + s),
            ("nash",     lambda: FixedMixedBaseline(A.shape[0], A), 3000 + s),
            ("bayes",    lambda: BayesianBaseline(A.shape[0], A), 4000 + s),
        ]:
            rng = np.random.default_rng(rng_seed)
            player = player_ctor()
            payoffs, opp_acts = play(player, opp_factory(), A, T, rng)
            reg = static_regret(payoffs, opp_acts, A)
            rows[key].append(payoffs)
            regret_rates[key].append(reg / np.arange(1, T + 1))
            if key == "adaptive":
                shift_counts.append(len(player.shifts))

    conv = {k: float(np.mean([convergence_time(rr, eps) for rr in regret_rates[k]]))
            for k in regret_rates}

    return {
        "opponent": opp_name,
        "eps": eps,
        "mean_adaptive": float(np.mean(rows["adaptive"])),
        "mean_hedge": float(np.mean(rows["hedge"])),
        "mean_nash": float(np.mean(rows["nash"])),
        "mean_bayes": float(np.mean(rows["bayes"])),
        "conv_adaptive": conv["adaptive"],
        "conv_hedge": conv["hedge"],
        "conv_nash": conv["nash"],
        "conv_bayes": conv["bayes"],
        "speedup_vs_hedge": conv["hedge"] / max(conv["adaptive"], 1.0),
        "speedup_vs_nash": conv["nash"] / max(conv["adaptive"], 1.0),
        "speedup_vs_bayes": conv["bayes"] / max(conv["adaptive"], 1.0),
        "avg_shifts_detected": float(np.mean(shift_counts)),
        "adaptive_runs": np.array(rows["adaptive"]),
        "hedge_runs": np.array(rows["hedge"]),
        "nash_runs": np.array(rows["nash"]),
        "bayes_runs": np.array(rows["bayes"]),
        "regret_rate_adaptive": np.array(regret_rates["adaptive"]),
        "regret_rate_hedge": np.array(regret_rates["hedge"]),
        "regret_rate_nash": np.array(regret_rates["nash"]),
        "regret_rate_bayes": np.array(regret_rates["bayes"]),
    }


def run_suite(A, T=1500, seeds=12, eps=0.15):
    results = []
    rng_master = np.random.default_rng(0)
    suite_factories = build_opponent_suite(
        A, lambda: np.random.default_rng(rng_master.integers(1 << 31))
    )
    for name, factory in suite_factories.items():
        results.append(run_opponent(name, factory, A, T, seeds, eps=eps))
    return results


def print_table(results):
    print(f"\n{'Opponent':<22} {'Ad µ':>7} {'Hd µ':>7} {'Nh µ':>7} {'By µ':>7} "
          f"{'T*ad':>7} {'T*hd':>7} {'T*nh':>7} {'T*by':>7} "
          f"{'×hedge':>8} {'×nash':>8} {'×bayes':>8} {'shifts':>7}")
    print("-" * 118)
    for r in results:
        print(f"{r['opponent']:<22} "
              f"{r['mean_adaptive']:>+7.3f} {r['mean_hedge']:>+7.3f} "
              f"{r['mean_nash']:>+7.3f} {r['mean_bayes']:>+7.3f} "
              f"{r['conv_adaptive']:>7.1f} {r['conv_hedge']:>7.1f} "
              f"{r['conv_nash']:>7.1f} {r['conv_bayes']:>7.1f} "
              f"{r['speedup_vs_hedge']:>7.2f}x {r['speedup_vs_nash']:>7.2f}x "
              f"{r['speedup_vs_bayes']:>7.2f}x "
              f"{r['avg_shifts_detected']:>7.1f}")
    avg_h = np.mean([r["speedup_vs_hedge"] for r in results])
    avg_n = np.mean([r["speedup_vs_nash"] for r in results])
    avg_b = np.mean([r["speedup_vs_bayes"] for r in results])
    print("-" * 118)
    print(f"{'AVERAGE':<22} {'':>7} {'':>7} {'':>7} {'':>7} "
          f"{'':>7} {'':>7} {'':>7} {'':>7} "
          f"{avg_h:>7.2f}x {avg_n:>7.2f}x {avg_b:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["rps", "pennies", "random"], default="rps")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    if args.game == "rps":
        A = rock_paper_scissors()
    elif args.game == "pennies":
        A = matching_pennies()
    else:
        A = random_zero_sum(args.n, np.random.default_rng(42))

    results = run_suite(A, T=args.T, seeds=args.seeds, eps=args.eps)
    print_table(results)

    if args.save:
        slim = [{k: v for k, v in r.items()
                 if not isinstance(v, np.ndarray)} for r in results]
        with open(args.save, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
