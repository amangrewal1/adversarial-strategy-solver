import argparse
import numpy as np
import matplotlib.pyplot as plt

from games import rock_paper_scissors, matching_pennies
from experiments import run_suite


def plot_convergence(results, outfile):
    fig, axes = plt.subplots(2, len(results), figsize=(3.4 * len(results), 6.5),
                             sharex=True)
    for col, r in enumerate(results):
        T = r["adaptive_runs"].shape[1]
        t = np.arange(1, T + 1)
        adaptive_rm = np.cumsum(r["adaptive_runs"], axis=1) / t
        hedge_rm = np.cumsum(r["hedge_runs"], axis=1) / t
        nash_rm = np.cumsum(r["nash_runs"], axis=1) / t
        bayes_rm = np.cumsum(r["bayes_runs"], axis=1) / t

        ax_top = axes[0, col] if len(results) > 1 else axes[0]
        ax_bot = axes[1, col] if len(results) > 1 else axes[1]

        ax_top.plot(t, adaptive_rm.mean(0), label="adaptive", lw=2)
        ax_top.fill_between(t, adaptive_rm.mean(0) - adaptive_rm.std(0),
                            adaptive_rm.mean(0) + adaptive_rm.std(0), alpha=0.2)
        ax_top.plot(t, hedge_rm.mean(0), label="fixed hedge", lw=1.5, ls="--")
        ax_top.plot(t, nash_rm.mean(0), label="fixed nash", lw=1.5, ls=":")
        ax_top.plot(t, bayes_rm.mean(0), label="bayesian", lw=1.5, ls="-.")
        ax_top.set_title(r["opponent"], fontsize=10)
        ax_top.grid(alpha=0.3)

        ax_bot.plot(t, r["regret_rate_adaptive"].mean(0), label="adaptive", lw=2)
        ax_bot.plot(t, r["regret_rate_hedge"].mean(0), label="fixed hedge", lw=1.5, ls="--")
        ax_bot.plot(t, r["regret_rate_nash"].mean(0), label="fixed nash", lw=1.5, ls=":")
        ax_bot.plot(t, r["regret_rate_bayes"].mean(0), label="bayesian", lw=1.5, ls="-.")
        ax_bot.axhline(r["eps"], color="red", lw=0.7, alpha=0.5, label=f"ε={r['eps']}")
        ax_bot.set_xlabel("round")
        ax_bot.grid(alpha=0.3)

    (axes[0, 0] if len(results) > 1 else axes[0]).set_ylabel("running mean payoff")
    (axes[1, 0] if len(results) > 1 else axes[1]).set_ylabel("regret rate R(t)/t")
    (axes[0, -1] if len(results) > 1 else axes[0]).legend(loc="lower right", fontsize=8)
    (axes[1, -1] if len(results) > 1 else axes[1]).legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    print(f"saved {outfile}")


def plot_speedup_bar(results, outfile):
    names = [r["opponent"] for r in results]
    sp_h = [r["speedup_vs_hedge"] for r in results]
    sp_n = [r["speedup_vs_nash"] for r in results]
    sp_b = [r["speedup_vs_bayes"] for r in results]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w, sp_h, w, label="vs fixed hedge")
    ax.bar(x, sp_n, w, label="vs fixed nash")
    ax.bar(x + w, sp_b, w, label="vs bayesian")
    ax.axhline(3.0, ls="--", color="red", alpha=0.5, label="3× target")
    ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("convergence speedup")
    ax.set_title("adaptive solver — speedup per opponent")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    print(f"saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["rps", "pennies"], default="rps")
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    A = rock_paper_scissors() if args.game == "rps" else matching_pennies()
    results = run_suite(A, T=args.T, seeds=args.seeds)
    plot_convergence(results, f"{args.outdir}/convergence_{args.game}.png")
    plot_speedup_bar(results, f"{args.outdir}/speedup_{args.game}.png")


if __name__ == "__main__":
    main()
