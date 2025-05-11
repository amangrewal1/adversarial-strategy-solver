# Reproducibility

The benchmark numbers in the README are reproduced by:

```bash
python experiments.py --game rps --T 1500 --seeds 15 --save results_rps.json
python plot_results.py --game rps --T 1500 --seeds 15
```

**Seeds.** The experiment harness uses master seed `0` to generate per-run
RNGs in a deterministic sequence. Each of the four players gets a distinct
seed offset so that opponent streams are identical across players within a
seed.

**Hardware.** Results do not depend on hardware — the simulation is CPU-only
numpy. Runtime on a modern laptop is under 60 seconds for the full
T=1500, seeds=15 sweep.

**Environment.** numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7.
