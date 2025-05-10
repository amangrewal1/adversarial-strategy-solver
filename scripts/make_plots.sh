#!/usr/bin/env bash
set -euo pipefail

# Regenerate convergence and speedup plots from the latest results
python plot_results.py --game rps --T 1500 --seeds 15
