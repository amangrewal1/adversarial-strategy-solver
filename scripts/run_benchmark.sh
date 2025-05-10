#!/usr/bin/env bash
set -euo pipefail

# Reproduce the main RPS benchmark (T=1500, 15 seeds)
python experiments.py --game rps --T 1500 --seeds 15 --save results_rps.json
