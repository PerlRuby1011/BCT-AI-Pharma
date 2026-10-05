#!/bin/bash
# OPTION11 unified-enforcement run (pre-registered). Logs to results/option11_run.log
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
{
  echo "=== $(date) | $(which python) ==="
  python run_full_pts_experiment.py --enforcement --smoke-test && \
  python run_full_pts_experiment.py --enforcement
  echo "=== FINISHED $(date) ==="
} 2>&1 | grep --line-buffered -v -e FutureWarning -e "prior_flags = " -e "fillna(False).astype(int)$" | tee results/option11_run.log
