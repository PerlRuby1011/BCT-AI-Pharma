#!/bin/bash
# Runs review items 1-3 in order and logs everything to results/review_run.log
# Usage (from Terminal):  bash "$HOME/Articles Manusript/CODE/BCT-AI-Pharma/run_review_experiments.sh"
cd "$(dirname "$0")" || exit 1
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
if ! command -v python >/dev/null 2>&1; then python() { python3 "$@"; }; fi
mkdir -p results
LOG=results/review_run.log
PAPER="../../02 - Systems Paper (Computers & Industrial Engineering)/C&IE/new_experiments"
{
  echo "=== $(date) | python: $(which python) ==="
  echo "=== STEP 1: tests ==="
  python -m pytest tests/test_product_state_machine.py tests/test_pts.py -q
  echo "=== STEP 2: IF factorial ==="
  python experiments/if_factorial.py . results/if_factorial_results.json 100
  echo "=== STEP 3: full PTS smoke test ==="
  python run_full_pts_experiment.py --smoke-test && {
    echo "=== STEP 4: full PTS, 10 seeds x 2 arms ==="
    python run_full_pts_experiment.py
  }
  echo "=== FINISHED $(date) ==="
} 2>&1 | tee "$LOG"
