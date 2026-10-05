#!/bin/bash
# Round-2 review: tests, then smoke + full chronological replay (background, resumable log).
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
python -m pytest tests/test_product_state_machine.py tests/test_contract_integration.py \
  tests/test_causal_features.py tests/test_pts.py -q || { echo "TESTS FAILED - not starting run"; exit 1; }
nohup bash -c 'python -u run_chronological_replay.py --smoke-test && python -u run_chronological_replay.py; echo "=== FINISHED $(date) ==="' \
  > results/round2_replay_run.log 2>&1 &
echo "tests passed; replay started in background (pid $!). Progress: tail -f results/round2_replay_run.log"
