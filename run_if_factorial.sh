#!/bin/bash
# Item 3: Isolation Forest factorial. Runs in the background (survives closing the tab),
# saves after every seed, resumes if restarted. Log: results/if_factorial_run.log
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
nohup python -u experiments/if_factorial.py . results/if_factorial_results.json 100 \
  > results/if_factorial_run.log 2>&1 &
echo "started in background (pid $!). Progress: tail -f results/if_factorial_run.log"
