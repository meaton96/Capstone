#!/bin/bash
# train-ft-het (10-25, plan §2b, user decision 10-25: fine-tune targets 100 machines + spread 1.0): the final policy
# (train-v5-cmix s0-s4) resumed on B2 with machine-speed spread s = 1 (capacity-matched, hetero.py) for 24 updates, a
# checkpoint per update. Local, all 5 seeds at once with 3 twin workers each (lab: 20 slots; the final policy trained
# with 2 workers per seed at ~195 s per update on the cluster). Relaunching skips seeds with DONE_s<seed> and resumes
# the others from their latest checkpoint_step*.pt.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$(cat results/train-ft-het/code/target)" == het && "$(cat results/train-ft-het/code/spread)" == "1,1" \
   && "$(cat results/train-ft-het/code/c_range)" == "1.25 3.5" ]] || { echo "code/target, spread or c_range wrong"; exit 1; }
bash results/scripts/ft_train.sh train-ft-het "0 1 2 3 4" 3
