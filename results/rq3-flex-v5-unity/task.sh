#!/bin/bash
# rq3-flex-v5-unity (10-17, user: set up RQ3 machine flexibility): one (level, group, seed) Unity evaluation of B2 with
# multi-capability machines. The twin refuses flexibility (Unity draws each machine's secondary capabilities from the
# scenario seed), so this is Unity only.
#   levels  flex-p0.3 / flex-p0.5: --machine-flexibility p (probability that a machine can also run each other type)
#           --secondary-time-multiplier 1.25 (a secondary type runs 25% slower)
#   groups  fa  SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT
#           fb  ATC-PECT, ATC-PTECT, B2-trained pick (train-v5-final s4)
#           fc  MDD-PTECT, loose-trained pick (train-v5-loose s2), final policy = mixed-c pick (train-v5-cmix s3)
#           (fc's MDD-PTECT repeats fa's: a run-to-run check that the flexible floor is the same per seed.)
# Same code / player / settings as ../eval-v5-cmix-unity/task.sh (6d13412a snapshot, linux_server/ 5e6e0f87 without
# -pyrules, B2 params, random warm-up, 21,600 s window, 900 s slots, deterministic, discount horizon 10,800 s), through
# ../rq4-agvfail-v5-rules/eval_wrap.py. Skips a finished task (DONE).
# Usage: bash results/rq3-flex-v5-unity/task.sh LEVEL GROUP SEED WORKER_ID
set -uo pipefail
LV=$1; G=$2; SEED=$3; WID=$4
H=results/rq3-flex-v5-unity
D=$H/runs/$LV/$G/s$SEED
[[ -f $D/DONE ]] && { echo "exists $D"; exit 0; }
CODE=results/rq2-rebase-v5/code/env
PY=${PY:-.venv/bin/python}
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
case $LV in
    flex-p0.3) FLEX=(--machine-flexibility 0.3 --secondary-time-multiplier 1.25) ;;
    flex-p0.5) FLEX=(--machine-flexibility 0.5 --secondary-time-multiplier 1.25) ;;
    *) echo "unknown level $LV"; exit 2 ;;
esac
BASE=results/train-v5-final_s4/checkpoint.pt; LOOSE=results/train-v5-loose_s2/checkpoint.pt
CMIX=results/train-v5-cmix_s3/checkpoint.pt
case $G in
    fa) ARGS=(--pdr SRT-PECT,SRT-PTECT,MDD-PECT,MDD-PTECT); N=4 ;;
    fb) ARGS=(--pdr ATC-PECT,ATC-PTECT --checkpoint $BASE); N=3 ;;
    fc) ARGS=(--pdr MDD-PTECT --checkpoint $LOOSE --checkpoint $CMIX); N=3 ;;
    *) echo "unknown group $G"; exit 2 ;;
esac
for f in $BASE $LOOSE $CMIX; do [[ -f $f ]] || { echo "missing checkpoint $f"; exit 1; }; done
rm -rf "$D"; mkdir -p "$D"
t0=$(date +%s)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY results/rq4-agvfail-v5-rules/eval_wrap.py $CODE \
    --unity-path linux_server/capstone.x86_64 --no-graphics \
    --base-worker-id "$WID" --reward-spec $CODE/config/rewards/tardiness.json --scenario-generator randomized \
    --random-warmup --episode-duration-seconds 21600 --params "$P" "${FLEX[@]}" --seeds "$SEED" --slot-seconds 900 \
    --discount-horizon-s 10800 --device cpu "${ARGS[@]}" --out "$D" > "$D/eval.out" 2>&1
rc=$?
n=$(( $(wc -l < "$D/episodes.csv" 2>/dev/null || echo 1) - 1 ))
if [[ $rc -eq 0 && $n -eq $N ]]; then echo "$(( $(date +%s) - t0 ))" > "$D/DONE"; echo "ok $LV $G s$SEED ($(( $(date +%s) - t0 )) s)";
else echo "FAIL $LV $G s$SEED rc=$rc episodes=$n (see $D/eval.out)"; exit 1; fi
