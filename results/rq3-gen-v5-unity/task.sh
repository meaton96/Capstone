#!/bin/bash
# rq3-gen-v5-unity (10-14): one (condition, group, seed) of the RQ3 Unity subset: does the twin's RQ3 answer on the
# v5 rules (../rq3-gen-v5) hold in Unity? Conditions (conds.json, built from ../rq3-gen-v5/run_task.py CONDITIONS):
# load-1.2 (busiest floor), due-tight (the best pair changes in the twin), lay-g (deeper floor; the H15 policies
# turned positive there). B2 seeds 0-19, the same instances as the twin sweep.
#   pa  SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT      pb  ATC-PECT, ATC-PTECT, the final PPO policy
#   (the seed ../eval-v5-final chose on the fitting seeds, selected.txt, shipped by submit.sh)
# Same code / player / settings as ../eval-transfer-unity-v5/task.sh (6d13412a snapshot, linux_server/ 5e6e0f87
# without -pyrules, random warm-up, 21,600 s window, 900 s slots, deterministic). Skips a finished task (DONE).
# Usage: bash results/rq3-gen-v5-unity/task.sh COND GROUP SEED WORKER_ID
set -uo pipefail
C=$1; G=$2; SEED=$3; WID=$4
H=results/rq3-gen-v5-unity
D=$H/runs/$C/$G/s$SEED
[[ -f $D/DONE ]] && { echo "exists $D"; exit 0; }
CODE=results/rq2-rebase-v5/code/env
PY=${PY:-.venv/bin/python}
P=$($PY -c "import json,sys; print(json.dumps(json.load(open('$H/conds.json'))['$C']['params']))")
L=$($PY -c "import json,sys; print(json.load(open('$H/conds.json'))['$C']['layout'] or '')")
LAY=(); [[ -n $L ]] && LAY=(--layout "$L")
case $G in
    pa) ARGS=(--pdr SRT-PECT,SRT-PTECT,MDD-PECT,MDD-PTECT); N=4 ;;
    pb) CK=$(cut -d' ' -f2 results/eval-v5-final/selected.txt)
        [[ -f $CK ]] || { echo "no final policy checkpoint ($CK)"; exit 1; }
        ARGS=(--pdr ATC-PECT,ATC-PTECT --checkpoint "$CK"); N=3 ;;
    *) echo "unknown group $G"; exit 2 ;;
esac
rm -rf "$D"; mkdir -p "$D"
t0=$(date +%s)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY $CODE/evaluate.py --unity-path linux_server/capstone.x86_64 --no-graphics \
    --base-worker-id "$WID" --reward-spec $CODE/config/rewards/tardiness.json --scenario-generator randomized \
    --random-warmup --episode-duration-seconds 21600 --params "$P" "${LAY[@]}" --seeds "$SEED" --slot-seconds 900 \
    --discount-horizon-s 10800 --device cpu "${ARGS[@]}" --out "$D" > "$D/eval.out" 2>&1
rc=$?
n=$(( $(wc -l < "$D/episodes.csv" 2>/dev/null || echo 1) - 1 ))
if [[ $rc -eq 0 && $n -eq $N ]]; then echo "$(( $(date +%s) - t0 ))" > "$D/DONE"; echo "ok $C $G s$SEED ($(( $(date +%s) - t0 )) s)";
else echo "FAIL $C $G s$SEED rc=$rc episodes=$n (see $D/eval.out)"; exit 1; fi
