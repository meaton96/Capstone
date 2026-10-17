#!/bin/bash
# eval-v5-cmix-unity (10-15): one (condition, group, seed) of the Unity check of the due-date-trained PPO policies:
# do the twin gains of eval-v5-cmix (loose-trained -5.3% at loose c, mixed-c -2.1% at base) hold in Unity?
# Conditions (conds.json, built from ../rq3-gen-v5/run_task.py CONDITIONS, as ../eval-v5-cmix/cmix_task.py uses them):
# due-loose (c 2.5-3.5) and base (c 1.75-2.5). B2 test seeds 0-19, the same instances as the twin runs.
#   pa  SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT                 (due-loose only)
#   pb  ATC-PECT, ATC-PTECT, base-trained pick (train-v5-final s4) (due-loose only)
#   pc  MDD-PTECT, loose-trained pick (train-v5-loose s2), mixed-c pick (train-v5-cmix s3)
#   (picks made on the fitting seeds 2000-2159 in ../eval-v5-cmix. At base the pairs and s4 are reused from
#   ../eval-transfer-unity-v5 and the no-breakdown rows of ../rq4-agvfail-v5-*; pc's MDD-PTECT checks that reuse
#   seed for seed, and at due-loose repeats pa's as a run-to-run check.)
# Same code / player / settings as ../rq3-gen-v5-unity/task.sh (6d13412a snapshot, linux_server/ 5e6e0f87 without
# -pyrules, random warm-up, 21,600 s window, 900 s slots, deterministic, discount horizon 10,800 s, which only changes
# the discounted return column), through ../rq4-agvfail-v5-rules/eval_wrap.py (Unity timeout_wait 1,800 s, nothing
# else changed). Skips a finished task (DONE).
# Usage: bash results/eval-v5-cmix-unity/task.sh COND GROUP SEED WORKER_ID
set -uo pipefail
C=$1; G=$2; SEED=$3; WID=$4
H=results/eval-v5-cmix-unity
D=$H/runs/$C/$G/s$SEED
[[ -f $D/DONE ]] && { echo "exists $D"; exit 0; }
CODE=results/rq2-rebase-v5/code/env
PY=${PY:-.venv/bin/python}
P=$($PY -c "import json,sys; print(json.dumps(json.load(open('$H/conds.json'))['$C']['params']))") || { echo "unknown condition $C"; exit 2; }
L=$($PY -c "import json,sys; print(json.load(open('$H/conds.json'))['$C']['layout'] or '')")
LAY=(); [[ -n $L ]] && LAY=(--layout "$L")
BASE=results/train-v5-final_s4/checkpoint.pt; LOOSE=results/train-v5-loose_s2/checkpoint.pt
CMIX=results/train-v5-cmix_s3/checkpoint.pt
case $G in
    pa) ARGS=(--pdr SRT-PECT,SRT-PTECT,MDD-PECT,MDD-PTECT); N=4 ;;
    pb) ARGS=(--pdr ATC-PECT,ATC-PTECT --checkpoint $BASE); N=3 ;;
    pc) ARGS=(--pdr MDD-PTECT --checkpoint $LOOSE --checkpoint $CMIX); N=3 ;;
    *) echo "unknown group $G"; exit 2 ;;
esac
for f in $BASE $LOOSE $CMIX; do [[ -f $f ]] || { echo "missing checkpoint $f"; exit 1; }; done
rm -rf "$D"; mkdir -p "$D"
t0=$(date +%s)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY results/rq4-agvfail-v5-rules/eval_wrap.py $CODE \
    --unity-path linux_server/capstone.x86_64 --no-graphics \
    --base-worker-id "$WID" --reward-spec $CODE/config/rewards/tardiness.json --scenario-generator randomized \
    --random-warmup --episode-duration-seconds 21600 --params "$P" "${LAY[@]}" --seeds "$SEED" --slot-seconds 900 \
    --discount-horizon-s 10800 --device cpu "${ARGS[@]}" --out "$D" > "$D/eval.out" 2>&1
rc=$?
n=$(( $(wc -l < "$D/episodes.csv" 2>/dev/null || echo 1) - 1 ))
if [[ $rc -eq 0 && $n -eq $N ]]; then echo "$(( $(date +%s) - t0 ))" > "$D/DONE"; echo "ok $C $G s$SEED ($(( $(date +%s) - t0 )) s)";
else echo "FAIL $C $G s$SEED rc=$rc episodes=$n (see $D/eval.out)"; exit 1; fi
