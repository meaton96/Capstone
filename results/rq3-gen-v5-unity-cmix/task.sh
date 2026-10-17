#!/bin/bash
# rq3-gen-v5-unity-cmix (10-17): one (condition, seed) of the RQ3 Unity subset for the final policy (cmix pick s3)
# and the loose-trained pick s2 next to it, on rq3-gen-v5-unity's conditions (load-1.2, due-tight, lay-g; conds.json is
# its copy). B2 test seeds 0-19. Only group pc runs here:
#   pc  MDD-PTECT, loose-trained pick (train-v5-loose s2), mixed-c pick (train-v5-cmix s3)
# The 6 pairs and the B2-trained pick s4 on the same seeds are ../rq3-gen-v5-unity's pa / pb runs; pc's MDD-PTECT checks
# them seed for seed.
# Same code / player / settings as ../rq3-gen-v5-unity/task.sh and ../eval-v5-cmix-unity/task.sh (6d13412a snapshot, linux_server/ 5e6e0f87 without
# -pyrules, random warm-up, 21,600 s window, 900 s slots, deterministic, discount horizon 10,800 s, which only changes
# the discounted return column), through ../rq4-agvfail-v5-rules/eval_wrap.py (Unity timeout_wait 1,800 s, nothing
# else changed). Skips a finished task (DONE).
# Usage: bash results/rq3-gen-v5-unity-cmix/task.sh COND GROUP SEED WORKER_ID
set -uo pipefail
C=$1; G=$2; SEED=$3; WID=$4
H=results/rq3-gen-v5-unity-cmix
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
