#!/bin/bash
# rq4-agvfail-v5 (10-14): one (level, group, seed) Unity evaluation of the RQ4 rerun on the v5 rules, under AGV
# breakdowns. Shared by rq4-agvfail-v5-rules (fixed pairs) and rq4-agvfail-v5-policy (PPO + BC), which pass their own
# ROOT. Same setup as ../eval-transfer-unity-v5/task.sh (its no-breakdown rows are reused as level none):
#   code   rq2-rebase-v5/code/env, the snapshot of train-v5-nokl / train-v5-final (6d13412a, diff e3b0c442)
#   player linux_server/ (Simulation.dll 5e6e0f87, without -pyrules = the v5 player; the cluster copy is glibc-patched)
#   B2 instances (randomized generator, B2 params, --random-warmup, 21,600 s window, tardiness spec), slot actions 900 s,
#   deterministic, discount horizon 10,800 s as eval-transfer-unity-v5 (it only changes the discounted return column)
#   levels: none | l3000 | l1500 = AGV Weibull failures with lambda 3000 / 1500 s, other fields at
#           scenarios.AGV_FAILURE_DEFAULTS (as ../rq4-agvfail-policy)
#   groups: pa  SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT        pb  ATC-PECT, ATC-PTECT, MDD-TECT (H15 best), MDD-PTECT
#           pn  (level none) SRT-PECT, SRT-PTECT, ATC-PECT, ATC-PTECT, MDD-PTECT (the 4 pairs eval-transfer-unity-v5
#               did not run; MDD-PTECT again as a run-to-run check against it)
#           ppoa PPO train-v5-nokl s0 / s1 / s2 + MDD-PTECT     ppob PPO train-v5-final s3 / s4
#           bc   BC rq2-oracle-bc-v5 bc_full_s0 / s1 / s2
#   (pb's MDD-PTECT and ppoa's MDD-PTECT repeat pa's: run-to-run checks under breakdowns.)
# Skips a finished task (DONE). Usage: bash results/rq4-agvfail-v5-rules/task.sh ROOT LEVEL GROUP SEED WORKER_ID
set -uo pipefail
ROOT=$1; LV=$2; G=$3; SEED=$4; WID=$5
D=$ROOT/runs/$LV/$G/s$SEED
[[ -f $D/DONE ]] && { echo "exists $D"; exit 0; }
CODE=results/rq2-rebase-v5/code/env
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
case $LV in
    none)  FAIL=() ;;
    l3000) FAIL=(--agv-failures '{"agvWeibullLambda": 3000}') ;;
    l1500) FAIL=(--agv-failures '{"agvWeibullLambda": 1500}') ;;
    *) echo "unknown level $LV"; exit 2 ;;
esac
case $G in
    pa)   ARGS=(--pdr SRT-PECT,SRT-PTECT,MDD-PECT,MDD-PTECT); N=4 ;;
    pb)   ARGS=(--pdr ATC-PECT,ATC-PTECT,MDD-TECT,MDD-PTECT); N=4 ;;
    pn)   ARGS=(--pdr SRT-PECT,SRT-PTECT,ATC-PECT,ATC-PTECT,MDD-PTECT); N=5 ;;
    ppoa) ARGS=(--pdr MDD-PTECT --checkpoint results/train-v5-nokl_s0/checkpoint.pt
                --checkpoint results/train-v5-nokl_s1/checkpoint.pt --checkpoint results/train-v5-nokl_s2/checkpoint.pt); N=4 ;;
    ppob) ARGS=(--pdr none --checkpoint results/train-v5-final_s3/checkpoint.pt
                --checkpoint results/train-v5-final_s4/checkpoint.pt); N=2 ;;
    bc)   ARGS=(--pdr none --checkpoint results/rq2-oracle-bc-v5/bc/bc_full_s0.pt
                --checkpoint results/rq2-oracle-bc-v5/bc/bc_full_s1.pt --checkpoint results/rq2-oracle-bc-v5/bc/bc_full_s2.pt); N=3 ;;
    *) echo "unknown group $G"; exit 2 ;;
esac
rm -rf "$D"; mkdir -p "$D"
PY=${PY:-.venv/bin/python}
t0=$(date +%s)
# eval_wrap.py: the frozen evaluate.py with mlagents' timeout_wait raised from 300 to 1,800 s (seed 32's 8,399 s warm-up
# outlasted 300 s on cluster nodes, 10-14); results are unchanged by construction.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY results/rq4-agvfail-v5-rules/eval_wrap.py $CODE --unity-path linux_server/capstone.x86_64 --no-graphics \
    --base-worker-id "$WID" --reward-spec $CODE/config/rewards/tardiness.json --scenario-generator randomized \
    --random-warmup --episode-duration-seconds 21600 --params "$P" --seeds "$SEED" --slot-seconds 900 \
    --discount-horizon-s 10800 --device cpu "${FAIL[@]}" "${ARGS[@]}" --out "$D" > "$D/eval.out" 2>&1
rc=$?
n=$(( $(wc -l < "$D/episodes.csv" 2>/dev/null || echo 1) - 1 ))
if [[ $rc -eq 0 && $n -eq $N ]]; then echo "$(( $(date +%s) - t0 ))" > "$D/DONE"; echo "ok $LV $G s$SEED ($(( $(date +%s) - t0 )) s)";
else echo "FAIL $LV $G s$SEED rc=$rc episodes=$n (see $D/eval.out)"; exit 1; fi
