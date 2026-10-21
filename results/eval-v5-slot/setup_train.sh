#!/bin/bash
# eval-v5-slot (10-21, user: PPO slot-length sweep): sets up one training run per slot length,
# results/train-v5-slot<X>/ (X in 60 120 180 300 600 1200 1500 1800 s; 900 s = the final-policy run train-v5-cmix).
# Each gets a copy of train-v5-cmix's frozen code/ (6d13412a snapshot, c_range 1.25 3.5, train_only) and a launch.sh
# that trains seeds 0-2 locally with the final policy's recipe, changing only --slot-seconds and --total-timesteps:
# equal experience = the cmix run's 48 updates x 900 / X (2,048 steps per update), so every arm sees about the same
# number of simulated episodes. Idempotent: existing code/ dirs are left alone.
set -euo pipefail
cd "$(dirname "$0")/../.."
for X in 60 120 180 300 600 1200 1500 1800; do
    D=results/train-v5-slot$X
    mkdir -p $D
    if [[ ! -d $D/code/env ]]; then
        rsync -a --exclude __pycache__ results/train-v5-cmix/code/ $D/code/
    fi
    UPD=$(python3 -c "print(round(48 * 900 / $X))")
    STEPS=$((UPD * 2048))
    cat > $D/launch.sh <<EOS
#!/bin/bash
# train-v5-slot$X (10-21, user: PPO slot-length sweep, results/eval-v5-slot/README.md): the final policy's recipe
# (train-v5-cmix: frozen code 6d13412a, c ~ U[1.25, 3.5], 16 envs, shared baseline groups of 4, 5,400 s discount
# horizon, entropy 0.01 -> 0.001) with --slot-seconds $X and $UPD updates ($STEPS agent steps, the same simulated
# experience as cmix's 48 updates at 900 s). Seeds 0-2 in parallel, locally, 2 twin workers each. Resumes.
set -euo pipefail
cd "\$(dirname "\$0")/../.."
DEVICE=cpu PY=.venv/bin/python bash results/scripts/dev_b2_slot_train_eval_cluster_c.sh train-v5-slot$X "0 1 2" 2 \\
    --prior-init offset --discount-horizon-s 5400 --shared-baseline-group 4 --rollout-length 128 --batch-size 256 \\
    --prior-pair "" --prior-kl-coef 0 --slot-seconds $X --total-timesteps $STEPS
EOS
    chmod +x $D/launch.sh
    echo "$D: $UPD updates, $STEPS steps"
done
