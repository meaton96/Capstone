#!/bin/bash
# Check that notifications work end to end. Run on the cluster from ~/capstone/linux_server after putting the webhook
# URL in ~/.capstone_webhook. Posts once from the login node, then submits a 2-minute, 1-CPU debug job that posts
# from a compute node (compute nodes may not have internet access; this is how to find out).
# Usage: ACCOUNT=drl-scheduling [NOTIFY_SLACK=@<slack user>] slurm/test_notify.sh
set -euo pipefail
: "${ACCOUNT:?}"
REPO="$(dirname "$PWD")"
mkdir -p logs
"$REPO/.venv/bin/python" -c "import sys; sys.path.insert(0, '$REPO/env'); from notify import notify, webhook_url; \
print('webhook configured:', bool(webhook_url())); print('login-node post ok:', notify(':wave: test from the login node'))"
sbatch --account="$ACCOUNT" --partition=debug --time=0-00:02:00 --cpus-per-task=1 --mem=1g --job-name=notify_test \
    --output=logs/notify_test_%j.out \
    ${NOTIFY_SLACK:+--mail-user=slack:$NOTIFY_SLACK --mail-type=END,FAIL} \
    --wrap "$REPO/.venv/bin/python -c \"import sys; sys.path.insert(0, '$REPO/env'); from notify import notify; \
print('compute-node post ok:', notify(':wave: test from compute node ' + __import__('socket').gethostname()))\""
echo "Submitted. Expect a webhook message from a compute node (and a Slack notice if NOTIFY_SLACK is set);"
echo "the result is also in logs/notify_test_<jobid>.out."
