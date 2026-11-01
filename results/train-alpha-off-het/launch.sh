#!/bin/bash
# train-alpha-off-het (10-25, user: "the law sets alpha, the head picks an offset"; results/dev-alpha-offset): the v7
# action space (job x machine x offset heads; alpha = min(4, max(0, 3 - 6 sigma) x offset), sigma from the floor's
# machine speed factors) with the final policy's recipe on B2 with a machine-speed spread s ~ U[0, 1] per instance
# (code/hetero.py, as train-alpha-het), 5 seeds in parallel locally, 2 twin workers each, one process per seed with a
# 12 GB address-space cap. Question: does law + offset keep the het arm's gain on spread floors without its cost on B2?
# Resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=cpu PY=.venv/bin/python bash results/scripts/alpha_offset_train.sh train-alpha-off-het "0 1 2 3 4" 2 0,1
