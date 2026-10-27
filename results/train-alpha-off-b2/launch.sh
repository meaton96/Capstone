#!/bin/bash
# train-alpha-off-b2 (10-25, user: "the law sets alpha, the head picks an offset"; results/dev-alpha-offset): the v7
# action space (job x machine x offset heads) with the final policy's recipe on B2 itself (spread 0,0: sigma 0, so the
# law is alpha 3 = v5 and the offsets reach 0.75-4), 5 seeds in parallel locally, 2 twin workers each, one process per
# seed with a 12 GB address-space cap. Question: on the training floor, does the offset head only add (regime offsets)
# to the final policy's action space? Resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=cpu PY=.venv/bin/python bash results/scripts/alpha_offset_train.sh train-alpha-off-b2 "0 1 2 3 4" 2 0,0
