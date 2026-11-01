#!/bin/bash
# dev-size-1000-probe (10-19): one MDD-PTECT and one final-policy episode on the 1000-machine floor, in parallel,
# 8 h cap each (a timeout is itself the answer: record it). Then DONE.
set -u
cd "$(dirname "$0")/../.."
H=results/dev-size-1000-probe
for g in ref s3; do
    ( /usr/bin/time -v timeout 28800 .venv/bin/python $H/probe.py --group $g > $H/probe_$g.out 2> $H/probe_$g.time; echo "rc=$?" >> $H/probe_$g.out ) &
done
wait
date -Is > $H/DONE
