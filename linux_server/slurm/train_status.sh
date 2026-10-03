#!/bin/bash
# Training status from your machine: queue state plus each seed's progress.json (written by train.py every update).
# Usage (from the repo's linux_server/): slurm/train_status.sh RUN_NAME [STALE_MINUTES, default 60]
# A seed is flagged STALE when progress.json is older than STALE_MINUTES while its job is running, and MISSING when
# there is no progress.json yet (normal only for the first few minutes after a job starts).
set -euo pipefail
RUN="${1:?usage: slurm/train_status.sh RUN_NAME [STALE_MINUTES]}"
STALE="${2:-60}"
source "$(dirname "$0")/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" bash -s -- "$RUN" "$STALE" <<'REMOTE_EOF' 2>/dev/null | sed -n '/^=== /,$p'
RUN="$1"; STALE="$2"; cd ~/capstone
echo "=== queue ($RUN)"
squeue --me --name="$RUN" -o "%.20i %.9T %.11M %.11l %R" || true
now=$(date +%s)
for d in results/"$RUN"_s*; do
  [ -d "$d" ] || continue
  f="$d/progress.json"
  if [ ! -f "$f" ]; then echo "=== ${d#results/}: MISSING progress.json (not past its first update)"; continue; fi
  age=$(( (now - $(stat -c %Y "$f")) / 60 ))
  flag=""; [ "$age" -gt "$STALE" ] && flag="  <-- STALE"
  python3 - "$f" "$age" "$flag" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); age, flag = sys.argv[2], sys.argv[3]
ret = f", recent return {p['recent_return']:.2f}" if p.get("recent_return") is not None else ""
flow = f", recent mean flow {p['recent_mean_flow']:.0f} s (censored)" if p.get("recent_mean_flow") is not None else ""
print(f"=== {p['run_id']}: step {p['global_step']:,}/{p['total_timesteps']:,} "
      f"({100 * p['global_step'] / max(p['total_timesteps'], 1):.1f}%), {p['sps']:.1f} SPS, ETA {p['eta_hours']:.1f} h, "
      f"{p['episodes']} episodes{ret}{flow}; updated {age} min ago{flag}")
PY
done
REMOTE_EOF
