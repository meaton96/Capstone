"""
@file make_b2tb.py
@brief agv-study-rerun, 105-machine b2t cells (user 10-16): regime b2tb = the first 1.5 h load block (block 0,
       arrivals in [0, 5400) s) of agv5_b2t_t7iso_s0 / agv5_b2t_t7open_s0, run to empty. The full b2t-105 cells
       (5,893 jobs) take more than 30 h per cell in the player (dev-agv105-probe: 0.73 sim-s per wall-s at 84 AGVs),
       so the user chose one block: the same jobs as the full cells' first block (1,685 jobs, opening burst included),
       tiling, due dates and everything else unchanged. Writes linux_server/BatchConfigs/Scenarios/agv5_b2tb_t7*_s0.json.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "linux_server" / "BatchConfigs" / "Scenarios"
END = 5400.0

for scope in ("iso", "open"):
    src = f"agv5_b2t_t7{scope}_s0"
    s = json.loads((SC / f"{src}.json").read_text())
    n_all = len(s["jobs"])
    s["jobs"] = [j for j in s["jobs"] if j["arrivalTime"] < END]
    s["_phases"] = [p for p in s["_phases"] if p["start"] < END]
    s["_meta"] = dict(s["_meta"], blocks=[b for b in s["_meta"]["blocks"] if b["start"] < END])
    s["name"] = f"agv5_b2tb_t7{scope}_s0"
    s["_comment"] = f"b2tb: block 0 (arrivals < {END:.0f} s) of {src}, {len(s['jobs'])} of {n_all} jobs. " + s["_comment"]
    out = SC / f"{s['name']}.json"
    out.write_text(json.dumps(s))
    print(f"{out.relative_to(REPO)}: {len(s['jobs'])} of {n_all} jobs, last arrival "
          f"{max(j['arrivalTime'] for j in s['jobs']):.0f} s")
