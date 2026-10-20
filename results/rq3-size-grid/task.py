"""
@file task.py
@brief rq3-size-grid (2026-10-20, user: floor-size study phase 3): the cross-evaluation grid. Every training arm on every
       test floor size, twin, test seeds 0-39, c ~ U[1.75, 2.5] (B2's), next to the fixed pairs on the same floor.
       A thin wrapper around ../rq3-size-v5/task.py (rq3-gen-v5's run_task: frozen v5 code, B2 window / slots / reward,
       fleet wrapper, 10-seed chunks).
Floors (same per-machine scaling as rq3-size-v5): m15 = B2 itself (floor7, 7 AGVs, B2 load; condition "base");
m20 / m30 / m75 / m150 / m200 = ../dev-size-floors exports (M bays), round(0.52 M) AGVs on duty (short blocks half),
B2 load x 0.9, arrival cap off; m50 / m100 = rq3-size-v5's (agv-ratio-scale floors, identical to the new exports).
Groups (each policy group + MDD-PTECT as the run-to-run check):
  pdr      the 6 v5 pairs + MDD-TECT
  cmix     the final policy family (train-v5-cmix s0-s4, trained on 15 machines)        = arm T15
  size50   train-size-m50 s0-s4                                                          = arm T50
  size100  train-size-m100 s0-s4                                                         = arm T100
  sizemix  train-size-mix s0-s4 (trained on 15 + 50 + 100)                               = arm Tmix
Output: <cond>/<group>_s<a>-<b>/, in 10-seed chunks up to m100 and 5-seed chunks on m150 / m200 (one m200 episode
takes ~300 s locally, rule or policy; an m100 10-seed chunk took ~2.6 h on the cluster, so a 10-seed m200 chunk of 7
policies would come close to the 10 h limit). A finished task (DONE) is skipped.
m15 / m50 / m100 pdr and cmix are not rerun here: ../rq3-gen-v5 base, ../eval-v5-cmix/test base and ../rq3-size-v5.
@par Usage
@code{.sh}
python results/rq3-size-grid/task.py --cond m200 --group cmix --chunk 0-9
@endcode
"""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("size_task", HERE.parent / "rq3-size-v5" / "task.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)
rt = st.rt
FLOORS = REPO / "results" / "dev-size-floors" / "floors"
NEW_SIZES = [20, 30, 75, 150, 200]
for m in NEW_SIZES:
    fleet = round(0.52 * m)
    rt.CONDITIONS[f"m{m}"] = (st.size_params(m // 5, fleet, round(fleet / 2)), FLOORS / f"m{m}" / "des_floor.json",
                              None, (fleet, m))
rt.CONDITIONS["m15"] = rt.CONDITIONS["base"]
ARMS = {"size50": "train-size-m50", "size100": "train-size-m100", "sizemix": "train-size-mix"}
for g, run in ARMS.items():
    ck = [f"results/{run}_s{k}/checkpoint.pt" for k in range(5)]
    rt.GROUPS[g] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ck for a in ("--checkpoint", c)])
CHUNKS10 = st.CHUNKS                                              # 0-9 ... 30-39
CHUNKS5 = [f"{a}-{a + 4}" for a in range(0, 40, 5)]               # 0-4 ... 35-39
for g in ("pdr", "cmix", *ARMS):
    for c in CHUNKS10 + CHUNKS5:
        rt.GROUPS[f"{g}_s{c}"] = rt.GROUPS[g]
CONDS = ["m15", "m20", "m30", "m50", "m75", "m100", "m150", "m200"]
BIG = {"m150", "m200"}


def chunks(cond):
    return CHUNKS5 if cond in BIG else CHUNKS10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--group", required=True, choices=["pdr", "cmix", *ARMS])
    ap.add_argument("--chunk", required=True, choices=CHUNKS10 + CHUNKS5)
    ap.add_argument("--out-root", default=str(HERE))
    a = ap.parse_args(argv)
    if a.chunk not in chunks(a.cond):
        ap.error(f"{a.cond} runs in chunks {chunks(a.cond)}")
    rt.main(["--cond", a.cond, "--group", f"{a.group}_s{a.chunk}", "--seeds", a.chunk, "--out-root", a.out_root])


if __name__ == "__main__":
    main()
