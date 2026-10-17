"""
@file task.py
@brief rq4-uncert-v5-policy (2026-10-17, user: RQ4 covers disruptions and uncertainty): the final policy (PPO trained on
       mixed due dates, cmix pick s3), the other PPO families and the 6 v5 pairs in the cells of the twin uncertainty
       screen (../rq4-uncert-screen: rules / planner / oracle only; gate G1 stopped training under noise, so no policy
       was ever scored there). B2, test seeds 0-39, noise seed = instance seed (common random numbers across policies).
@details A thin wrapper around ../rq3-gen-v5/run_task.py (frozen code ../rq2-rebase-v5/code/env = the checkpoints' code,
         B2 settings, condition "base") that adds `--uncertainty <cell>` to evaluate.py. The policy sees the observed
         (noisy) state, as the uncertainty layer gives it (twin_env uses twin.observed). The frozen code's
         des_twin/uncertainty.py differs from the screen's snapshot only by the 10-11 redraw clip fix, which plain
         forward runs do not use; analyze.py checks MDD-PTECT against the screen's rule outputs seed for seed.
Cells: the 18 realistic / severe cells of the 9 noise types (E1, E2, E2d, E3, E4n, O1, O2, O3, mix) and the 6 online
estimator cells (E2, E2d, mix at realistic / severe with info=I1). "none" = B2 (../rq3-gen-v5 base, ../eval-v5-cmix/test).
Groups: pdr (6 v5 pairs + MDD-TECT), ppo (B2-trained s0-s4), cmix (s0-s4), loose (s0-s4); each policy group + MDD-PTECT.
Output: <cell dir>/base/<group>/ (cell dir = cell with "=" -> "-", "," -> "_", as the screen); DONE skips.
@par Usage
@code{.sh}
python results/rq4-uncert-v5-policy/task.py --cell "E2d=realistic" --group cmix [--seeds 0-39]
@endcode
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rq3-gen-v5"))
import run_task as rt                                              # noqa: E402  (rq3-gen-v5's harness)

TYPES = ["E1", "E2", "E2d", "E3", "E4n", "O1", "O2", "O3", "mix"]
CELLS = [f"{t}={lv}" for t in TYPES for lv in ("realistic", "severe")] + \
    [f"{t}={lv},info=I1" for t in ("E2", "E2d", "mix") for lv in ("realistic", "severe")]
for name, run in (("loose", "train-v5-loose"), ("cmix", "train-v5-cmix")):
    ck = [f"results/{run}_s{k}/checkpoint.pt" for k in range(5)]
    rt.GROUPS[name] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ck for a in ("--checkpoint", c)])


def cell_dir(cell):
    return cell.replace(",", "_").replace("=", "-")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--group", required=True, choices=["pdr", "ppo", "cmix", "loose"])
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--out-root", default=str(HERE))
    a = ap.parse_args(argv)
    slot, extra = rt.GROUPS[a.group]
    rt.GROUPS[a.group] = (slot, extra + ["--uncertainty", a.cell])
    rt.main(["--cond", "base", "--group", a.group, "--seeds", a.seeds, "--out-root", str(Path(a.out_root) / cell_dir(a.cell))])


if __name__ == "__main__":
    main()
