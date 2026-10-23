"""
@file analyze.py
@brief base-late-share (2026-10-22): the share of late jobs on B2 by due-date allowance c and rule. Per setting and pair:
       the late share (jobs due in the window that finish late or are still open at its end, over all jobs due in it),
       pooled over seeds with a seed-bootstrap 95% CI; the same share among jobs that exit in the window; mean window
       tardiness. Seeds 0-39 and 1000-1039 (80 instances), also shown separately for 0-39. -> analysis.out, summary.csv
"""
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("r", HERE / "run.py")
r = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r)


def share(rows, key_late, key_all, rng=None):
    late = np.array([x[key_late] for x in rows], float)
    tot = np.array([x[key_all] for x in rows], float)
    p = late.sum() / tot.sum()
    if rng is None:
        return p, None, None
    idx = rng.integers(0, len(rows), (5000, len(rows)))
    b = late[idx].sum(1) / tot[idx].sum(1)
    return p, np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    rng = np.random.default_rng(0)
    print("== base-late-share: share of late jobs on B2 (15 machines, 7 AGVs, 6 h window), seeds 0-39 + 1000-1039")
    print("   late share = jobs due in the window that finish late or are still open at its end / jobs due in it")
    out = []
    for setting, c in r.SETTINGS.items():
        f = HERE / "out" / f"{setting}.json"
        if not f.exists():
            print(f"\n-- {setting}: not run yet")
            continue
        res = json.loads(f.read_text())["results"]
        lab = "B2: c ~ U[1.75, 2.5] per 1.5 h block" if c is None else f"c = {c:g}"
        print(f"\n-- {lab} ({len(res)} instances)")
        for p in r.PAIRS:
            rows = [v[p] for v in res.values()]
            test = [v[p] for s, v in res.items() if int(s) < 1000]
            sh, lo, hi = share(rows, "late", "due_in", rng)
            st, _, _ = share(test, "late", "due_in")
            ex, _, _ = share(rows, "exited_late", "exited")
            tard = float(np.mean([x["tard"] for x in rows]))
            print(f"   {p:10s} late share {100 * sh:5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}] (test seeds 0-39: "
                  f"{100 * st:5.1f}%) | among jobs that exit in the window {100 * ex:5.1f}% | mean window tardiness "
                  f"{tard:6.1f} k job-s")
            out.append({"setting": setting, "c": c, "pair": p, "late_share": sh, "ci_lo": lo, "ci_hi": hi,
                        "late_share_test": st, "late_share_exited": ex, "mean_window_tardiness_k": tard,
                        "instances": len(rows), "jobs_due_in_window": int(sum(x["due_in"] for x in rows))})
    df = pd.DataFrame(out)
    df.to_csv(HERE / "summary.csv", index=False)
    if not df.empty:
        piv = df.pivot_table(index="setting", columns="pair", values="late_share").reindex(
            [s for s in r.SETTINGS if s in set(df.setting)])
        print("\n== late share by setting (pooled, 80 instances)")
        print((100 * piv).round(1).to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
