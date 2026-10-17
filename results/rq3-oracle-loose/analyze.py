"""
@file analyze.py
@brief rq3-oracle-loose (2026-10-14): the switching-oracle ceiling at loose due dates (c ~ U[2.5, 3.5]) on B2 test seeds
       0-39, against MDD-PTECT (the B2 reference and due-loose's best v5 pair in rq3-gen-v5) and the per-instance
       hindsight best v5 pair, in pooled window tardiness and in job-seconds saved per instance. The base-c reference
       is rq2-rebase-v5 (oracle -9.4% vs MDD-PTECT). -> analysis.out
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(f.read_text()) for f in sorted(HERE.glob("out/s*.json"))]
    if not rows:
        print("no outputs yet")
        return
    o = np.array([r["oracle"] for r in rows])
    ref = np.array([r["fixed"]["MDD-PTECT"] for r in rows])
    hind = np.array([min(r["fixed"].values()) for r in rows])
    rng = np.random.default_rng(0)

    def cmp(y, b):
        idx = rng.integers(0, len(y), (10000, len(y)))
        bs = 100 * (y[idx].sum(1) / b[idx].sum(1) - 1)
        d = y - b
        return (f"{100 * (y.sum() / b.sum() - 1):+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}] "
                f"{int((d < 0).sum())}/{int((d > 0).sum())} p={wilcoxon(d).pvalue if np.any(d) else 1:.2g}; "
                f"saved {-(d.mean()):.2f} k job-s per instance")
    print(f"== rq3-oracle-loose: {len(rows)} seeds, c ~ U[2.5, 3.5], vs MDD-PTECT (mean {ref.mean():.2f} k job-s)")
    print(f"  switching oracle        {cmp(o, ref)}")
    print(f"  hindsight best v5 pair  {cmp(hind, ref)}")
    use = {}
    for r in rows:
        for p in r["schedule"]:
            use[p] = use.get(p, 0) + 1
    tot = sum(use.values())
    print("  oracle pair use: " + ", ".join(f"{p} {100 * n / tot:.0f}%" for p, n in sorted(use.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
