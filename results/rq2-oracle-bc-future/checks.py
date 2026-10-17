"""
@file checks.py
@brief rq2-oracle-bc-future: local checks before the cluster runs (output in smoke/checks.out).
  1. collect.label on an evaluation seed reproduces rq2-rebase-v5's stored oracle npz (every key) exactly.
  2. The exact future table through the published-schedule adapter equals the direct table at every slot start.
  3. Perturbed levels change the table (mean |difference| vs exact), and redrawn changes it most.
@par Usage
@code{.sh}
python results/rq2-oracle-bc-future/checks.py label 2000
python results/rq2-oracle-bc-future/checks.py adapter 0,7,2003
@endcode
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect                                     # noqa: E402
import common as C                                 # noqa: E402

import numpy as np                                 # noqa: E402


def check_label(seed):
    out = C.HERE / "smoke" / "labels"
    print(collect.label(seed, out_dir=out))
    a, b = np.load(out / f"s{seed}.npz"), np.load(C.ORACLE_DIR / f"s{seed}.npz")
    for k in b.files:
        if k == "stamp":
            continue
        same = np.array_equal(a[k], b[k])
        print(f"  {k}: {'equal' if same else 'DIFFERS'}")
        assert same, k
    print(f"label s{seed}: every stored key reproduced; future {a['future'].shape}, t_slot[:3] {a['t_slot'][:3]}")


def check_adapter(seeds):
    for seed in seeds:
        ep = C.sim.Episode("B2", seed)
        sched = list(np.load(C.ORACLE_DIR / f"s{seed}.npz")["schedule"])
        views = {"direct": C.Published(seed, "exact"), "adapter": C.Published(seed, "exact", via_adapter=True)}
        views.update({lv: C.Published(seed, lv) for lv in C.LEVELS if lv != "exact"})
        tabs = {name: [] for name in views}

        def cap(tw, dec, k):
            for name, v in views.items():
                tabs[name].append(v(tw, C.STORE_ROWS))

        ep.play(sched, capture=cap)
        d, a = np.stack(tabs["direct"]), np.stack(tabs["adapter"])
        assert np.allclose(d, a, atol=2e-4), np.abs(d - a).max()
        diffs = {lv: float(np.abs(np.stack(tabs[lv]) - d).mean()) for lv in C.LEVELS if lv != "exact"}
        print(f"s{seed}: exact via the adapter = direct (max |diff| {np.abs(d - a).max():.1e}, {len(d)} slots, "
              f"{int(d[:, :, 0].sum(1).mean())} rows on average); mean |table - exact|: "
              + ", ".join(f"{k} {v:.4f}" for k, v in diffs.items()))
        assert diffs["redrawn"] > diffs["jit1"]


if __name__ == "__main__":
    C.check_frozen()
    if sys.argv[1] == "label":
        check_label(int(sys.argv[2]))
    else:
        check_adapter([int(s) for s in sys.argv[2].split(",")])
    C.check_frozen()
