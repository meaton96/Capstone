"""
@file rollout.py
@brief rq2-oracle-bc-future (2026-10-15): play the BC clones (bc/bc_<arm>_s<k>.pt) as 900 s slot policies on B2, in the
       frozen twin of the labels (common.py; port of ../rq2-oracle-bc-v5/rollout.py).

At each slot's first decision the network sees the observation (caps 15 / 256, all-ones action mask, as in the labels)
and, for a future arm, the future job table (its rows, 2 h). It plays argmax job rule x argmax machine rule for the
slot. Future arms run at every forecast level of common.LEVELS: exact = the real schedule; jit1 / jit10 / jit30 /
churn10 / churn25 / redrawn = rq2-perturbed-v5's published schedule at that level (the planner's draws), read through
eval-v5-lookahead-fcast's PublishedFuture as twin.unc.plan_future_jobs. The world the policy acts in is always the
real instance. The state arm reads no future, so it runs at exact only.
Output: out/rollout/s<seed>.json {level: {checkpoint: {tard, played}}, "_stamp": ...}; entries already there are
skipped (reruns resume).
@par Usage
@code{.sh}
python results/rq2-oracle-bc-future/rollout.py 0-4
python results/rq2-oracle-bc-future/rollout.py 0 --checkpoints bc/bc_future64_s0.pt --levels exact,redrawn
@endcode
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                 # noqa: E402  (frozen code first; overlays the future table)

import numpy as np                                 # noqa: E402
import torch                                       # noqa: E402
from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES   # noqa: E402
from des_twin.observation import ObservationBuilder     # noqa: E402
from models.network import encoder_config_for           # noqa: E402

MAIN_ARMS = ("state", "future64", "future128")


def load_policy(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]
    nj = len(JOB_HEAD_RULES)
    width = sd["actor_critic.actor.net.3.bias"].shape[0]
    assert width == nj + len(MACHINE_HEAD_RULES), f"{path}: {width} logits, heads are v5"
    net = C.future_net.network_for(sd, encoder_cfg=encoder_config_for(sd))
    net.load_state_dict(sd)
    net.eval()
    rows = int(state.get("run_settings", {}).get("future_rows", 0))
    assert bool(rows) == C.future_net.reads_future_table(sd), f"{path}: future rows {rows} vs the weights"
    return net, rows


def play(ep, net, rows, view, builder):
    nj = len(JOB_HEAD_RULES)

    def policy(tw, dec, k, _ep):
        o = builder.build(tw, dec)
        o["action_mask"] = np.ones_like(o["action_mask"])
        if rows:
            o["future_job_table"] = view(tw, rows)
        with torch.no_grad():
            t = {key: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for key, v in o.items()}
            logits = net.actor_critic.actor(net._features(t))[0]
        return f"{JOB_HEAD_RULES[int(logits[:nj].argmax())]}-{MACHINE_HEAD_RULES[int(logits[nj:].argmax())]}"

    return ep.play(policy)


def rollout(seed, nets, levels, out_dir):
    """@p nets: {checkpoint path: (network, future rows)}."""
    out = out_dir / f"s{seed}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    todo = [(ck, lv) for ck, (_, rows) in nets.items() for lv in levels
            if (rows or lv == "exact") and Path(ck).name not in res.get(lv, {})]   # the state arm reads no future
    if not todo:
        return f"s{seed}: done"
    t0 = time.time()
    ep = C.sim.Episode("B2", seed)
    builder = ObservationBuilder(C.sim.sig.floor(), *C.CAPS)
    for ck, lv in todo:
        net, rows = nets[ck]
        tard, played = play(ep, net, rows, C.Published(seed, lv), builder)
        res.setdefault(lv, {})[Path(ck).name] = {"tard": tard, "played": played}
    res["_stamp"] = C.stamp()
    C.write_atomic(out, json.dumps(res))
    return f"s{seed}: {len(todo)} runs, {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", help="e.g. 0-4 or 2000-2009")
    ap.add_argument("--checkpoints", default="", help="comma list (relative to this folder); default: every "
                    "<bc-dir>/bc_<arm>_s<k>.pt of the main arms without a tag")
    ap.add_argument("--bc-dir", default="bc", help="checkpoint folder (bc4k: the 4,000-seed data-scaling run, 10-16)")
    ap.add_argument("--levels", default=",".join(C.LEVELS))
    ap.add_argument("--out", default=str(C.HERE / "out" / "rollout"))
    ap.add_argument("--expect", type=int, default=0, help="fail unless this many checkpoints are found")
    a = ap.parse_args()
    if a.checkpoints:
        cks = [str(C.HERE / c) if not Path(c).is_absolute() else c for c in a.checkpoints.split(",")]
    else:
        cks = sorted(str(p) for arm in MAIN_ARMS for p in (C.HERE / a.bc_dir).glob(f"bc_{arm}_s*.pt")
                     if p.stem.split("_s")[-1].isdigit())
    if not cks or (a.expect and len(cks) != a.expect):
        raise SystemExit(f"{len(cks)} checkpoints, expected {a.expect or '> 0'}")
    levels = a.levels.split(",")
    assert all(lv in C.LEVELS for lv in levels), levels
    torch.set_num_threads(1)
    nets = {ck: load_policy(ck) for ck in cks}
    for s in C.sim.parse_seeds(a.seeds):
        print(rollout(s, nets, levels, Path(a.out)), flush=True)


if __name__ == "__main__":
    C.check_frozen()
    main()
    C.check_frozen()
