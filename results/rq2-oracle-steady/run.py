"""
@file run.py
@brief rq2-oracle-steady (2026-10-07): switching oracle + oracle imitation at steady state with long regime pieces.

Why (user, 10-07): on the 6 h B2 benchmark nothing beats MDD-TECT clearly, and behavior cloning of the oracle is at
the trivial baseline per slot (dev-oracle-bc). base-eplen found 6 h episodes sit in the warm-up transient, and that
6 h regime blocks give the largest, most persistent chunk-to-chunk differences. If imitation cannot clearly beat the
best fixed rule on that better-posed setting either, rule switching gives little beyond picking the right fixed rule.

Setting S6 (sim.py): warm-up >= 24 h, 12 h window, 6 h regime blocks, 1,800 s slots, B2 load mix, twin, tardiness.

Modes (one call = one seed; skips existing outputs; cluster: twin_array.sbatch):
  - oracle:  H15 greedy oracle (hold-to-end tails, all 15 kept per stage) + full obs v3 at each slot start along the
             oracle trajectory (caps 15 / 256, all-ones mask), as dev-oracle-bc's collect.py -> data/s<seed>.npz
             (same keys, so bc_train.py reads it with --data-dir).
  - rollout: deterministic slot policies from checkpoints on the seed -> rollout/s<seed>.json (CPU torch).
@par Usage
@code{.sh}
python results/rq2-oracle-steady/run.py --mode oracle --seed 0
python results/rq2-oracle-steady/run.py --mode rollout --seed 0 --checkpoints a.pt,b.pt
@endcode
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import sim

HERE = Path(__file__).resolve().parent
CAPS = (15, 256)


def oracle(seed, setting):
    from des_twin.observation import ObservationBuilder
    out = HERE / "data" / f"s{seed}.npz"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    ep = sim.Episode(setting, seed)
    n = sim.n_slots(setting)
    prefix, tails = [], np.zeros((n, len(sim.PAIRS)))
    for k in range(n):
        for i, p in enumerate(sim.PAIRS):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(sim.PAIRS[int(np.argmin(tails[k]))])
    builder = ObservationBuilder(sim.sig.floor(), *CAPS)
    obs = {}

    def capture(tw, dec, k):
        o = builder.build(tw, dec)
        o["action_mask"] = np.ones_like(o["action_mask"])
        obs[k] = o

    replay, _ = ep.play(prefix, capture=capture)
    missing = [k for k in range(n) if k not in obs]
    for k in missing:                      # a slot without a decision (rare): reuse the previous slot's obs
        obs[k] = obs[max(i for i in obs if i < k)]
    keys = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **{k: np.stack([obs[i][k] for i in range(n)]).astype(
        np.float16 if k in ("factory_grid", "job_table") else np.float32) for k in keys},
        tails=tails, schedule=np.array(prefix), pairs=np.array(sim.PAIRS), seed=seed, oracle=float(tails[-1].min()),
        warmup=ep.warmup, regimes=np.array([ep.regime_at(ep.warmup + (k + 0.5) * ep.cfg["slot"]) for k in range(n)]))
    fixed = dict(zip(sim.PAIRS, tails[0]))
    best = min(fixed, key=fixed.get)
    return (f"s{seed} {setting}: best fixed {best} {fixed[best]:.1f}, MDD-TECT {fixed['MDD-TECT']:.1f}, oracle "
            f"{tails[-1].min():.1f} ({100 * (tails[-1].min() / fixed[best] - 1):+.1f}% vs best fixed), replay diff "
            f"{abs(replay - tails[-1].min()):.1e}, missing obs {len(missing)}, {time.time() - t:.0f} s")


def rollout(seed, setting, checkpoints):
    import torch
    from des_twin.observation import ObservationBuilder
    from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES
    from models.network import SchedulingNetwork, encoder_config_for
    out = HERE / "rollout" / f"s{seed}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    todo = [c for c in checkpoints if c not in res]
    if not todo:
        return f"exists {out}"
    torch.set_num_threads(1)
    ep = sim.Episode(setting, seed)
    builder = ObservationBuilder(sim.sig.floor(), *CAPS)
    nj = len(JOB_HEAD_RULES)
    for ck in todo:
        state = torch.load(ck, map_location="cpu", weights_only=True)
        net = SchedulingNetwork(encoder_cfg=encoder_config_for(state["model_state_dict"]))
        net.load_state_dict(state["model_state_dict"])
        net.eval()

        def policy(tw, dec, k, _ep):
            o = builder.build(tw, dec)
            o["action_mask"] = np.ones_like(o["action_mask"])
            with torch.no_grad():
                t = {key: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for key, v in o.items()}
                logits = net.actor_critic.actor(net.fusion(net.encoder(t)))[0]
            return f"{JOB_HEAD_RULES[int(logits[:nj].argmax())]}-{MACHINE_HEAD_RULES[int(logits[nj:].argmax())]}"

        tard, played = ep.play(policy)
        res[ck] = {"tard": tard, "played": played}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    return f"s{seed}: " + ", ".join(f"{Path(c).stem} {res[c]['tard']:.1f}" for c in checkpoints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oracle", "rollout"), default="oracle")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--setting", default="S6")
    ap.add_argument("--checkpoints", default="")
    a = ap.parse_args()
    if a.mode == "oracle":
        print(oracle(a.seed, a.setting), flush=True)
    else:
        print(rollout(a.seed, a.setting, [c for c in a.checkpoints.split(",") if c]), flush=True)


if __name__ == "__main__":
    main()
