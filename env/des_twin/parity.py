"""Observation parity: the twin's observation and action masks against the Unity player's, decision by decision.

Launches the player in RL mode (as training does) with "-destrace", replays one scripted scenario under a fixed
rule given as the two action heads, and drives the twin (agent mode, DES-1k) with the same answers from the
floor and jobs the player exported. With one AGV the twin reproduces the player's decisions exactly (twin-dock),
so every decision's observation must match; with more AGVs they agree until the first AGV conflict, after which
the decision times drift apart (reported, and the comparison stops there).

Columns that cannot match by design are reported apart from the strict check (see des_twin/observation.py):
the job channel of the grid and the routing focus distance (machine column 15), which Unity takes from the job
visuals. Everything else must agree within --tol (float32 rounding).

Needs a player built after 2026-10-01 (observation frame in des_floor.json, -destrace honoured under ML-Agents).

  python -m des_twin.parity --player ../linux_server/capstone.x86_64 \\
      --scenario ../linux_server/BatchConfigs/Scenarios/rnd_load_s0.json --agvs 1 --rule SPT_ECT --out ../results/dev-twin-parity

Writes <out>/parity_decisions.csv (one row per decision: time gap, kind, mask match, max error per stream) and
<out>/parity_columns.csv (per stream and column: decisions compared, mismatches, max error), and prints a summary.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES  # noqa: E402
from des_twin.engine import Twin, TwinConfig  # noqa: E402
from des_twin.floor import Floor  # noqa: E402
from des_twin.observation import ObservationBuilder  # noqa: E402
from des_twin.scenario import episode_settings, resolve_jobs  # noqa: E402

STREAMS = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags")
# (stream, column) pairs Unity computes from job visuals; for the grid the "column" is the channel
BY_DESIGN = {("factory_grid", 1), ("machine_table", 15)}


def _columns(stream, arr):
    """(column key, values) pairs: grid by channel, tables by feature column, vectors by index."""
    if stream == "factory_grid":
        return [(c, arr[c]) for c in range(arr.shape[0])]
    if arr.ndim == 2:
        return [(c, arr[:, c]) for c in range(arr.shape[1])]
    return [(c, arr[c:c + 1]) for c in range(arr.shape[0])]


def _heads(rule):
    job, machine = rule.upper().split("_")
    if job not in JOB_HEAD_RULES or machine not in MACHINE_HEAD_RULES:
        raise SystemExit(f"{rule}: both halves must be RL heads (job {JOB_HEAD_RULES}, machine {MACHINE_HEAD_RULES})")
    return JOB_HEAD_RULES.index(job), MACHINE_HEAD_RULES.index(machine)


def _wait_for(path, timeout=60.0):
    end = time.time() + timeout
    while not path.exists():
        if time.time() > end:
            raise SystemExit(f"{path} was not written: is the player built after 2026-10-01 (-destrace under "
                             "ML-Agents)?")
        time.sleep(0.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", required=True, help="Unity player executable")
    ap.add_argument("--scenario", required=True, help="scenario JSON (ScenarioLoader schema)")
    ap.add_argument("--agvs", type=int, default=1, help="fleet size (1: decisions must match exactly)")
    ap.add_argument("--layout", default=None, help="layout override (default: the scenario's)")
    ap.add_argument("--rule", default="SPT_ECT", help="fixed rule, both halves RL heads (e.g. SRT_TECT)")
    ap.add_argument("--keep-stochastic", action="store_true",
                    help="keep the scenario's warm-up / cap (failures are refused either way)")
    ap.add_argument("--max-decisions", type=int, default=0, help="stop after this many decisions (0: the episode)")
    ap.add_argument("--max-skip-steps", type=int, default=200_000,
                    help="steps allowed to finish the player's start-up episode (its default config)")
    ap.add_argument("--tol", type=float, default=1e-5, help="absolute tolerance per value")
    ap.add_argument("--obs-caps", type=int, nargs=2, default=(15, 256), metavar=("MACHINES", "JOBS"))
    ap.add_argument("--out", required=True, help="output directory (also the player's -decisionlogdir)")
    ap.add_argument("--worker-id", type=int, default=0)
    args = ap.parse_args(argv)

    from env_wrappers.unity_env import UnitySchedulingEnv   # mlagents_envs only needed here

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(args.scenario) as f:
        scenario = json.load(f)
    scenario["agvCount"] = args.agvs
    if args.layout:
        scenario["layout"] = args.layout.upper()
    if not args.keep_stochastic:
        scenario.pop("stochastic", None)
    episode_settings(scenario)                     # refuses failures before launching anything
    scen_path = out / "parity_scenario.json"
    scen_path.write_text(json.dumps(scenario, indent=1))
    heads = _heads(args.rule)

    env = UnitySchedulingEnv(file_name=args.player, no_graphics=True, worker_id=args.worker_id,
                             obs_caps=tuple(args.obs_caps), log_file=str(out / "Player.log"),
                             extra_args=["-decisionlogdir", str(out), "-destrace"])
    try:
        env.load_scenario(scen_path)
        env.reset()
        for f in ("des_floor.json", "des_jobs.json"):
            (out / f).unlink(missing_ok=True)      # the start-up episode's export; the scenario's comes next
        for n in range(args.max_skip_steps):       # finish the episode that was running when Python connected
            obs_u, _, done, _ = env.step(heads)
            if done:
                break
        else:
            raise SystemExit("the player's start-up episode did not end within --max-skip-steps")
        _wait_for(out / "des_floor.json")
        _wait_for(out / "des_jobs.json")
        floor = Floor.load(out / "des_floor.json")
        with open(out / "des_jobs.json") as f:
            jobs = json.load(f)

        resolved = resolve_jobs(scenario, floor)
        same_jobs = [(j["id"], np.float32(j["arrival"]), [[(m, np.float32(d)) for m, d in o["eligible"]]
                                                          for o in j["ops"]]) for j in resolved["jobs"]] == \
                    [(j["id"], np.float32(j["arrival"]), [[(m, np.float32(d)) for m, d in o["eligible"]]
                                                          for o in j["ops"]]) for j in jobs["jobs"]]
        print(f"scenario jobs resolved in Python == player's des_jobs.json: {same_jobs}")
        if jobs.get("instance") != scenario.get("name"):
            print(f"WARNING: the export is instance {jobs.get('instance')!r}, not the scenario {scenario.get('name')!r}")

        warmup, cap, warm_rule = episode_settings(scenario)
        twin = Twin(floor, jobs, TwinConfig(rule=warm_rule, transport="kinematic", agv_count=args.agvs,
                                             routing_trigger=scenario.get("routingTrigger"),
                                             warmup_seconds=warmup, episode_duration_seconds=cap,
                                             max_sim_seconds=100_000.0))
        builder = ObservationBuilder(floor, *args.obs_caps, instant_fleet=args.agvs)
        gen = twin.agent_decisions()
        dec = next(gen)
        halves = (JOB_HEAD_RULES[heads[0]], MACHINE_HEAD_RULES[heads[1]])

        col_stats = {}
        rows = []
        desync = None
        end_differs = False
        i = 0
        while True:
            obs_t = builder.build(twin, dec)
            t_u = env.current_metrics.sim_time
            gap = t_u - twin.now
            row = {"decision": i, "unity_time": round(t_u, 4), "twin_time": round(twin.now, 4),
                   "gap": round(gap, 4), "kind": dec.kind,
                   "mask_match": bool(np.array_equal(obs_u["action_mask"], obs_t["action_mask"]))}
            if abs(gap) > twin.dt / 2 and desync is None:
                desync = i
            for s in STREAMS:
                diff = np.abs(obs_u[s].astype(np.float64) - obs_t[s].astype(np.float64))
                strict = [d for c, d in _columns(s, diff) if (s, c) not in BY_DESIGN]
                row[f"{s}_max"] = float(max(d.max() for d in strict)) if strict else 0.0
                if desync is None:
                    for c, d in _columns(s, diff):
                        st = col_stats.setdefault((s, c), [0, 0, 0.0])
                        st[0] += 1
                        st[1] += int((d > args.tol).any())
                        st[2] = max(st[2], float(d.max()))
            rows.append(row)
            i += 1
            if args.max_decisions and i >= args.max_decisions:
                break
            obs_u, _, done_u, _ = env.step(heads)
            try:
                dec = gen.send(halves)
                done_t = False
            except StopIteration:
                done_t = True
            if done_u or done_t:
                if done_u != done_t:
                    end_differs = True
                    print(f"episode end differs: Unity done={done_u}, twin done={done_t} after {i} decisions")
                break
    finally:
        env.close()

    with open(out / "parity_decisions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "parity_columns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "column", "by_design", "decisions", "mismatched", "max_abs_error"])
        for (s, c), (n, bad, mx) in sorted(col_stats.items()):
            w.writerow([s, c, int((s, c) in BY_DESIGN), n, bad, f"{mx:.3g}"])

    compared = desync if desync is not None else len(rows)
    strict_bad = {k: v for k, v in col_stats.items() if k not in BY_DESIGN and v[1]}
    masks_bad = sum(not r["mask_match"] for r in rows[:compared])
    print(f"\n{len(rows)} decisions; compared {compared} before "
          + (f"the first time gap (decision {desync})" if desync is not None else "the end (no time gap)"))
    print(f"action masks: {compared - masks_bad}/{compared} identical")
    for (s, c), (n, bad, mx) in sorted(col_stats.items()):
        if (s, c) in BY_DESIGN:
            print(f"  by design  {s}[{c}]: {bad}/{n} decisions differ, max {mx:.3g}")
    if strict_bad:
        print("STRICT MISMATCHES:")
        for (s, c), (n, bad, mx) in sorted(strict_bad.items()):
            print(f"  {s}[{c}]: {bad}/{n} decisions, max {mx:.3g}")
    else:
        print(f"every other column identical within {args.tol} at every compared decision")
    # A multi-AGV run may drift in time (desync), and then may end at a different decision too.
    end_ok = not end_differs or (args.agvs > 1 and desync is not None)
    if not end_ok:
        print("EPISODE END MISMATCH: Unity and the twin ended at different decisions")
    ok = same_jobs and not strict_bad and masks_bad == 0 and (args.agvs > 1 or desync is None) and end_ok
    print("PARITY", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
