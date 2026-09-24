#!/usr/bin/env python3
"""Resumable, grid-wide sweep runner: one worker pool across a whole experiment grid.

run_gridlock_sweep.sh pools workers within one (scenario, layout) invocation, so long cells leave cores idle at the
end of each batch. This expands the full grid (scenario x layout x AGV count x rule x seed) into one queue, runs the
longest cells first, and skips any cell whose results.csv already exists, so a sweep can be stopped and restarted
(or survive a reboot) without redoing finished work. One process per seed: each seed gets a seed-shifted copy of
the scenario (the player runs seed = scenario seed + repeat index, so -repeats 1 on a copy with seed+k reproduces
exactly what -repeats N would have run as repeat k).

Output: Results/<exp>/<scenario>/<layout>/agv<N>_<RULE>_s<seed>/{results,agv_performance,...}.csv + sim.log

Usage (from linux_server/):
  python3 run_experiment_queue.py --exp E1_load --workers 12 \
      --scenarios _mfsweep_control:1 _mfsweep_shard0:5 compound_scenario:1 compound_scenario_fail:5 \
      --layouts D C G J --agv 3 5 7 9 12 15 --rules SPT_SRWT LRT_MMUR \
      --extra "-reservation holdPrevious -parking lane"
  (scenario:N = run N seeds of that scenario)  add --dry-run to print the plan only.
  --shard i/N splits the grid across N array tasks/nodes (see slurm/run_queue.sbatch).
  --exe ../linux_server_e2/capstone.x86_64 runs a second build; its results go to that build's own Results/.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN_DIR = os.path.join(HERE, "BatchConfigs", "Scenarios")
RESULTS = None   # <exe dir>/Results: the player writes there (LoggingInit), so a second build keeps its own tree


def cost(scenario_json):
    """Relative wall-time estimate (last arrival x failures) for longest-first ordering."""
    d = json.load(open(scenario_json))
    last = max(j["arrivalTime"] for j in d["jobs"])
    return last * (3.0 if (d.get("stochastic") or {}).get("machineFailuresEnabled") else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True, help="name[:seeds] (file in BatchConfigs/Scenarios)")
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--agv", nargs="+", type=int, required=True)
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timescale", default="100")
    ap.add_argument("--loglevel", default="Low")
    ap.add_argument("--extra", default="")
    ap.add_argument("--exe", default="./capstone.x86_64")
    ap.add_argument("--shard", default="0/1", help="i/N: run only every Nth cell of the (stable, longest-first) grid, so N "
                    "Slurm array tasks split one experiment across nodes; membership ignores which cells are already done")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global RESULTS
    exe = os.path.abspath(os.path.join(HERE, a.exe))
    RESULTS = os.path.join(os.path.dirname(exe), "Results")

    seed_dir = os.path.join(RESULTS, a.exp, "_scenarios")
    os.makedirs(seed_dir, exist_ok=True)
    cells = []
    for spec in a.scenarios:
        name, _, n = spec.partition(":")
        n = int(n or 1)
        src = os.path.join(SCEN_DIR, name + ".json")
        base = json.load(open(src))
        c = cost(src)
        for k in range(n):
            seed = base.get("seed", 42) + k
            copy = os.path.join(seed_dir, f"{name}_s{seed}.json")
            if not os.path.exists(copy):   # write-then-rename: shards on other nodes may read it concurrently
                tmp = f"{copy}.{os.getpid()}.tmp"
                json.dump(dict(base, seed=seed), open(tmp, "w"))
                os.replace(tmp, copy)
            for layout in a.layouts:
                for agv in a.agv:
                    for rule in a.rules:
                        rel = f"{a.exp}/{name}/{layout}/agv{agv}_{rule}_s{seed}"
                        cells.append((c * (1 + agv / 30), rel, copy, layout, agv, rule))
    shard_i, _, shard_n = a.shard.partition("/")
    shard_i, shard_n = int(shard_i), int(shard_n or 1)
    if not 0 <= shard_i < shard_n:
        sys.exit(f"--shard {a.shard}: need 0 <= i < N")
    cells.sort(key=lambda x: (-x[0], x[1]))          # stable order, identical on every shard
    total_cells = len(cells)
    cells = cells[shard_i::shard_n]
    todo = [x for x in cells if not os.path.exists(os.path.join(RESULTS, x[1], "results.csv"))]
    todo.sort(key=lambda x: -x[0])   # longest first
    print(f"[Queue] exe {exe} -> results {RESULTS}", flush=True)
    print(f"[Queue] {a.exp}: shard {shard_i}/{shard_n} = {len(cells)} of {total_cells} cells, {len(cells) - len(todo)} already done, {len(todo)} to run, "
          f"{a.workers} workers", flush=True)
    if a.dry_run:
        for x in todo[:10]: print("  ", x[1])
        return

    def run(x):
        _, rel, scen, layout, agv, rule = x
        out = os.path.join(RESULTS, rel); os.makedirs(out, exist_ok=True)
        cmd = [exe, "-batchmode", "-nographics", "-scenario", scen, "-agvcount", str(agv), "-rules", rule,
               "-repeats", "1", "-timescale", a.timescale, "-loglevel", a.loglevel, "-layout", layout,
               "-outputdir", rel, "-logFile", os.path.join(out, "sim.log")] + a.extra.split()
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=os.path.dirname(exe), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        ok = os.path.exists(os.path.join(out, "results.csv"))
        return rel, rc, ok, time.time() - t0

    done = 0
    t_start = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(run, x) for x in todo]):
            rel, rc, ok, dt = fut.result(); done += 1
            eta = (time.time() - t_start) / done * (len(todo) - done) / 3600
            print(f"[Queue] {done}/{len(todo)} {'ok ' if ok else 'FAIL'} {dt/60:5.1f} min  {rel}  (eta {eta:.1f} h)",
                  flush=True)
    print(f"[Queue] finished {a.exp} in {(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    sys.exit(main())
