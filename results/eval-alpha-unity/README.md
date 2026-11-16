# eval-alpha-unity: paper B's α policy and the transport law in Unity, through the Python rule mode (2026-10-30, user: "Set up plan §3.2: the Unity check of the static law + gate through the Python rule mode on B2 and one large floor, with pre-declared criteria, in the lab")

Status: **done 11-01 17:36 (lab `eval-alpha-unity`, 308 / 308 tasks, 560 episodes). K1-K5 PASS, U1-U4 PASS; N: paper
B states the static law + gate and the law alone as Unity results on B2 and at 100 machines.** The machine powered
off at 10-31 19:15 mid-run; requeued 11-01, and `task.py` reran the unfinished tasks. At 16 players a 100-machine
episode took a median 89 min (max 178), a 10-seed B2 chunk 61-95 min. Both dry runs passed (below). Criteria pre-declared 10-30, before any Unity result of the policy. The dry runs (below) checked the
plumbing on seed 0 only and are not counted.

## Why
- Paper B's α policy stays the static law + gate (`../eval-alpha-online`, 10-30): train-alpha-off-het (v7: the law's
  α × an offset head) with the transport law of `../rq3-alpha-transport` and the size gate of
  `../eval-alpha-transport-size` (above 50 machines the job and machine heads are fixed to MDD-PTECT; only the offset
  head chooses).
- Every result for it is from the twin. Plan §3.2 asks for the Unity check through the Python rule mode before paper B
  reports it.
- Two cells:
  - **B2**, the 15-machine floor every policy is judged on. Its law α is 3, so the law alone is MDD-PTECT itself, and
    the check is the full policy (twin: −1.0% vs MDD-PTECT, n.s.).
  - **One large floor, m100_r052**: 100 machines, 52 AGVs on duty (the size grid's ratio). This is where the law
    acts (α 1.431, h 0.477) and the gate and offset head act (twin: law + gate −4.25% [−6.71, −1.87], p 0.006; law
    alone −2.90%).
  - m100_r052 is preferred to m100_r035 and m200_r052, which have larger twin gains. There the law plays α 0, so every
    offset gives the same α, and the policy would not be tested at all.
- Earlier Unity checks found a Unity premium at 100 machines (`../agv-ratio-unity`: +7-21% mean time in system at the
  twin knee with 240 s operations, congestion). This check says whether the law's gain survives it.

## Design (`task.py`, `launch.sh`, `analyze.py`, `code/`)
- **v7 in Unity, no C# change.** The player's agent has two discrete branches; v7 has three.
  - `task.py` sets the Unity env's branch list to the first two.
  - It pads Unity's action mask with the offset part as the twin builds it: all five offsets where the machine head
    matters, else offset 0 only.
  - It plays every three-head action in the Python rule mode (player `-pyrules`, `env/rules_api`). The job and machine
    heads name the built-in Python ports. `AlphaPair` sets the view's PECT / PTECT α to law × offset (capped at 4, v7's
    `offset_alpha`) before it picks.
  - A bare two-head "MDD-PTECT" still goes to C# at the episode's α 3.
  - The Python ports equal the C# rules in Unity bit for bit (`../dev-pyrules`: 18 / 18, incl. MDD-PTECT), and check
    K1 below repeats that through this harness.
- **Code** (`code/STAMP`): the v7 arm's frozen code, with three files from the main env at 09f53cbb.
  - `env_wrappers/unity_env.py`: vs the v7 copy, only the Python rule mode additions.
  - `channels/channels.py`: adds `send_pick` and `DecisionViewChannel`.
  - `rules_api/`.
  - Nothing else differs.
- **Player:** `linux_server/` (Simulation.dll `22891261...`, the resume player, checked per task). B2 and the size grid
  have machine failures off, so it plays as the 5e6e0f87 player of the earlier Unity checks; K2 checks that.
- **Instances, window, slots and reward** are the twin runs' own, built by the same code:
  - B2: `../rq3-gen-v5` "base". `../eval-alpha-offset`'s off-het group ran it through `hetero.py` at spread 0, which
    leaves the instance untouched.
  - m100_r052: `../rq3-size-grid`'s m100 (B2 load × 0.9, arrival cap off, 100 bays) with the fleet wrapper (52 on
    duty, 26 in short blocks), as `../eval-alpha-transport-size`.
  - Both: random warm-up, 21,600 s window, 900 s slots, discount horizon 5,400 s, test seeds 0-39.
  - σ = 0 is asserted per instance, so the law is a cell constant: α_law = round(3 h, 3), 3 on B2 and 1.431 on
    m100_r052.
- **Policies, one Unity task per policy × seed chunk:**

  | key | B2 | m100_r052 |
  |---|---|---|
  | ck0-ck4 | train-alpha-off-het s0-s4, full policy | the same, gated (`../eval-alpha-only-size`'s AlphaOnlyPolicy) |
  | pdr | MDD-PTECT, C#, α 3 (the reference) | MDD-PTECT, C#, α 3 (the reference) |
  | py3 / law | MDD-PTECT@3 in Python (check K1) | MDD-PTECT@x1 in Python = the law alone at α 1.431 |

  - B2 runs in 10-seed chunks; m100_r052 in 1-seed tasks, since an episode there takes far longer.
  - 280 + 28 = 308 tasks, 560 episodes in all.
  - Each task has its own player port, from its index in the task list. `task.py` refuses a port already in use:
    gRPC binds with SO_REUSEPORT, so two processes on one port both start and their players can cross-connect. That
    happened once in the dry run and was caught.
- **Twin side (existing runs, same seeds):**
  - B2: `../eval-alpha-offset/out/0/off-het` (MDD-PTECT, MDD-PTECT@x1 = α 3, s0-s4).
  - m100_r052: `../eval-alpha-transport-size/out/m100_r052/lg_s*` (MDD-PTECT, gated s0-s4). The law alone comes from
    `../rq3-alpha-online/out/m100_r052/law/` ("static", MDD-PTECT at α 1.431; that simulator equals evaluate.py there,
    checked in eval-alpha-online).
  - The final policy in Unity on B2 (train-v5-cmix s3) is `../eval-v5-cmix-unity/runs/base/pc` (reported).

## Analysis (`analyze.py`, pre-declared 10-30 before any Unity result of the policy)
- **Metric:** window tardiness, % vs MDD-PTECT in the same simulator, pooled over test seeds 0-39, 95% instance
  bootstrap CI, better / worse, Wilcoxon ("significant" = p < 0.05). The policy is the family of 5 (per-instance mean
  of s0-s4).
- **Checks (all must pass before the criteria are read; otherwise fix and rerun):**
  - **K1, α plumbing:** on B2, MDD-PTECT@3 in Python equals C# MDD-PTECT on every seed (window tardiness, jobs exited,
    decisions, makespan, instance hash).
  - **K2, instances and player:** on B2, C# MDD-PTECT equals `../eval-v5-cmix-unity`'s base rows (player 5e6e0f87) on
    every seed, with the same instance hash.
  - **K3, the gate:** on m100_r052 every checkpoint slot plays MDD-PTECT on the pair heads.
  - **K4:** every policy and Python-rule task picked in Python (count > 0).
  - **K5:** no deadlock, time-out or tick-error episode.
- **Criteria (Unity):**
  - **U1, no loss on B2:** the family is not significantly worse than MDD-PTECT.
  - **U2, the policy's gain holds at 100 machines:** on m100_r052 the family is significantly better than MDD-PTECT.
  - **U3, the law's gain holds at 100 machines:** on m100_r052 the law alone (MDD-PTECT at α 1.431) is significantly
    better than MDD-PTECT.
  - **U4, size of the transfer:** on m100_r052 the family's Unity % keeps at least half of its twin % on the same seeds
    (Unity % ≤ 0.5 × twin %).
- **Reported, not gating:**
  - each checkpoint;
  - Unity − twin of each %, with a joint seed-bootstrap CI;
  - MDD-PTECT Unity vs twin (the Unity premium);
  - what the offset head plays by fleet state, Unity vs twin;
  - on B2, the family vs the final policy in Unity.
  - (Unity's ML-Agents episode rows carry no collision or stall counts; K5's deadlock and time-out flags are the
    safety check here, as in the earlier Unity policy checks.)
- **N, the next step:**
  - U1-U4 PASS → paper B states the static law + gate and the law alone as Unity results on B2 and at 100 machines.
    No further Unity work for paper B.
  - U3 PASS, U2 or U4 FAIL → the law holds in Unity at 100 machines but the offset head's extra does not. Paper B's
    Unity claim is the law alone there; the policy is a twin result at 100 machines.
  - U3 FAIL → the transport term does not carry to Unity at 100 machines. Report the law as a twin result and
    diagnose with the MDD-PTECT Unity premium (congestion) before any further α work at scale.
  - U1 FAIL (any case) → the policy costs on B2 in Unity: paper B reports it as a twin result on B2.
- **Expectation, stated before the run (a guess, not a criterion):**
  - B2 tracks the twin within noise, as every earlier B2 Unity check did (eval-v5-cmix-unity: Unity − twin +0.1 to
    +2.2 pp at base).
  - At 100 machines Unity congestion raises every tardiness. A lower α favours shorter trips, so the law's gain may
    grow rather than shrink, but this is the first policy run in Unity at 100 machines.

## Dry runs (10-30; seed 0; not counted)
- **B2** (scratchpad; pdr, py3, ck0 at once, 204-211 s per episode):
  - K1: Python MDD-PTECT@3 = C# MDD-PTECT (94,388 job-s, 1,708 decisions, the same makespan and instance hash).
  - K2: C# MDD-PTECT = eval-v5-cmix-unity's base row (instance 6711812f52868218).
  - ck0 ran end to end: 1,708 decisions, all picked in Python at law × offset (MDD-PTECT@1.5 in most slots).
  - The first attempt put two tasks on one player port. gRPC let both start, so their players could cross-connect;
    those two were stopped and rerun. `task.py` now refuses a port in use, and dry runs take a port from their pid.
- **m100_r052** (lab `dev-alpha-unity-dry`, `dry.sh` → `dry/`, `dry/dry_check.out`; 4 players at once, 2,850-2,887 s per
  episode):
  - ck0 with the offset forced to ×1 under the gate equals MDD-PTECT@x1 in Python (the law alone) exactly: 34.1 k
    job-s, 1,465 jobs, 8,399 decisions. The gate and the α mapping hold in Unity.
  - Every measured decision of the Python tasks was picked in Python. The 92 extra picks per task are Unity's startup
    default episode (5 machines), which evaluate.py plays with the all-zero action and discards. K4 counts picks ≥
    decisions for that reason.
  - All 24 gated slots played MDD-PTECT on the pair heads (K3).
  - One seed, for orientation only: MDD-PTECT 40.1 k (twin 38.2 k), the law alone −14.8% (twin −5.3%), ck0 −5.5% (twin
    −11.0%).
- "single parking" in the player's spawn line is only AGVPool's label for "not multiple". The scenario asks for lane
  parking, which the pool applies first, as the twin floors (`parking_method` lane) assume.

## Cost
- m100_r052: 280 episodes × ~48 min (4 players at once) ≈ 224 player-h.
- B2: 280 episodes × ~3.5 min ≈ 16 player-h.
- 16 players at once (each ~1.7 cores, ~0.3 GB at start): ~15-18 h wall, 100-machine tasks first.

## Run
`launch.sh [W]`: `task.py list` → `tasks.txt`, W players at a time, the 100-machine tasks first. Finished tasks are
skipped. `ALL_DONE` is written when every task has its output.

## Outputs
- `out/<cell>/<policy>_s<a>-<b>/`: `episodes.csv`, `decisions.csv`, `eval.out`, `Player.log`, `cond.json` (the law, the
  Python pick counts), `blocks.json`, `DONE`;
- `analysis.out`, `summary.csv`, `played.csv`.

## Results (`analysis.out`, `summary.csv`, `played.csv`; 11-01)
Window tardiness, % vs MDD-PTECT in the same simulator, seeds 0-39, [95% CI], better / worse seeds, Wilcoxon p.

**Checks.** All pass, so the criteria are read.
- K1: on B2, Python MDD-PTECT@3 equals C# MDD-PTECT on 40 / 40 seeds (tardiness, jobs exited, decisions, makespan,
  instance hash).
- K2: on B2, C# MDD-PTECT equals eval-v5-cmix-unity's base rows (player 5e6e0f87) on 40 / 40 seeds, so the resume
  player plays the failure-free floors exactly as before.
- K3: on m100_r052, 0 of 4,800 gated checkpoint slots left MDD-PTECT on the pair heads.
- K4: 264 / 264 policy and Python-rule tasks picked every measured decision in Python.
- K5: no deadlock, time-out, or tick-error episode.

**Criteria (Unity, family of 5).**

| | Unity | twin (same seeds) | Unity − twin |
|---|---|---|---|
| U1 B2, family | −0.16 [−2.08, +1.57] 22/18 p 0.56 → PASS | −1.01 [−3.07, +0.62] p 0.12 | +0.84 pp [−0.83, +2.67] |
| U2 m100_r052, family | **−4.81 [−7.49, −2.33] 31/9 p 0.00026** → PASS | −4.25 [−6.74, −1.90] p 0.006 | −0.56 pp [−3.05, +1.35] |
| U3 m100_r052, the law alone (MDD-PTECT@1.431) | **−4.20 [−6.54, −2.13] 34/6 p 7.5e-06** → PASS | −2.90 [−4.14, −1.88] p 9e-06 | −1.29 pp [−3.60, +0.69] |
| U4 m100_r052, family keeps ≥ half its twin % | −4.81 ≤ −2.13 → PASS | | |

- **B2:** the family ties MDD-PTECT in Unity, as in the twin. Each checkpoint lies within −1.1..+0.7% in Unity
  (twin −2.7..+1.2%), none significant. Beside the final policy (train-v5-cmix s3, eval-v5-cmix-unity, +0.50%
  [−2.56, +3.53] in Unity) the family is −0.66% [−2.78, +1.39], p 0.81: a tie.
- **100 machines:** every checkpoint is significantly better than MDD-PTECT in Unity, −3.0% to −6.5% (twin −2.7% to
  −5.6%), and keeps its twin rank (ck3 best, ck1 worst in both). The law alone gains more in Unity than in the twin
  (−4.2 vs −2.9%), and the offset head adds about 0.6 points on top of it in Unity (1.4 in the twin). Neither
  difference from the twin is significant.
- **The Unity premium at 100 machines is large but does not hurt the ranking.** MDD-PTECT's window tardiness is +50.4%
  [+39.9, +64.1] above the twin on all 40 seeds (685.9 k vs 456.0 k job-s), while its window time in system is +11.5%
  and jobs exited −0.8%. The time-in-system premium is within `../agv-ratio-unity`'s +7-21% (congestion); tardiness
  grows faster because it counts only the time past each due date, so a uniform delay pushes many jobs over it. On B2
  the premium is +1.26% [−0.99, +3.18], n.s.
- **What the offset head plays is the same in both simulators** (`played.csv`): on m100_r052 ×0.5 in 66-68% of slots,
  played α 0.85-0.88 in Unity vs 0.83-0.87 in the twin; on B2 ×0.5 in 58-67%, played α 2.20 (full fleet) and 1.86-1.87
  (short) in both.

**Reading (rule N).** U1-U4 pass: the static law + gate and the law alone hold in Unity on B2 (a tie, no loss) and at
100 machines (−4.8% and −4.2%, both at least as large as in the twin). Paper B states both as Unity results. No
further Unity work for paper B. Limits: one large floor (m100_r052; the law's α 0 floors were not run, since there
the policy is MDD-TECT), one fleet ratio, and the failure-free floors.
