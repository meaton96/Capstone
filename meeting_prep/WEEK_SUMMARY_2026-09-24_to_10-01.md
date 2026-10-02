# Week summary: 2026-09-24 to 2026-10-01

Covers everything from the 8-composite-PDR gap tests onward. The order follows the argument: (1) why the old action
space failed, (2) what replaced it, (3) training on it, (4) floor/AGV experiments, (5) new model features, and
(6) what is still running or waiting on a decision.

Status as of **10-02 03:00** (updated overnight; the 10-01 afternoon version is superseded). Live status is in
`docs/WIP/RUNNING_EXPERIMENTS.md`. **New since the afternoon:** sections 0, 2 (incl. new 2.1 the reward problem and
2.2 bugs), 3 (twin rows), 4, 5, 6, 7, 8.

---

## 0. TL;DR

1. **The legacy 8 composite PDRs could not support RL.** On the randomized training distribution, 20-32% of decisions
   are identical under all 9 actions, and even the 8-rule *segment oracle* is 2.6% worse than the best new fixed rule.
2. **New rules plus a two-head action space.** ECT/TECT machine rules win every seed (2-14.5% better than the best
   legacy rule). Job head {SPT, SRT, PTWINQ, FIFO} × machine head {ECT, TECT, SRWT}, with masking.
3. **Training on the two-head space did not learn, and the runs are cancelled (10-01 23:50).** The latest checkpoints
   (1.3-1.8M steps) always pick SPT; rnd02_s1 *is* SPT-SRWT. Training flow was flat from 0 to 2M steps.
4. **Bigger problem found: the reward and the thesis target measure different things.** The reward is time in
   system of every job (WIP integral). On that measure SRT-TECT is best and only **0.4%** from the per-seed best
   fixed pair. The thesis target "beat SPT-ECT on total flow (8.1% headroom)" uses a censored metric (finished jobs
   only) that the reward does not optimize. **Switching oracle (done 10-02 02:48):** switching pairs every 900 s
   with full foresight gains only **0.54%** over the best fixed pair per seed and **~1%** over SRT-TECT. **On this
   training mix there is not enough headroom to retrain for.** The meeting question is where to go instead (§7).
5. **Bugs found and fixed or explained (10-01):** a float32 clock in the AGV zone-retry timers changes results after
   ~131,000 s of player life (affects long training players; fix not applied yet), and the truncation bootstrap
   used V(all-zero obs) (fixed). Fixes also landed for the per-decision discount (now over simulated time) and
   BatchNorm (now GroupNorm).
6. **Sim-to-real (RQ1, environment half) answered by the DES twin.** The twin reproduces Unity exactly with 1 AGV.
   Instant transfers (the usual DRL abstraction) understate flow by 35%+ and rank rules differently wherever
   transport is the bottleneck; free-flow vehicles stay within a few % until the fleet nears capacity (then D
   congests to 8.7× at 25 AGVs). On the machine-bound training mix the gap is within about ±3.5%.
7. **AGV breakdowns (RQ4, fixed-rule half) done:** at ~2 / 6 / 11 breakdowns per episode, mean flow per job +1 to
   +7%, rule ranking unchanged, each breakdown costs the fleet ~3× its repair in queueing. 420 runs, 0 deadlocks.
8. **AGV/floor study closed on the protocol side:** releasePrevious default, 0 collisions / 0 deadlocks in 864 + 756
   + 400 runs; knee at 7 AGVs per 15 machines; layout effects only in transport-bound regimes.
9. **Thesis:** ch1 edited (intro citations, contributions, RQ3/RQ4 reworded), AGV breakdown model in ch4, results in
   ch6.

---

## 1. Why we swapped the action space (8 composite PDRs → two heads)

### 1.1 8-rule baseline on the new randomized generator (09-24 → 09-25)
- Built `env/scenarios/randomized.py`: 15 machines, 7 AGVs, failures on in 2/3 of episodes (λ=7200 s, ~10 min repair),
  varied op length and job mix, 4 h of arrivals with a 90-min agent window and a random warm-up.
- **First baseline (`Results/gen_baseline`, seeds 0-8):** the SRWT rules are the top 3 on every seed, MMUR is 1.3-2× worse, and SMPT is 2-4×
  worse. The per-machine speed factor makes SMPT a "speed trap". **Segment-oracle headroom was only 0.1-3.4%.**
- **Cost-structure tuning (`gen_tune`):** any fixed per-machine speed difference keeps SMPT a trap. Per-op affinity
  makes the rules competitive, with headroom at 2.6%.
- **Load retune (`gen_load`, → `rnd_load_s0-8`, now the generator defaults):** op 240-480 s, utilization 0.7-1.4,
  lulls 0.2-0.4. The share of degenerate dispatch (≤1 candidate) fell from 58-98% to 28-70%. Job-order spread became SMPT +57% and SRWT +17%.
  Oracle headroom is 5.0% on average (0.9-10.4%). 0 deadlocks and 0 collisions.

### 1.2 MMUR isolation (09-24, existing data)
- LPT_SMPT vs LPT_MMUR isolates the machine half. They are identical in 5/8 scenarios, MMUR is 7-40% worse on compound and +66% on compound_v2.
- Mechanism: `MachineUtilization` is the lifetime busy fraction. It lags and ignores queued and in-transit work, so an early-idle
  machine keeps attracting jobs. This is not a bug, just a poor signal.

### 1.3 What the old policy actually learned (ep_warmup01, 09-24)
- The greedy choice was SPT-SMPT on 98% of decisions, and the probabilities were nearly flat (.27/.21/.21). The "1.9% gap" was a
  fixed rule, not switching.
- **Observation v1 was broken:**
  - The job matrix showed the first 20 jobs *ever*.
  - Only 8 of the 15 machines were encoded.
  - There was no machine-health signal: a repairing machine looked idle.
  - The scalars saturated.
- → **Observation schema v2** (09-24): a 100×16 machine table, a 64×17 active-job table, and set encoders. On 09-27 the row caps became per-launch, auto-sized
  from the scenarios (15 machines = 16,902 floats). v1 checkpoints are refused.

### 1.4 Action aliasing in the 9-rule space (09-26, `gen_load` decision logs)
| decision type | share | distinct outcomes among 9 actions |
|---|---|---|
| dispatch, ≤1 queued job | 12-31% | 1 |
| dispatch with a real choice | 18-30% | 5 (job rules only) |
| routing, machine choice only | 35-46% | 3 (SMPT/SRWT/MMUR) |
| routing, machine + job pool | 4-15% | up to 9 |

Every decision went to the agent as a 9-way categorical with no masking. This explains the flat probabilities.

### 1.5 Rule catalog sweep: 19 rules (`gen_rules0926`, 09-26, 171 runs)
- New rules:
  - **ECT** = machine load + p.
  - **TECT** = max(travel, load) + p.
  - **PTWINQ** = p + min load at the next op.
- Fix: `GetMachineLoad` now includes the in-process op's remaining time. This changes SRWT, the obs load feature, and the failure redirect.
- **ECT/TECT win every seed.** The best new rule beats the best legacy rule by 2-14.5% (mean ~8%).
- Mean gap to the per-seed best: SPT_TECT +3.1%, SPT_ECT +3.2%, SRT_ECT +4.2% … best legacy SRT_SRWT +12.2%, MMUR +130-166%.
- **Headroom by action set** (segment oracle vs best single rule of all 19):

| action set | oracle |
|---|---|
| legacy 8 (old RL space) | **+2.6%** (can't even reach the best fixed rule) |
| {SPT,SRT,PTWINQ}×{ECT,TECT} | −3.9% |
| job half only (machine = ECT) | −3.3% |
| **{SPT,SRT,PTWINQ,FIFO}×{ECT,TECT,SRWT}** | **−5.5%** |
| all 19 | −5.7% |

- Most of the switching headroom is in the **job** half.
- Writeups: `docs/experiments/RULE_CATALOG_0926_findings.md`, `docs/features/DECISION_POINTS.md` §7, and the notebook
  `results/notebooks/action_space_decisions.ipynb`.

### 1.6 Decision: option C, two factorized heads (implemented 09-26)
- Two ML-Agents discrete branches: job head {SPT, SRT, PTWINQ, FIFO} × machine head {ECT, TECT, SRWT}.
- Per-decision masking of unused heads. The log-prob and entropy cover the used heads only. Checkpoints carry `action_layout`.
- Legacy catalog rules (22 JOB_MACHINE combos) are still available for baselines and warm-up.

### 1.7 New reference baselines (`results/eval_pdr12_heads`, 09-27, 240 episodes)
- 12 head pairs × seeds 0-19, through `evaluate.py` (the same path the checkpoints are scored on).
- **SPT-ECT is the best on average: +8.1% vs the per-seed best.** Wins are spread over 9 of the 12 rules, and the worst rule averages 25.8% behind the best per seed.
  → There is real per-instance headroom for an adaptive policy. **Target: beat SPT-ECT.** (**10-01:** this is on the
  censored total-flow metric; on the reward's own metric the headroom is 0.4% and the best pair is SRT-TECT, see §2.1.)
- Note: the rank correlation with the batch sweep is only 0.06-0.87, so score checkpoints against *this* set.

---

## 2. Training runs

| run | when | status / result |
|---|---|---|
| rnd01 | 09-25 | failed at `import numpy` (spack PYTHONPATH); fixed in `train.sbatch` |
| rnd02 (job 21786794) | 09-26 | never trained: submitted from the wrong player dir (one-branch player), hung 25 h; unity_env now closes the player on a failed startup |
| **rnd02** (21787413) | 09-28 → 10-01 | 3 seeds × 3M steps, ent 0.01→0.001, ~5.5-6 SPS; s0/s2 forked to rnd02ent, s1 **cancelled 10-01 at ~1.8M** |
| **rnd02ent** s0, s2 | 09-30 → 10-01 | forked from rnd02 s0 (571k) and s2 (551k) with **constant ent 0.02**; **cancelled 10-01 at ~1.5M / ~1.35M** |

**Interim greedy evals** (`compare_to_baselines.py`, seeds 0-19, total flow):

| checkpoint | vs per-seed best | vs SPT-ECT | behaviour |
|---|---|---|---|
| s0 @ 96k | +17.3% | +8.7% | collapsed to SPT-SRWT (weakest machine rule) |
| s0 @ 499k | +8.1% | **+0.1%** | became pure SPT-ECT |
| s1 @ 96k | +13.6% | +5.1% | SPT-TECT |
| s1 @ 518k | +10.6% | +2.5% | mixes ECT/TECT/PTWINQ, beats SPT-ECT on 10/20 seeds; only seed with per-instance adaptivity |
| s2 @ 77k | +12.9% | +4.5% | SPT + ECT/TECT mix |
| s2 @ 480k | +13.6% | +5.1% | became pure SPT-TECT |

- Training curves are flat (mean flow ~1650-1780 s), entropy oscillates 0.2-0.8 (no collapse to 0), and value loss is stable.
- **Reading:** the policies find the best fixed rule rather than learning state-dependent switching. rnd02ent raised s0's entropy
  0.46→0.89.
**Final check of the latest checkpoints (10-01, seeds 0-19, clean clock):**

| checkpoint | free job decisions | free machine decisions | vs SRT-TECT on the reward's metric |
|---|---|---|---|
| rnd02_s1 @ 1.80M | SPT 100% | SRWT 100% (= SPT-SRWT exactly) | +9.9%, better on 0/20 seeds |
| rnd02ent_s0 @ 1.47M | SPT 100% | ECT 52 / TECT 37 / SRWT 11% | +5.3%, 0/20 |
| rnd02ent_s2 @ 1.32M | SPT 100% | ECT 80 / TECT 20% | +4.6%, 0/20 |

Training mean flow was flat (~1,700-1,750 s) from 0 to 2M steps on every seed. None of them ever picks SRT, the
better job rule on the reward's metric. The job head is free on only ~27% of decisions, the machine head on ~52%.
**All three runs and their queued continuations were cancelled 10-01 23:50.**

### 2.1 The reward does not optimize the thesis target (found 10-01)

The `flow_time` reward is −(time in system of every job, finished or not)/1000, i.e. the WIP integral. The thesis
scores policies on total flow time of *exited* jobs, which is censored: a rule that finishes fewer jobs leaves its
slowest ones uncounted.

| pair (20 held-out seeds) | gap to per-seed best, **reward's metric** | gap, total flow of exited jobs | jobs finished |
|---|---|---|---|
| SRT-TECT | **0.4%** | 9.5% | 51.6 |
| SRT-ECT | 0.8% | 9.0% | 51.4 |
| PTWINQ-ECT | 4.4% | 9.8% | 49.4 |
| SPT-ECT (current thesis target) | 5.1% | **8.1%** | 49.0 |
| FIFO-SRWT | 16.1% | 18.2% | 44.4 |

- A policy that learns the reward perfectly should look like SRT-TECT, which *loses* to SPT-ECT on the thesis metric.
- Picking the best fixed pair per seed gains only 0.4% on the reward's metric. That is not enough headroom.
- The open question is **mid-episode switching**, never measured on this metric. The earlier segment oracle (−5.5%)
  was on censored total flow and ignored state carry-over.
- **`rq2-switch-oracle` (done 10-02 02:48, 1,440 episodes):** greedy and fully simulated. 6 segments of 900 s; at
  each stage keep the chosen prefix and try all 12 pairs for the rest of the episode. Monotone (never worse than the
  best fixed pair), and every schedule is a real run. Stage 1 reproduced the clean baseline exactly on all 240 runs.

| comparison (time in system, 20 seeds) | gain |
|---|---|
| best fixed pair per seed vs SRT-TECT | 0.4% |
| switching oracle vs best fixed pair per seed | **0.54%** (median 0.15%, best seed 2.7%; >1% on only 5 seeds) |
| switching oracle vs SRT-TECT | **0.96%** |

  - The schedules are mostly SRT with the machine rule flipping between ECT and TECT. The job rule is nearly settled.
  - It is a lower bound (coarse segments, greedy), but it has full foresight, which a policy does not. A learned
    policy on this mix would likely get less than 1%, about the size of the evaluation noise.
  - Writeup: `docs/experiments/rq2-switch-oracle_findings_1002.md`.

### 2.2 Bugs found this week that affect training

| issue | effect | status |
|---|---|---|
| **Float32 clock**: `AGVController` zone-retry, stall and block timers use `Time.fixedTime`, never reset | after 131,072 s of player life (the 25th 5,400 s episode) rounding moves 0.05 s retries by one physics step and trajectories diverge. Training players run millions of seconds, so the training env drifts with player age (past ~3,100 episodes retries happen every step) | proven 10-01: seed 2 matches the old baseline exactly as the 25th episode and the clean result as the 23rd/24th. **Fix not applied** (`fixedTimeAsDouble`); needs rebuild |
| **Truncation bootstrap** used V(all-zero obs) on every time-capped episode | wrong value target at the end of every training episode, all randomized runs | fixed 10-01 (last decision's obs), committed `5f5e86dd` |
| **Per-decision γ = 0.99** in a semi-MDP (decisions unevenly spaced) | effective horizon ~1,370 s, varies 2× between instances | fixed 10-01: discount over sim time, 3,000 s horizon, committed `3f41192b` |
| **BatchNorm** in the grid CNN, eval() rollouts vs train() updates | PPO ratio ≠ 1 before any update (16-17% of samples outside the clip at init) | fixed 10-01: GroupNorm, committed `3f41192b` |
| Reward vs evaluation metric (2.1) | training cannot succeed by the thesis's stated test | **decision needed** |

Checked and fine: action masking in log-prob/entropy, GAE, per-episode reseeding of Unity's global RNG, observation
noise/dropout (disabled), no history dependence below 131,072 s.

### 2.3 Infra built along the way

- RIT SPORC pipeline: `train.sbatch` with resume, a graceful SIGTERM checkpoint, and `submit_train.sh` with ARRAY/DEPENDENCY.
- glibc patch for the Unity player.
- Webhook/Slack notifications plus a stall watchdog.
- `train_status.sh` / `pull_training.sh`.
- Per-episode `episodes.csv` flush.

---

## 3. Experiments run (floor, AGVs, scale)

| exp | date | size | key finding | writeup |
|---|---|---|---|---|
| **E2** pass-through input corner | 09-24 | 48 smoke | siding (v1) worse; bypass (v2) cuts corner waiting 20-90% but flow is worse in every cell (detour, floor 12 units wider). releasePrevious is the better lever | `E1_E2_findings.md` §6 |
| **E4 pilot** m100 monolithic | 09-25 | 2 | 100 machines / 33 AGVs on D: AGVs 99% busy, machines 6%; flow 4,799 s vs ~300 at 15 machines → transport-bound | memory / findings |
| **E1_rel3** (cluster) | 09-26 | 864 | releasePrevious: **0 collisions / 0 deadlocks**, 3-15 AGVs, all layouts; knee = 7 AGVs everywhere; J best → **release is now the default** | `E1_E2_findings.md` §6b |
| **Tiled floor** phase 1 | 09-26 | regression + pilot | 7 tiles × 15 machines, 35 AGVs: flow 181 s vs 4,799 monolithic; tiles:1 byte-identical | `TILED_LAYOUT_SCOPE.md` §9 |
| **E6** two-way perimeter K-O | 09-28 | 144 | girth-safe, 0 collisions; **every K-O slower than D** (path length 72-95 s vs 54 s, not congestion); J still best; linked K-O 3.8× worse | `E4_E6_…_0928.md` §3 |
| **Linked tiles** + **E4S** scale | 09-28/29 | 72 + 32 | isolated tiles scale flat; linked floors degrade ~linearly with tiles; only SPT_TECT @ 7/tile stays flat (183→200 s); SRWT 10-18× worse; past 7/tile, congestion at spine-merge corners | `E4_E6_…_0928.md` §2, §5 |
| **AGVR** (15 layouts on rnd_load) | 09-29/30 | 756 | all layouts within ~4% (noise) → **rnd_load is machine-bound** (util 0.81, AGV travel 34%); layout effects need transport-bound scenarios; MMUR 2.6× SPT_ECT | §6, §6.1 |
| **AGVR_hold** | 09-30 | 126 | hold = release ≤ 9 AGVs; at 12-15 hold stalls/deadlocks (MMUR 11/18 deadlock), release 0. **Use penalized flow** (deadlocked runs look faster) | §7 |
| **FLEX** (cluster, partial) | 09-28 → | 222 analysed (of 756) | flexibility p 0.15/0.3: flow −11/−23% (m1.0), −4/−8% (m1.25); job rule matters less (FIFO 20%→5%), SRWT penalty grows (to 20%); ECT≈TECT; SPT_ECT still best | `FLEX_findings_0930_partial.md` |
| **AGVF_check** / **check5** | 09-30 | 18 + 18 | AGV breakdowns: 0 collisions/deadlocks/stalls, distributions validated (KS p=0.77) | gap plan D3 |
| **twin-dock** (V1) | 10-01 | 20 | event twin reproduces Unity exactly with 1 AGV: every event and decision identical, 0.0000% flow diff | `twin-dock_findings_1001.md` |
| **twin-gap** (G1) Unity vs DES twin | 10-01 | 400 | instant transfers (DES-0): flow 35-173% too low on transport-bound mixes even with ample AGVs, and ranks rules differently (SPT-ECT vs Unity's SRT-TECT); free-flow DES-1k within 1-5% at 5-15 AGVs, then D compound collapses at 25 AGVs (8.7× trips, rule ranking inverts); rnd_load within ±3.5% to 15 AGVs; trip inflation >1 whenever >1 AGV | `twin-gap_findings_1001.md` |
| **rq4-agvfail** (AGVF) | 10-01 | 420 | see §4 | `rq4-agvfail_findings_1001.md` |

Other changes from these experiments:
- **routingTrigger onTransport** (09-25, advisor-endorsed, now the default): routing waits for a free AGV and the rule ranks the waiting pool.
  It only matters when AGVs are scarce (pools >1 on 43% of decisions at 3 AGVs, ~1 at 7).
- The **input corner** (`LeftVert_TopConn`) is the dominant hotspot on every floor. On linked tiles it is also the spine merge.

---

## 4. AGV failures (new this week)

- **Found 09-30:** `agvFailuresEnabled` was an unimplemented stub, a silent no-op. It is now rejected by the Python schema until the new player is in use.
- **Design decisions (09-30):**
  - A failed AGV stops in place and keeps its zones for the repair.
  - A loaded job stays on board. Unloaded pickups are handed back via `StalledFlag`.
  - TTF counts operating time only.
  - Each AGV has its own RNG stream, and its first life is drawn from the equilibrium residual distribution.
  - Waiting behind a broken AGV is exempt from the 180 s stall timer.
- **Validation (09-30):** `-validatestochastic` passed 16/16. `validate_agv_failures.py` gave a repair KS p=0.77 and life counts O/E in line.
  The pairing check passed (0 failures → byte-identical to off).
- **Calibration finding:** 7 AGVs average only ~36% busy (13.5k operating-s per 5,400 s episode), so:

| setting | breakdowns / episode | flow vs off (7 AGVs) | flow vs off (5 AGVs) |
|---|---|---|---|
| default λ=8400 | ~1.7 | ±1.5% (noise) | +2.1% (sd 3.9) |
| stress λ=1500 | ~10 | +6.7% (sd 4.0) | +4.3% (sd 1.8) |

- Queue time behind a broken AGV is ~3× the repair time. Stress hurts less at 5 AGVs because fewer AGVs pile up.
- **rq4-agvfail dose-response (10-01, 420 episodes, 20 held-out seeds, D, 7 AGVs, paired vs breakdowns off):**

| scale λ (op-s) | breakdowns / ep | others queued behind (s / ep) | Δ mean flow per job: SPT-ECT / SRT-TECT / FIFO-SRWT | Δ jobs finished |
|---|---|---|---|---|
| 8400 (default) | 2.0 | 766 | +1.4 / +1.3 / +0.1% | about 0 |
| 3000 | 6.4 | 2,222 | +5.2 / +3.1 / +0.7% | −0.3 to −1.0 |
| 1500 | 11.5 | 4,362 | +5.0 / +7.3 / +2.7% | −1.2 to −1.6 |

  - Ranking unchanged: SRT-TECT best on mean flow at every rate (its lead over SPT-ECT narrows 4.2% → 1.8%),
    FIFO-SRWT last.
  - Each breakdown costs the rest of the fleet ~3× its repair in queueing; flow barely moves because AGVs are ~36% busy.
  - Per-seed total flow swings −15% to +30%: one breakdown changes every later decision, so read the 20-seed means.
  - Written up: `docs/experiments/rq4-agvfail_findings_1001.md`, thesis ch4 (model) and ch6 (`tab:agv-breakdowns`).
  - Not run yet: the trained policy under breakdowns (RQ4 second half); waits on the retrain.
- **Machine-failure audit (09-24):**
  - TTF runs on calendar time.
  - A failed op restarts rather than resuming.
  - The initial age is Uniform(0, T) rather than the stationary residual, so first failures come ~30% early. This is documented as a bias.
  - Machine and AGV draws used to share one RNG. They are now separate.

---

## 5. Currently running / queued (checked 10-01 23:55; oracle row 10-02 03:00)

| item | where | progress | ETA / action needed |
|---|---|---|---|
| **rq2-switch-oracle** (§2.1) | local `linux_server` | **done 10-02 02:48**: ~1% headroom over SRT-TECT | retrain on this mix not worth it; see §7 |
| **rq3-flex** (FLEX) | cluster 21794034-38 (re-split into 9 small tasks per group) | 590 / 756 cells | p0.15 / p0.3 ~10-02 06:00; p0.5 m1.0 ~10-02 14:00; p0.5 m1.25 ~10-03 14:30 (Slurm estimates) |
| **Retrain** | cluster | rnd02 / rnd02ent cancelled 10-01 23:50 | waits on the regime / metric decision (§7); needs rebuild (float-clock fix) + env sync |
| **twin-transfer** (RQ1 policy half) | local twin + cluster Unity | code written and tested (Python); nothing trained | needs rebuild + parity check + a transport-bound generator variant |
| **rq3-gen** (conditions sweep) | local | 0 | baselines can run now; policy rows wait on the retrain |
| **Girth sweep A2** (not started) | — | 0/360 | D + J × 16-30 AGVs; needs A1 counters + rebuild |

---

## 6. Thesis / writing

- **09-28:** thesis updated with the post-09-22 findings: E1_rel3, two-way F-J/K-O, E2, tiles/linked, rule catalog, two-head, obs v2,
  rnd02 interim, and flexibility. All edits use review-color markup (`\showchangestrue`).
- **09-30:** simulated review panel added ~90 `\revnote` comments on ch1-5. The main gaps it raised:
  - RQ1 has no DES comparison → the DES twin work.
  - "3D physics" is claimed but the physics engine is unused.
  - Due dates appear in the domain model but not in the code.
  - The PPO BN issue.
- **Gap plan (`docs/THESIS_GAP_PLAN_2026-09-30.md`):**
  - A, girth bound: below girth (20 on D, 16 on J), deadlocks are guaranteed absent by construction, so E1's
    "0 deadlocks" is not empirical evidence. Sweep above the bound. **Not started.**
  - B, sim-to-real reframed as a "modeling gap, not transfer". **Done in text.**
  - C, due dates removed (kept in the literature only). The EDD example became ECT. **Done.**
  - D, AGV failures. **Implemented, validated, dose-response done and written up (10-01).**
- **10-01 edits (all in review markup):**
  - Ch1 intro: suggested citations added (Graham 1979, Pinedo 2022, Destouet 2025, Zhang 2025, Ngwu 2025); the
    sim-to-real sentence trimmed.
  - Ch1 contributions: event-twin item rescoped (8.7× is compound on D at 25 AGVs); PPO item rewritten with the
    pipeline specifics and **written as the target outcome, with `\todo` placeholders for the final numbers**.
  - RQ3 reworded with operational conditions; RQ4 restored to machine *and* AGV breakdowns, split into a fixed-rule
    half and a trained-policy half.
  - Ch4: AGV breakdown model paragraph. Ch6: breakdown dose-response table and text, plus a `\todo` on the baseline
    mismatch (now explained by the float clock). Conclusion: RQ4 fixed-rule answer.
  - **Will need revisiting** once the metric decision is made (§7): the 8.1% target, the 25.8% spread and the
    segment-oracle table are all on the censored total-flow metric, and so is the PPO contribution item.
- **§7.1 security fixes (09-29, committed):**
  - The gRPC server binds to loopback.
  - Strict config validation rejects bad configs (exit 3), with an AGV-fleet bound.
  - Config/instance hashes are written to results.
  - `weights_only` checkpoint loading (all 105 checkpoints load).
  - Hash-locked RIT requirements (torch 2.14 cpu, for the CVEs).
  - A build manifest is checked by train/evaluate.

---

## 7. Decisions / questions for the meeting

1. **Evaluation metric for the RL result.** Switch from total flow of exited jobs (censored, not what the reward
   optimizes) to mean time in system including unfinished jobs (the reward's own quantity)? This changes the target
   from SPT-ECT to SRT-TECT and means recomputing the thesis headroom numbers.
2. **The switching oracle shows ~1% headroom on the training mix. Where should RL go?** Options:
   - train where rules matter more (transport-bound floors, flexibility, heavier failures), checking each with the
     oracle first (~3 h per 20-seed regime locally). Candidates from existing data: the transport-bound compound
     mix (SRT-TECT vs SPT-ECT swap), FLEX p0.3 m1.25 (fixed-rule spread widest), AGV breakdowns at λ 1500;
   - reframe RL as cross-regime generalization: the best fixed pair changes between regimes, so one policy near the
     best everywhere beats any single fixed pair across the RQ3 set (measurable from fixed-rule runs alone);
   - direct job×machine actions (more headroom, much bigger change this late).
3. **Retrain plan once 1-2 are settled (not before):** rebuild with the float-clock fix, sync env (bootstrap, SMDP discount,
   GroupNorm), fresh runs (no warm start: the old job head is locked on SPT). About two months left, so several runs
   are possible.
4. **AGV breakdown rate for the thesis:** report the dose-response (the realistic ~2/episode is invisible; stress
   ~11 gives +5-7% mean flow). Also run the trained policy under breakdowns once it exists?
5. **Open-scope pins on linked floors:** should a pin expand to the equivalent machine in every tile? Linked compound
   cells are confounded until decided (78% of routing wait is pinned phases). Rerun ~5 h.
6. **Layout chapter framing:** layouts are irrelevant when machine-bound (the training mix) and matter when
   transport-bound; twin-gap supports this. Unseen-layout evaluation must use transport-bound scenarios.
7. **Girth sweep A2** (360 runs): run alongside the rebuild?
8. **Spine merges:** express lane / move belt docks off merge corners, or future work?

---

## 8. Housekeeping

- **Uncommitted:** `env/switch_oracle.py` (new 10-01), the 10-01 C# changes for the twin (`DesTwinExport`,
  `ObservationBuilder`, `FactoryOrchestrator`), and the `results/rq4-agvfail/` and `results/rq2-switch-oracle/` outputs.
- **Player directories:**
  - `linux_server`: AGV-failure build (09-30); free to rebuild (rq2-switch-oracle done).
  - `linux_server_des`: twin-gap (done).
  - `linux_server_dev`: free.
  - Cluster `~/capstone/linux_server`: the rnd02 player (pre-security, pre-AGV-failure); nothing training on it now.
- **Cluster vs local:** rebuild and sync the player and env together for the retrain (an old player with new env
  code fails at startup); the RIT venv must be rebuilt for the hash lock.
- **Baselines and batching:** keep `evaluate.py` runs to ≤ 24 episodes per player until the float-clock fix, and
  compare against `results/rq4-agvfail/reg` (clean on all 20 seeds), not `eval_pdr12_heads` (clean on 10 of 20).
- **Pre-09-25 PDR numbers are `onReady` routing and pre-`GetMachineLoad`-fix.** Don't compare SRWT across that boundary.
- Tests: from `env/`, run `../.venv/bin/python -m pytest tests --ignore=tests/test_channels.py`.
- Commits this week: `a3355fc2` obs v2 → `38780030` generator → `8c538a8d` routingTrigger → `2ea28fc1` two-head + PDR
  rework → `b69d1f42` 12-PDR testing → `23cabd9e` dynamic obs → `e3f73bea` flexibility → `2735f05c` linked tiles →
  `ee6f9467` two-way spines → `b87cf925`/`373adc29`/`7c317587` security → `92ebeb22`/`5d3d5293` AGV failures →
  `5f5e86dd` twin training backend + truncation bootstrap → `3f41192b` SMDP discount + GroupNorm.
