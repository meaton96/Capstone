# rq2-twin-blocks (+ -ctrl): do long, multi-regime episodes create in-episode switching headroom? (twin, 2026-10-04)

**Setup.** H15 greedy oracle on the event-based twin, tardiness (late-WIP integral), c ~ U[1.75, 2.5], random warm-up,
failures off, 7 AGVs, 900 s switch slots (4 per hour in every setting), seeds 0-39 per setting:
- `single`: one regime per episode, load base / utilhi 50:50, 5,400 s window (the current training setup);
- `blocks`: regime blocks of 5,400 s (c, load profile, op mean redrawn per block), same load mix, 21,600 s window;
- `long` (ctrl): one regime, same load mix, 21,600 s window;
- `blocks-base` (ctrl): regime blocks at base load only, 21,600 s window.
`run.py`, `../rq2-twin-blocks-ctrl/run.py`; `analyze.py ../rq2-twin-blocks-ctrl` -> `analysis.out`, `per_seed.csv`.
switch_% = oracle vs the seed's own best fixed pair (in-episode headroom).

| setting | switch_% median | seeds > 2% | saved per hour (job-s) | best-pair tardiness per hour (job-s) |
|---|---|---|---|---|
| single (1.5 h) | 1.43 | 42.5% | 414 | 23,300 |
| long (6 h, one regime) | 4.87 | 87.5% | 2,080 | 46,400 |
| blocks (6 h, 4 regimes) | 5.76 | 90.0% | 2,851 | 50,500 |
| blocks-base (6 h, 4 regimes, base load) | 8.39 | 87.5% | 2,130 | 29,000 |

**Findings.**
1. **Episode length is what creates the headroom**, at the same switch density (4 slots per hour): one regime over
   6 h already gives a 4.9% median (vs 1.4% over 1.5 h), on 87.5% of seeds.
2. **Regime blocks add a little on top** at the same load mix (blocks 5.8% vs long 4.9% median; saved per hour 2,851
   vs 2,080). The oracle's choices do not follow the block labels: job-rule shares are nearly the same in every c
   tercile, and the pair changes as often within a block (28-29%) as at a boundary (27%). The rule choice follows the
   state as it evolves (backlog), not the regime parameters.
3. **Caveat: the backlog grows in every 6 h setting.** Best-pair tardiness per hour, hours 1-6: blocks 21.7 -> 77.8,
   long 21.1 -> 67.1, blocks-base 14.5 -> 42.9 (thousand job-s). Even base load (utilization 0.7-1.4 outside lulls)
   is not sustainable over 6 h; the 5,400 s windows never showed it. Part of the long-episode headroom may be managing a
   plant that is falling behind. Next: recalibrate the load so per-hour tardiness levels off, then re-measure.
4. Per-block hindsight selection explains about half the blocks gain (blocksel 2.7% of 5.8%; blocks-base 5.4% of 8.4%).
