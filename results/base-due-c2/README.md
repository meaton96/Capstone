# base-due-c2: fixed-pair baseline on tardiness (Unity, c = 2, 20 seeds x 21 pairs; 2026-10-03)

**Setup.** `eval_task.sh` per seed: `evaluate.py --pdr all` (the 21 pairs of the action schema v3 job head SPT / SRT /
PTWINQ / FIFO / EDD / MDD / ATC x ECT / TECT / SRWT), randomized generator with `due_date_allowance` 2, random warm-up,
5,400 s window, tardiness reward, player `linux_server_due/`. 4 seeds ran in queue-1003; 16 failed on evaluate.py's
120,000 s player budget and reran 10-03 21:00 (`rerun_1003.sh`) after dev-clock-check removed the budget.
`analyze.py` -> `analysis.out`, `pairs.csv`.

**Result** (tardiness over the window, 1000 job-s; gaps to the per-seed best pair):

| pair | mean tardiness | mean gap | median gap | wins | time-in-system gap |
|---|---|---|---|---|---|
| **MDD-TECT** | 54.71 | **2.73%** | 0.20% | **8** | 1.29% |
| MDD-ECT | 55.11 | 5.39% | 3.30% | 3 | 1.48% |
| SRT-TECT | 55.69 | 5.58% | 1.90% | 2 | 0.92% |
| SRT-ECT | 55.68 | 6.75% | 3.25% | 3 | 0.92% |
| EDD / ATC pairs (ECT, TECT) | 61.9-62.8 | 13-16% | 11.5-15.3% | 4 (ATC-ECT) | 4.8-5.2% |
| PTWINQ-ECT | | 19.7% | | 0 | |
| FIFO pairs | | 55-73% | | 0 | |

- MDD-TECT is the best fixed pair on tardiness; the SRT pairs are close and have the lowest time in system.
- PTWINQ and FIFO never win (with the oracle's choices, the reason they left the RL head in action schema v4).
- **Consistency:** the 300 pairs shared with rq2-oracle-due stage 1 (seeds 0-19) equal it to 2.8e-14.
