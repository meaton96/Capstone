# rq2-oracle-due-csweep: Unity tardiness oracle at c = 1.5 / 2.5 / 3 (2026-10-03 22:10 - 10-04 ~06:10)

Same oracle as rq2-oracle-due (H15, 6 x 900 s, random warm-up, failures as generated, 7 AGVs), seeds 0-7 per c;
outputs in `../rq2-oracle-due-c{1.5,2.5,3}/` (`analysis.out`, `per_seed.csv`). c = 2 is rq2-oracle-due seeds 0-7.

| c | median gain vs best-on-average pair | mean | seeds > 2% | switching alone (median) | best-pair tardiness (1000 job-s) | saved per window (job-s) |
|---|---|---|---|---|---|---|
| 1.5 | 2.43% (vs MDD-ECT) | 4.90% | 4/8 | 1.13% | 70.8 | 2,259 |
| 2.0 | 3.43% (vs MDD-TECT) | 5.69% | 5/8 | 1.43% | 51.8 | 1,992 |
| 2.5 | 19.44% (vs SRT-ECT) | 18.70% | 5/8 | 1.52% | 38.6 | 3,216 |
| 3.0 | 7.44% (vs MDD-TECT) | 23.29% | 4/8 | 0.39% | 27.2 | 1,888 |

- **Same trend as the twin (rq2-twin-scen):** tight c (1.5) gives the least headroom; looser c gives more, but mostly
  because the best fixed pair changes from seed to seed (at c = 2.5 the lowest-mean pair is SRT-ECT, yet MDD-TECT has
  the smallest mean gap; at c = 3 ATC-ECT has the smallest gap, MDD-TECT is 57% off on average). Switching within an
  episode stays at a 0.4-1.5% median at every c. A policy that reads the state can use both.
- **Rule use moves with c:** SRT 52% of oracle slots at c = 1.5 and 2.5, MDD 50% and ATC 15% at c = 3.
- **c = 3 is noisy** (mean 23% vs median 7%): small tardiness on some seeds inflates percentages.
- 8 seeds per c: indicative only. Supports the training mix c ~ U[1.75, 2.5] (rq2-oracle-due-mix checks it).
