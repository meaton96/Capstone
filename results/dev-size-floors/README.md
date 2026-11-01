# dev-size-floors

Layout-D floor exports (M machines, M bays) at 20 / 30 / 50 / 75 / 100 / 150 / 200 / 1000 machines from the current player `linux_server/` (5e6e0f87) for the floor-size study (`../rq3-size-train/README.md`). `make_floor_scenarios.py` (5-job scenarios), `export_floor.sh M`, `launch.sh`, `compare.py` (summary, and new m50 / m100 vs the v4 exports of `../agv-ratio-scale`). m20: 120 s by hand (10-19).

## Results (done 10-19 23:40)
All 8 floors exported, each with the observation frame:

| machines | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 1000 |
|---|---|---|---|---|---|---|---|---|
| zones | 101 | 146 | 247 | 357 | 449 | 670 | 887 | 4,198 |
| footprint x × z (m) | 24 × 22.5 | 30 × 30 | 42 × 45 | 48 × 60 | 54 × 67.5 | 72 × 82.5 | 84 × 97.5 | 186 × 232.5 |
| export wall time | 120 s | 150 s | 118 s | 193 s | 432 s | 660 s | 761 s | see below |

- **The new 50 / 100-machine exports are identical to the v4 exports** used by rq3-size-v5 and the training arms: no
  differing field.
- **1000 machines:** `des_floor.json` was written within a minute of the start (19:51). The player then spent 3.8 h
  simulating the 5 placeholder jobs on 1000 machines / 1000 AGVs, and was stopped by hand at 23:39; the export
  itself was complete.
