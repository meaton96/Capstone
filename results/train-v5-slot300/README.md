# train-v5-slot300

Training arm of the PPO slot-length sweep (10-21, user): the final policy's recipe (train-v5-cmix, frozen code
6d13412a, c ~ U[1.25, 3.5]) with `--slot-seconds 300` and 144 updates (equal simulated experience to cmix's 48 updates
at 900 s), seeds 0-2, local. Design, cost and evaluation: `../eval-v5-slot/README.md`. Lab: `train-v5-slot300` (queued
behind rq3-alpha-size).
