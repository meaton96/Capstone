# train-v5-slot60

Training arm of the PPO slot-length sweep (10-21, user): the final policy's recipe (train-v5-cmix, frozen code
6d13412a, c ~ U[1.25, 3.5]) with `--slot-seconds 60` and 720 updates (equal simulated experience to cmix's 48 updates
at 900 s), seeds 0-2, local. Design, cost and evaluation: `../eval-v5-slot/README.md`. Lab: `train-v5-slot60`.

## Training (done 10-22 05:20, local)
All 3 seeds trained the full run and saved `checkpoint.pt` (`results/train-v5-slot60_s<k>/`). Wall time per seed
15.0-15.6 h (seeds in parallel, 2 twin workers each, the machine shared with other runs). Critic explained variance at the last
update +0.94 to +0.97. Episode returns in the logs are not comparable across seeds or slot lengths (different instances per
update, rewards per slot); the result is the evaluation in `../eval-v5-slot`.

