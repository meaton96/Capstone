# train-due-twin-slot-ext (10-06)

Baseline seeds 3-4 of train-due-twin-slot (identical command, 100k agent steps; run dirs `../train-due-twin-slot_s3`, `_s4`), so the action-prior comparison has 5 seeds per arm. Results and discussion: `../train-due-twin-slot-prior-kl/README.md` (baseline arm). Held-out: s3 +0.81% mean gap to MDD-TECT, s4 +6.21% (worst of the five baseline seeds).
