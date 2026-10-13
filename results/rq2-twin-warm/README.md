# rq2-twin-warm: is the short-window deficit a warm-up transient? (2026-10-04)

1.5 h windows after a warm-up of >= 2 h / >= 3 h (`randomized_generator(min_warmup_seconds=...)`), calibrated load,
seeds 0-39, oracle of `../rq2-twin-fleet/run.py`. Switch_% median 1.81% / 2.54% vs 2.38% (normal warm-up) and 8.17%
for 6 h windows, although tardiness per hour is high (25,000 job-s). **No: the window length itself matters** (likely
an end-of-window effect: consequences after the window are not counted). Write-up: `../rq2-twin-fleet/README.md`,
"Control".
