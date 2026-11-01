# CapstoneBuild-resume/linux_server_dev: build of 2026-10-28 15:25 UTC (preempt-resume machine failures)

Installed in `linux_server/` on 2026-10-28 (user 10-28: land failure-resume, install the resume player). Notes of the
build it replaces (pyrules, Simulation.dll `5e6e0f87`, 10-11) are in `linux_server_archive/notes/linux_server_pyrules_1011/`;
that player is also in git at df03817d.

- **Source:** worktree `~/Documents/Github/Capstone-resume` (branch `failure-resume`) = main df03817d plus the
  uncommitted failure-resume change, now applied to the main checkout (uncommitted as of 10-28). No C# changed on main
  between the pyrules player (71e34bdb) and df03817d, so this build = the pyrules player + the failure-resume change.
  `git diff df03817d -- Capstone/Assets Capstone/Packages Capstone/ProjectSettings | sha256sum` =
  `54d6407f054ed5365c929199c02c3fcdd2ad8639742e0bc691fc51b20dee19e2`.
- **How:** headless from a copy without Library/ (`~/Documents/Github/CapstoneBuild-resume/Capstone`; script
  `results/dev-failure-resume/build.sh`), `DOTNET_USE_POLLING_FILE_WATCHER=1`, `LinuxServerBuild.BuildDev`:
  "[LinuxServerBuild] Succeeded (0 errors)", 0 "is missing". BUILD_MANIFEST.json says "commit unknown" (no git in the
  copy): this file is the provenance. Installed by copying `capstone_Data/`, the Burst debug folder and
  BUILD_MANIFEST.json; `capstone.x86_64`, `UnityPlayer.so` and the libdecor libraries were byte-identical. Results/,
  BatchConfigs/, slurm/ and the scripts kept.
- **Simulation.dll:** md5 `22891261e82b72d3bb3dc0fbc3764e38` (pyrules player `5e6e0f8713e186a3c5555975ce0fa2d9`).
- **New vs the pyrules player:** a machine failure no longer restarts the operation in process. The job stays on the
  failed machine with its remaining time frozen and finishes the rest after the repair (preempt-resume); the finish
  moves back by R - F + 1 ticks, the work before the failure counts as busy time. Launch flag `-failurerestart` restores
  the old restart model. Failure-free runs are unchanged.
- **Verified** (`results/dev-failure-resume/`, lab `dev-failure-resume`, 10-28): 24/24 resume cells exact against the
  twin (event for event); 3/3 `-failurerestart` cells exact against the twin's `failure_restart` and every CSV identical
  to the pyrules player; 7-AGV native-sampler cells with `-failurerestart` and failure-free identical to the pyrules
  player.
- Not glibc-patched (local use). The cluster copy (`~/capstone/linux_server`) is this build glibc-patched
  (`slurm/patch_glibc.py`, memory: RIT cluster glibc patch).
