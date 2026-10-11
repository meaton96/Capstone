# linux_server_v5: build of 2026-10-10 21:54 (PECT / PTECT machine rules, v5 RL heads 3 x 2, rule override)

The previous v5 build (2026-10-10 00:30, layout-A single belt placement, Simulation.dll md5
`9f68bba9bbc62b9029ae52ee73ee17d7`) is kept, runnable, in `archive_1010/` with its own BUILD_NOTES.md and a copy of
BatchConfigs/. The 10-09 build is still in `archive_1009/`. BatchConfigs/ and Results/ stay here (BatchConfigs is
identical in all three builds).

- **Source:** worktree `agent-aba7760f4b3a0f0d6` (branch `worktree-agent-aba7760f4b3a0f0d6`): `action-prior` dd0b3092
  plus commit c557f5f7 (the dev-pect-unity changes).
  dd0b3092 has the same C# as 824dcad4, the source of the build now in archive_1010 (`git diff 824dcad4 dd0b3092 --
  Capstone/Assets Capstone/Packages Capstone/ProjectSettings` is empty). So this build = the archived 10-10 build + the
  dev-pect-unity C# changes only. `git diff dd0b3092 c557f5f7 -- Capstone/Assets Capstone/Packages Capstone/ProjectSettings |
  sha256sum` at build time = `c8fd662449d73d889f202d5cfdb10748a93d96a35dc4ef2cdf6c7d2e3a5e58ba`.
- **How:** headless from a copy without Library/ (`~/Documents/Github/CapstoneBuild-pect/Capstone`, rsync excluding
  Library/ Temp/ Logs/ obj/ *.csproj), `DOTNET_USE_POLLING_FILE_WATCHER=1`, `LinuxServerBuild.BuildDev`;
  "[LinuxServerBuild] Succeeded (0 errors)", 0 "is missing" in the build log (results/dev-pect-unity/build.log).
  BUILD_MANIFEST.json says "commit unknown" (no git in the copy): this file is the provenance.
- **Simulation.dll:** md5 `73bc9bc8f1d632b466b469b16694b9c8`.
- **New vs the archived 10-10 build** (results/dev-pect-unity/README.md, user approval 10-10):
  - Machine rules **PECT** = q / A + p + alpha g(p) and **PTECT** = max(travel, q / A) + p + alpha g(p)
    (DispatchingEngine.PtimeScores, a port of env/des_twin/rules.py ptime_scores), as `<JOB>_PECT` / `<JOB>_PTECT`
    for every job half (DispatchingRule values appended, so old values keep their ints). Parameters: scenario /
    config-message keys `ptimeAlpha`, `ptimeShape` (lin / quad / ratio / exp), `ptimeTau`, `ptimeReroute`, `ptimeAvail`
    (FJSSPConfig.PtimeRule); default quad, alpha 3, as the twin. JsonUtility batch configs (-config) cannot set them
    (defaults).
  - **RL action heads v5:** job head {SRT, MDD, ATC}, machine head {PECT, PTECT} (branches 3 x 2; were 5 x 3).
    Python code with the old heads refuses this player (branch check in UnitySchedulingEnv), and new code refuses the
    archived players.
  - **Rule override:** EpisodeConfigChannel message `{"ruleOverride": "MDD_TECT"}` / `{"ruleOverride": null}` makes
    SchedulingAgent step that catalog rule instead of the policy's heads (evaluate.py fixed rules outside the heads).
  - des_floor.json gains `ptime_rule` only when a scenario sets non-default ptime parameters.
  - Configs that set no ptime key keep their config_hash (FJSSPConfig.PtimeRule is null and left out of the
    fingerprint).
- **Checks:** results/dev-pect-unity/README.md: old rules identical to the archived build file by file, 1-AGV exact
  parity of the new rules against the twin, des_twin.parity in RL mode, evaluate.py smokes.
