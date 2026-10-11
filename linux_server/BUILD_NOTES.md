# CapstoneBuild-pyrules/linux_server_dev: build of 2026-10-11 16:09 (Python rule mode, -pyrules)

- **Source:** worktree `agent-acd5ad3b37177ebd7` (branch `worktree-agent-acd5ad3b37177ebd7`): `action-prior` c196b006 plus
  commit 71e34bdb (the dev-pyrules C# changes). c196b006 has the same C# as the current v5 player in `linux_server/`
  (`git diff c557f5f7 c196b006 -- Capstone/Assets Capstone/Packages Capstone/ProjectSettings` is empty), so this build =
  the v5 player + the Python rule mode only.
  `git diff c196b006 71e34bdb -- Capstone/Assets Capstone/Packages Capstone/ProjectSettings | sha256sum` =
  `b78a0484622d552a9fec9a17705a8e7ce9e51a9ae228a6a70e8df73bf48229c2`.
- **How:** headless from a copy without Library/ (`~/Documents/Github/CapstoneBuild-pyrules/Capstone`, rsync excluding
  Library/ Temp/ Logs/ obj/ *.csproj; script `results/dev-pyrules/build.sh` in the worktree),
  `DOTNET_USE_POLLING_FILE_WATCHER=1`, `LinuxServerBuild.BuildDev`: "[LinuxServerBuild] Succeeded (0 errors)", no
  "error CS", 0 "is missing" in the build log (`../build.log`) and in the Player.log of every evaluate.py run.
  BUILD_MANIFEST.json says "commit unknown" (no git in the copy): this file is the provenance.
- **Simulation.dll:** md5 `5e6e0f8713e186a3c5555975ce0fa2d9` (the v5 player's is `73bc9bc8f1d632b466b469b16694b9c8`).
- **New vs the v5 player:** launch flag `-pyrules` (Python rule mode, `PyRules.cs`): each agent decision also sends its
  candidate view on a new side channel (DecisionViewChannel, GUID d4a5b6c7-e8f9-4a0b-9c1d-2e3f4a5b6c7d), and a
  `{"pick": ...}` config-channel message applies a Python rule's choice (FactoryOrchestrator.StepPick, validated; an
  invalid pick stops the player). Without `-pyrules` the player is bit-identical to the v5 player
  (results/dev-pyrules/README.md: decision logs and episode rows equal). RL action branches, observation and
  checkpoints unchanged.
- Not glibc-patched (local use). For the cluster, run the glibc patch as for every player (memory: RIT cluster glibc
  patch).
