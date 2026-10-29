#!/bin/bash
# dev-failure-resume (2026-10-28): headless player build from a copy of the failure-resume worktree's Capstone/ without
# Library/ (memory: headless-build-from-project-copy, unity-headless-build-inotify). Output: $B/linux_server_dev/.
B=/home/ubuntupc/Documents/Github/CapstoneBuild-resume
WT=/home/ubuntupc/Documents/Github/Capstone-resume
mkdir -p "$B"
rsync -a --delete --exclude Temp/ --exclude Logs/ --exclude obj/ --exclude Library/ --exclude '*.csproj' \
    "$WT/Capstone/" "$B/Capstone/"
cd "$B" || exit 1
DOTNET_USE_POLLING_FILE_WATCHER=1 DOTNET_hostBuilder__reloadConfigOnChange=false timeout 2400 \
    /home/ubuntupc/Unity/Hub/Editor/6000.3.15f1/Editor/Unity -batchmode -nographics -quit \
    -projectPath "$B/Capstone" -executeMethod Assets.Scripts.Editor.LinuxServerBuild.BuildDev -logFile "$B/build.log"
echo "unity rc=$?"
grep -n "LinuxServerBuild\] \|error CS" "$B/build.log" | head
echo "is missing: $(grep -c 'is missing' "$B/build.log")"
ls -la "$B/linux_server_dev/capstone_Data/Managed/Simulation.dll" && md5sum "$B/linux_server_dev/capstone_Data/Managed/Simulation.dll"
