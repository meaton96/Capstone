#!/bin/bash
# Sourced by the sync/pull scripts. Uses the ~/.ssh/config alias "rit-research" and shares one SSH connection
# across every ssh/rsync call (ControlMaster), so Duo prompts once and the connection stays open 30 min after last use.
REMOTE="${RIT_REMOTE:-rit-research}"
DEST="${RIT_DEST:-capstone/linux_server}"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%C" -o ControlPersist=30m)
SSH=(ssh "${SSH_OPTS[@]}")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
