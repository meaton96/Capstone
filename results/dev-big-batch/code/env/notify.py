"""
@file notify.py
@brief Run notifications (Discord or Slack incoming webhook) and a stall watchdog for long training jobs.

@details
Why: rnd02 (2026-09-26) failed at startup and then sat in RUNNING for 25 h on three nodes; nothing reported it.
Slurm's own Slack notices (RIT: --mail-user=slack:@<user>, see slurm/submit_train.sh) cover BEGIN / END / FAIL,
but not a job that is alive and doing nothing. This module adds:
  - notify(text): post one message to a webhook. Never raises; a failed post is printed and ignored.
  - Watchdog: a daemon thread that posts an alert and hard-exits the process (exit code 3) when beat() has not
    been called for stall_minutes, so a hung job frees its node and Slurm reports FAIL.

The webhook URL comes from $NOTIFY_WEBHOOK_URL, else the first line of ~/.capstone_webhook (keep it chmod 600;
never commit it). No URL = notifications off (the watchdog still works). Discord URLs
(discord.com/api/webhooks/...) get {"content": ...}; anything else is treated as a Slack incoming webhook
({"text": ...}).
"""

import json
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

WEBHOOK_FILE = Path.home() / ".capstone_webhook"


def webhook_url() -> Optional[str]:
    """@brief The configured webhook URL, or None."""
    url = os.environ.get("NOTIFY_WEBHOOK_URL", "").strip()
    if not url:
        try:
            lines = WEBHOOK_FILE.read_text().strip().splitlines() if WEBHOOK_FILE.is_file() else []
        except (OSError, UnicodeDecodeError) as e:   # unreadable file: notifications off, never a crash
            print(f"[notify] cannot read {WEBHOOK_FILE}: {type(e).__name__}")
            lines = []
        url = lines[0].strip() if lines else ""
    return url or None


def run_label(run_id: str) -> str:
    """@brief "<run> (job <id>_<task> on <host>)" for message headers."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    where = f"job {job}_{task}" if job and task else (f"job {job}" if job else "local")
    return f"{run_id} ({where} on {socket.gethostname()})"


def notify(text: str, url: Optional[str] = None, timeout: float = 10.0) -> bool:
    """@brief Post @p text to the webhook. Returns True on success; never raises."""
    try:
        url = url or webhook_url()
        if not url:
            return False
        body = {"content": text[:1900]} if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url \
            else {"text": text[:3900]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "capstone-train"})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception as e:   # network down, blocked egress, malformed URL: training (and the watchdog) must not care
        # Only the type: the message of a URL error can contain the secret webhook URL, and this lands in Slurm logs.
        print(f"[notify] post failed: {type(e).__name__}")
        return False


class Watchdog:
    """@brief Hard-exits the process when beat() stops being called (a hung env step, a stuck startup).

    @details Starts on construction. The first stall window covers startup (players launching), so pick
    stall_minutes above the slowest normal gap: startup of every env, and the longest PPO update.
    """

    def __init__(self, stall_minutes: float, label: str, status: Optional[Callable[[], str]] = None,
                 check_seconds: float = 30.0):
        self.stall_seconds = stall_minutes * 60.0
        self.check_seconds = check_seconds
        self.label = label
        self.status = status
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stall-watchdog", daemon=True)
        if self.stall_seconds > 0:
            self._thread.start()

    def beat(self):
        self._last = time.monotonic()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.check_seconds):
            idle = time.monotonic() - self._last
            if idle > self.stall_seconds:
                detail = self.status() if self.status else ""
                msg = (f":warning: **STALLED** {self.label}: no progress for {idle / 60:.0f} min "
                       f"(limit {self.stall_seconds / 60:.0f}). Exiting with code 3 so the job frees its node. {detail}")
                print(f"[watchdog] {msg}", flush=True)
                try:
                    notify(msg)
                finally:
                    os._exit(3)   # the main thread is stuck, so a normal exception would never be seen
