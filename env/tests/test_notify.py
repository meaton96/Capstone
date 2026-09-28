"""
@file test_notify.py
@brief Tests for env/notify.py: webhook posting (Discord / Slack payloads, silent failure) and the stall watchdog.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import notify  # noqa: E402

ENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _capture_server():
    """@brief A local HTTP server that records each POST body; returns (server, received list)."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_slack_and_discord_payloads():
    server, received = _capture_server()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert notify.notify("hello slack", url=base + "/services/T/B/x")
        assert notify.notify("hello discord", url=base + "/discord.com/api/webhooks/1/abc")
    finally:
        server.shutdown()
    assert received[0][1] == {"text": "hello slack"}
    assert received[1][1] == {"content": "hello discord"}


def test_no_url_and_bad_url_never_raise(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "WEBHOOK_FILE", tmp_path / "missing")
    assert notify.webhook_url() is None
    assert notify.notify("nothing configured") is False
    assert notify.notify("unreachable", url="http://127.0.0.1:9/nope", timeout=2) is False


def test_url_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    f = tmp_path / "hook"
    f.write_text("https://discord.com/api/webhooks/1/abc\n")
    monkeypatch.setattr(notify, "WEBHOOK_FILE", f)
    assert notify.webhook_url() == "https://discord.com/api/webhooks/1/abc"


def test_watchdog_exits_a_stalled_process():
    """@brief A process whose main thread stops beating exits with code 3, and the alert is posted."""
    server, received = _capture_server()
    code = ("import sys, time; sys.path.insert(0, %r); import notify; "
            "w = notify.Watchdog(0.02, 'test-run', status=lambda: 'phase X', check_seconds=0.2); "
            "time.sleep(30)") % ENV_DIR
    env = dict(os.environ, NOTIFY_WEBHOOK_URL=f"http://127.0.0.1:{server.server_port}/hook")
    try:
        proc = subprocess.run([sys.executable, "-c", code], env=env, timeout=20, capture_output=True, text=True)
    finally:
        server.shutdown()
    assert proc.returncode == 3
    assert received and "STALLED" in received[0][1]["text"] and "phase X" in received[0][1]["text"]


def test_watchdog_quiet_while_beating():
    w = notify.Watchdog(0.02, "busy-run", check_seconds=0.1)
    import time
    for _ in range(20):          # 2 s of beats, well past the 1.2 s limit
        w.beat()
        time.sleep(0.1)
    w.stop()                     # reaching here means the process was not killed
