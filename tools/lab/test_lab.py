"""@file test_lab.py
@brief tools/lab/lab.py: local runs finish and fail correctly, futures are promoted, timers fire once, the web API
needs the token. Run: .venv/bin/python -m pytest tools/lab/test_lab.py -q"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "STATE", tmp_path / "state")
    monkeypatch.setattr(lab, "DB_PATH", tmp_path / "state" / "lab.db")
    monkeypatch.setattr(lab, "TOKEN_PATH", tmp_path / "state" / "token")
    monkeypatch.setattr(lab, "LOG_DIR", tmp_path / "state" / "logs")
    monkeypatch.setattr(lab, "STATUS_MD", tmp_path / "LAB_STATUS.md")
    monkeypatch.setattr(lab, "REPO", tmp_path)
    monkeypatch.setattr(lab, "notify", lambda text: True)
    return lab.connect()


def wait_tick(svc, con, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        svc.tick(con)
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_local_run_finishes_and_queues_the_next(con, tmp_path):
    lab.add_experiment(con, "dev-a", "queued", launch_cmd="mkdir -p out && sleep 0.5 && touch out/1 out/2",
                       done_glob="out/*", done_count=2, workers=1, analysis_cmd="ls out | wc -l")
    lab.add_experiment(con, "dev-b", "future", waits_on="dev-a", launch_cmd="true", workers=1)
    svc = lab.Service()
    assert wait_tick(svc, con, lambda: lab.get(con, "dev-a")["status"] == "done")
    ev = con.execute("SELECT * FROM events WHERE kind = 'finished'").fetchone()
    assert ev and "2" in ev["detail"] and ev["read"] == 0            # analysis output kept, in the inbox
    assert wait_tick(svc, con, lambda: lab.get(con, "dev-b")["status"] in ("running", "done"))
    rows = (tmp_path / "LAB_STATUS.md").read_text().splitlines()
    assert sum(r.startswith("| dev-a |") for r in rows) == 1


def test_process_ending_without_outputs_fails(con):
    lab.add_experiment(con, "dev-c", "queued", launch_cmd="exit 3", done_glob="nothing/*", done_count=1)
    svc = lab.Service()
    assert wait_tick(svc, con, lambda: lab.get(con, "dev-c")["status"] == "failed")


def test_free_text_wait_blocks_promotion_and_slots_limit_starts(con, monkeypatch):
    monkeypatch.setitem(lab.CONFIG, "max_local_workers", 2)
    lab.add_experiment(con, "dev-d", "done")
    lab.add_experiment(con, "dev-e", "future", waits_on="dev-d", waits_note="rebuild first", launch_cmd="sleep 5")
    lab.add_experiment(con, "dev-f", "queued", launch_cmd="sleep 5", workers=2)
    lab.add_experiment(con, "dev-g", "queued", launch_cmd="sleep 5", workers=1)
    svc = lab.Service()
    svc.tick(con)
    assert lab.get(con, "dev-e")["status"] == "future"
    assert lab.get(con, "dev-f")["status"] == "running" and lab.get(con, "dev-g")["status"] == "queued"
    lab.cancel(con, "dev-f")
    assert lab.get(con, "dev-f")["status"] == "cancelled"


def test_check_at_fires_once(con):
    lab.add_experiment(con, "dev-h", "future", check_at="2000-01-01T00:00:00")
    svc = lab.Service()
    svc.tick(con)
    svc.tick(con)
    assert con.execute("SELECT COUNT(*) FROM events WHERE kind = 'timer'").fetchone()[0] == 1


def test_web_writes_need_the_token(con, monkeypatch):
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"name": "dev-i", "status": "future"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/add", body, {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        req.add_header("X-Lab-Token", lab.token())
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state").read())
        assert [x["name"] for x in state["experiments"]] == ["dev-i"]
    finally:
        srv.shutdown()


def test_zero_slot_experiment_starts_when_slots_are_full(con, monkeypatch):
    monkeypatch.setitem(lab.CONFIG, "max_local_workers", 1)
    lab.add_experiment(con, "dev-j", "queued", launch_cmd="sleep 5", workers=1)
    lab.add_experiment(con, "dev-k", "queued", launch_cmd="sleep 5", workers=0)
    lab.Service().tick(con)
    assert lab.get(con, "dev-j")["status"] == "running" and lab.get(con, "dev-k")["status"] == "running"
    lab.cancel(con, "dev-j")
    lab.cancel(con, "dev-k")
