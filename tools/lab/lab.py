#!/usr/bin/env python3
"""
@file lab.py
@brief Experiment tracker and runner for the Capstone project (2026-10-04). One SQLite database, a background
       service that watches, finishes and starts experiments, a local web page, and a CLI.

@details
Experiments have a status: future (waits on something) -> queued -> running -> done | failed | cancelled.
The service (`lab.py serve`) ticks every `poll_seconds`:
  - running, local: alive while its pid / process pattern runs; done when its done-check holds (a glob count), failed
    when the process is gone without it;
  - running, cluster: done when none of its Slurm job ids is in `squeue` (over the shared SSH connection); then its
    pull command runs and the done-check is applied;
  - on finish: runs the analysis command (output kept in the event), notifies Discord (env/notify.py), and writes an
    inbox event that a Claude session can wait on (`lab.py wait`);
  - future experiments whose `waits_on` experiments are all done move to the queue (and notify);
  - queued experiments start in priority order while local worker slots are free (cluster ones start at once); a
    started command's output goes to tools/lab/state/logs/<name>.log and Slurm ids are read from "Submitted batch job";
  - timers: `check_at` and an overdue ETA notify once;
  - keeps the cluster SSH master alive while cluster experiments run (it cannot pass Duo itself: if the connection
    drops it notifies you to log in again with `linux_server/slurm/rit_ssh.sh`'s alias);
  - rewrites docs/WIP/LAB_STATUS.md (running / queued / future / recent) when anything changes.
The web page (http://127.0.0.1:<port>, localhost only) shows the same state with a live clock and lets you add
experiments and start / cancel / requeue them. Every write needs the token in tools/lab/state/token (the page gets it
from the server), so other web pages in the browser cannot drive it.

@par Usage
@code{.sh}
python3 tools/lab/lab.py serve                       # service + web page (keep it running: nohup / systemd --user)
python3 tools/lab/lab.py status                      # table of everything
python3 tools/lab/lab.py add NAME --status future --waits-on X,Y --note "..."
python3 tools/lab/lab.py add NAME --status queued --launch "bash results/NAME/run.sh" --done-glob 'results/NAME/s*/result.json' --done-count 20 --workers 20
python3 tools/lab/lab.py set NAME --check-at 2026-10-05T03:00 --eta 2026-10-05T05:00
python3 tools/lab/lab.py start|cancel|requeue|done NAME
python3 tools/lab/lab.py wait                        # blocks until the next finished / failed / timer event (Claude)
python3 tools/lab/lab.py inbox                       # unread events
@endcode
"""
import argparse
import glob
import json
import os
import re
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state"
DB_PATH = STATE / "lab.db"
TOKEN_PATH = STATE / "token"
LOG_DIR = STATE / "logs"
STATUS_MD = REPO / "docs" / "WIP" / "LAB_STATUS.md"
CONFIG = {
    "poll_seconds": 30,
    "port": 8765,
    "max_local_workers": 20,          # worker slots on this machine (one Unity player ~ one slot)
    "ssh_alias": "rit-research",
    "ssh_keepalive_seconds": 300,
}
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home()}/.ssh/cm-%C",
       "-o", "ControlPersist=30m"]
STATUSES = ("future", "queued", "running", "done", "failed", "cancelled")
FIELDS = ("purpose", "location", "launch_cmd", "cwd", "workers", "proc_pattern", "done_glob", "done_count",
          "slurm_jobs", "pull_cmd", "analysis_cmd", "results_path", "waits_on", "waits_note", "eta", "check_at",
          "notes", "writeup", "priority", "auto_start")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    name TEXT PRIMARY KEY, purpose TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'future',
    location TEXT DEFAULT 'local', launch_cmd TEXT DEFAULT '', cwd TEXT DEFAULT '', workers INTEGER DEFAULT 1,
    proc_pattern TEXT DEFAULT '', pid INTEGER, done_glob TEXT DEFAULT '', done_count INTEGER DEFAULT 0,
    slurm_jobs TEXT DEFAULT '', pull_cmd TEXT DEFAULT '', analysis_cmd TEXT DEFAULT '', results_path TEXT DEFAULT '',
    waits_on TEXT DEFAULT '', waits_note TEXT DEFAULT '', eta TEXT DEFAULT '', check_at TEXT DEFAULT '',
    notes TEXT DEFAULT '', writeup TEXT DEFAULT '', priority INTEGER DEFAULT 50, auto_start INTEGER DEFAULT 1,
    started_at TEXT DEFAULT '', finished_at TEXT DEFAULT '', log_path TEXT DEFAULT '', flags TEXT DEFAULT '{}',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, experiment TEXT, kind TEXT NOT NULL,
    message TEXT NOT NULL, detail TEXT DEFAULT '', read INTEGER DEFAULT 0
);
"""
INBOX_KINDS = ("finished", "failed", "timer", "overdue", "promoted", "cluster")


# ── storage ──────────────────────────────────────────────────────────────────────────────────────────

def now():
    return datetime.now().replace(microsecond=0)


def iso(t):
    return t.isoformat(timespec="seconds") if t else ""


def parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def connect():
    STATE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def get(con, name):
    r = con.execute("SELECT * FROM experiments WHERE name = ?", (name,)).fetchone()
    if r is None:
        raise KeyError(f"no experiment {name!r}")
    return dict(r)


def all_experiments(con):
    return [dict(r) for r in con.execute("SELECT * FROM experiments ORDER BY priority, created_at")]


def update(con, name, **fields):
    fields["updated_at"] = iso(now())
    cols = ", ".join(f"{k} = ?" for k in fields)
    con.execute(f"UPDATE experiments SET {cols} WHERE name = ?", (*fields.values(), name))


def add_experiment(con, name, status="future", **fields):
    if not re.fullmatch(r"[a-z0-9][a-z0-9.\-_]*", name):
        raise ValueError(f"name {name!r}: lowercase, digits, '-', '.', '_' (CLAUDE.md naming)")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    t = iso(now())
    vals = {k: v for k, v in fields.items() if k in FIELDS and v is not None}
    cols = ["name", "status", "created_at", "updated_at", *vals]
    con.execute(f"INSERT INTO experiments ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                (name, status, t, t, *vals.values()))
    event(con, name, "added", f"{name} added as {status}")


def event(con, name, kind, message, detail=""):
    con.execute("INSERT INTO events (ts, experiment, kind, message, detail, read) VALUES (?, ?, ?, ?, ?, ?)",
                (iso(now()), name, kind, message, detail, 0 if kind in INBOX_KINDS else 1))


def flags(exp):
    try:
        return json.loads(exp.get("flags") or "{}")
    except json.JSONDecodeError:
        return {}


# ── checks ───────────────────────────────────────────────────────────────────────────────────────────

def notify(text):
    """Discord through env/notify.py (webhook in $NOTIFY_WEBHOOK_URL or ~/.capstone_webhook); never raises."""
    try:
        sys.path.insert(0, str(REPO / "env"))
        from notify import notify as _notify
        return _notify(text)
    except Exception as e:   # noqa: BLE001 - a notification must never stop the service
        print(f"[lab] notify failed: {type(e).__name__}: {e}", flush=True)
        return False


def done_count(exp):
    if not exp["done_glob"]:
        return None
    return len(glob.glob(str(REPO / exp["done_glob"])))


def done_check(exp):
    n = done_count(exp)
    return n is not None and n >= max(1, exp["done_count"] or 1)


_CHILDREN = {}   # pid -> Popen of the commands this service started (polled, so finished ones are reaped)


def local_alive(exp):
    if exp["pid"] in _CHILDREN:
        if _CHILDREN[exp["pid"]].poll() is None:
            return True
        del _CHILDREN[exp["pid"]]
    elif exp["pid"]:
        try:
            os.kill(exp["pid"], 0)
            return True
        except OSError:
            pass
    if exp["proc_pattern"]:
        r = subprocess.run(["pgrep", "-f", exp["proc_pattern"]], capture_output=True, text=True)
        mine = str(os.getpid())
        return any(p and p != mine for p in r.stdout.split())
    return False


def ssh(cmd, timeout=60):
    return subprocess.run([*SSH, CONFIG["ssh_alias"], cmd], capture_output=True, text=True, timeout=timeout)


def cluster_jobs_left(exp):
    """Slurm ids of this experiment still in squeue; None when the cluster cannot be reached."""
    ids = [j for j in re.split(r"[,\s]+", exp["slurm_jobs"] or "") if j]
    if not ids:
        return []
    try:
        r = ssh(f"squeue --me -h -o %i -j {','.join(ids)} 2>/dev/null; echo __ok__")
    except subprocess.TimeoutExpired:
        return None
    if "__ok__" not in r.stdout:
        return None
    return [line.strip().split("_")[0] for line in r.stdout.splitlines() if line.strip() and line.strip() != "__ok__"]


def run_shell(cmd, cwd=None, timeout=1800):
    r = subprocess.run(cmd, shell=True, cwd=cwd or REPO, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-6000:]


# ── actions ──────────────────────────────────────────────────────────────────────────────────────────

def start(con, name):
    exp = get(con, name)
    if not exp["launch_cmd"]:
        raise ValueError(f"{name} has no launch command (set one with: lab.py set {name} --launch ...)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"
    cwd = REPO / exp["cwd"] if exp["cwd"] else REPO
    with open(log, "a") as fh:
        fh.write(f"\n[{iso(now())}] lab start: {exp['launch_cmd']}\n")
        fh.flush()
        if exp["location"] == "cluster":
            r = subprocess.run(exp["launch_cmd"], shell=True, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                               timeout=600)
            fh.flush()
            jobs = re.findall(r"Submitted batch job (\d+)", log.read_text())
            if r.returncode != 0 or not jobs:
                update(con, name, status="failed", finished_at=iso(now()), log_path=str(log))
                event(con, name, "failed", f"{name}: cluster launch failed (exit {r.returncode}, no job ids); see {log}")
                return
            update(con, name, status="running", started_at=iso(now()), slurm_jobs=",".join(jobs), log_path=str(log))
            event(con, name, "started", f"{name} submitted: jobs {','.join(jobs)}")
            return
        p = subprocess.Popen(exp["launch_cmd"], shell=True, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    _CHILDREN[p.pid] = p
    update(con, name, status="running", pid=p.pid, started_at=iso(now()), log_path=str(log), flags="{}")
    event(con, name, "started", f"{name} started (pid {p.pid}), log {log.relative_to(REPO)}")


def cancel(con, name):
    exp = get(con, name)
    if exp["status"] == "running":
        if exp["location"] == "cluster" and exp["slurm_jobs"]:
            ids = [j for j in re.split(r"[,\s]+", exp["slurm_jobs"]) if j]
            ssh("scancel " + " ".join(ids))
        elif exp["pid"]:
            try:
                os.killpg(exp["pid"], signal.SIGTERM)
            except OSError:
                pass
    update(con, name, status="cancelled", finished_at=iso(now()))
    event(con, name, "cancelled", f"{name} cancelled")


def finish(con, exp, ok, why):
    name = exp["name"]
    detail = ""
    if ok and exp["analysis_cmd"]:
        try:
            code, out = run_shell(exp["analysis_cmd"], timeout=1800)
            detail = f"$ {exp['analysis_cmd']}  (exit {code})\n{out}"
        except subprocess.TimeoutExpired:
            detail = "analysis timed out"
    status = "done" if ok else "failed"
    update(con, name, status=status, finished_at=iso(now()), pid=None)
    started = parse_time(exp["started_at"])
    took = f" after {str(now() - started).split('.')[0]}" if started else ""
    msg = f"{name} {'finished' if ok else 'FAILED'}{took}: {why}"
    event(con, name, "finished" if ok else "failed", msg, detail)
    tail = "\n".join(detail.strip().splitlines()[-12:]) if detail else ""
    notify(f"{':white_check_mark:' if ok else ':x:'} **lab** {msg}" + (f"\n```\n{tail[:1500]}\n```" if tail else ""))


# ── the service ──────────────────────────────────────────────────────────────────────────────────────

class Service:
    def __init__(self):
        self.last_keepalive = 0.0
        self.cluster_down_notified = False
        self.last_md = ""

    def tick(self, con):
        exps = all_experiments(con)
        status = {e["name"]: e["status"] for e in exps}
        uses_cluster = False
        for exp in exps:
            if exp["status"] != "running":
                continue
            if exp["location"] == "cluster":
                uses_cluster = True
                left = cluster_jobs_left(exp)
                if left is None:
                    self.cluster_unreachable(con)
                    continue
                self.cluster_down_notified = False
                if left:
                    continue
                if exp["pull_cmd"]:
                    try:
                        code, out = run_shell(exp["pull_cmd"], timeout=3600)
                    except subprocess.TimeoutExpired:
                        code, out = 1, "pull timed out"
                    if code != 0:
                        finish(con, exp, False, f"jobs ended, pull failed: {out[-300:]}")
                        continue
                ok = done_check(exp) if exp["done_glob"] else True
                finish(con, exp, ok, "all Slurm jobs ended" + ("" if ok else f"; done-check {done_count(exp)}/"
                                                               f"{exp['done_count']}"))
            else:
                if done_check(exp) and not local_alive(exp):
                    finish(con, exp, True, f"{done_count(exp)}/{exp['done_count']} outputs")
                elif not local_alive(exp):
                    if done_check(exp):
                        finish(con, exp, True, "outputs complete")
                    else:
                        n = done_count(exp)
                        finish(con, exp, n is None, "process ended" + ("" if n is None else
                                                                      f" with {n}/{exp['done_count']} outputs"))
        # promote futures whose dependencies are done
        status = {e["name"]: e["status"] for e in all_experiments(con)}
        for exp in all_experiments(con):
            if exp["status"] != "future" or not exp["waits_on"].strip():
                continue
            deps = [d.strip() for d in exp["waits_on"].split(",") if d.strip()]
            if deps and all(status.get(d) == "done" for d in deps) and not exp["waits_note"].strip():
                update(con, exp["name"], status="queued")
                event(con, exp["name"], "promoted", f"{exp['name']} queued: {', '.join(deps)} done")
                notify(f":arrow_forward: **lab** {exp['name']} is ready (waited on {', '.join(deps)})")
        # start queued experiments while slots are free
        running_workers = sum(e["workers"] or 0 for e in all_experiments(con)
                              if e["status"] == "running" and e["location"] == "local")
        for exp in all_experiments(con):
            if exp["status"] != "queued" or not exp["auto_start"] or not exp["launch_cmd"]:
                continue
            if exp["location"] == "local" and running_workers + (exp["workers"] or 1) > CONFIG["max_local_workers"]:
                continue
            try:
                start(con, exp["name"])
                if exp["location"] == "local":
                    running_workers += exp["workers"] or 1
                notify(f":rocket: **lab** started {exp['name']}")
            except Exception as e:   # noqa: BLE001
                event(con, exp["name"], "failed", f"{exp['name']}: start failed: {e}")
                update(con, exp["name"], status="failed")
        # timers
        t = now()
        for exp in all_experiments(con):
            f = flags(exp)
            ca, eta = parse_time(exp["check_at"]), parse_time(exp["eta"])
            changed = False
            if ca and ca <= t and f.get("check_at_fired") != exp["check_at"]:
                f["check_at_fired"] = exp["check_at"]
                changed = True
                event(con, exp["name"], "timer", f"{exp['name']}: check time {exp['check_at']} reached "
                                                 f"(status {exp['status']}, {done_count(exp)} outputs)")
                notify(f":alarm_clock: **lab** check {exp['name']} ({exp['status']}, outputs {done_count(exp)})")
            if exp["status"] == "running" and eta and eta < t and f.get("overdue_fired") != exp["eta"]:
                f["overdue_fired"] = exp["eta"]
                changed = True
                event(con, exp["name"], "overdue", f"{exp['name']} is past its ETA {exp['eta']}")
                notify(f":hourglass: **lab** {exp['name']} past ETA {exp['eta']}")
            if changed:
                update(con, exp["name"], flags=json.dumps(f))
        if uses_cluster and time.time() - self.last_keepalive > CONFIG["ssh_keepalive_seconds"]:
            self.last_keepalive = time.time()
            try:
                ssh("true", timeout=30)
            except subprocess.TimeoutExpired:
                pass
        self.write_markdown(con)

    def cluster_unreachable(self, con):
        if not self.cluster_down_notified:
            self.cluster_down_notified = True
            msg = (f"cluster unreachable over the shared SSH connection: log in once (ssh {CONFIG['ssh_alias']}, "
                   "Duo) so the lab service can watch the cluster jobs again")
            event(con, None, "cluster", msg)
            notify(f":warning: **lab** {msg}")

    def write_markdown(self, con):
        md = render_markdown(all_experiments(con))
        if md != self.last_md and STATUS_MD.parent.is_dir():
            STATUS_MD.write_text(md)
            self.last_md = md


def render_markdown(exps):
    out = ["# Lab status (generated by tools/lab/lab.py; do not edit by hand)", "",
           f"Updated {iso(now())}. Edit through the web page (`python3 tools/lab/lab.py serve`, "
           f"http://127.0.0.1:{CONFIG['port']}) or `tools/lab/lab.py add / set`.", ""]
    for st, title in (("running", "Running"), ("queued", "Queued"), ("future", "Future (waiting)"),
                      ("failed", "Failed"), ("done", "Done"), ("cancelled", "Cancelled")):
        rows = [e for e in exps if e["status"] == st]
        if not rows:
            continue
        out += [f"## {title}", "", "| name | where | started / finished | ETA / check | waits on | notes |",
                "|---|---|---|---|---|---|"]
        for e in rows:
            when = e["finished_at"] or e["started_at"] or ""
            prog = done_count(e)
            prog = f" ({prog}/{e['done_count']})" if prog is not None and st == "running" else ""
            waits = ", ".join(x for x in (e["waits_on"], e["waits_note"]) if x)
            out.append(f"| {e['name']} | {e['location']}{(' jobs ' + e['slurm_jobs']) if e['slurm_jobs'] else ''}"
                       f"{prog} | {when} | {e['eta'] or e['check_at']} | {waits} | "
                       f"{(e['notes'] or '').replace('|', '/').replace(chr(10), ' ')[:300]} |")
        out.append("")
    return "\n".join(out)


# ── web ──────────────────────────────────────────────────────────────────────────────────────────────

def token():
    STATE.mkdir(parents=True, exist_ok=True)
    if not TOKEN_PATH.exists():
        TOKEN_PATH.write_text(secrets.token_hex(16))
        TOKEN_PATH.chmod(0o600)
    return TOKEN_PATH.read_text().strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE / "index.html").read_text().replace("__LAB_TOKEN__", token())
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path == "/api/state":
            con = connect()
            exps = all_experiments(con)
            for e in exps:
                e["progress"] = done_count(e)
            evs = [dict(r) for r in con.execute("SELECT * FROM events ORDER BY id DESC LIMIT 60")]
            used = sum(e["workers"] or 0 for e in exps if e["status"] == "running" and e["location"] == "local")
            return self._send(200, json.dumps({"now": iso(now()), "experiments": exps, "events": evs,
                                               "workers_used": used, "workers_max": CONFIG["max_local_workers"]}))
        return self._send(404, '{"error": "not found"}')

    def do_POST(self):
        if self.headers.get("X-Lab-Token") != token() or self.headers.get("Host", "").split(":")[0] not in (
                "127.0.0.1", "localhost"):
            return self._send(403, '{"error": "bad token or host"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            con = connect()
            if self.path == "/api/add":
                name = body.pop("name", "").strip()
                st = body.pop("status", "future")
                add_experiment(con, name, st, **{k: v for k, v in body.items() if v not in ("", None)})
            elif self.path == "/api/set":
                name = body.pop("name")
                get(con, name)
                vals = {k: v for k, v in body.items() if k in FIELDS}
                if vals:
                    update(con, name, **vals)
            elif self.path == "/api/action":
                name, act = body["name"], body["action"]
                if act == "start":
                    start(con, name)
                elif act == "cancel":
                    cancel(con, name)
                elif act in ("queued", "future", "done"):
                    update(con, name, status=act, **({"finished_at": iso(now())} if act == "done" else {}))
                    event(con, name, "status", f"{name} -> {act} (web)")
                elif act == "read":
                    con.execute("UPDATE events SET read = 1 WHERE read = 0")
                else:
                    raise ValueError(f"unknown action {act}")
            else:
                return self._send(404, '{"error": "not found"}')
            return self._send(200, '{"ok": true}')
        except Exception as e:   # noqa: BLE001
            return self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def serve(args):
    srv = ThreadingHTTPServer(("127.0.0.1", CONFIG["port"]), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[lab] web page http://127.0.0.1:{CONFIG['port']}  (state {STATE})", flush=True)
    svc = Service()
    while True:
        try:
            svc.tick(connect())
        except Exception as e:   # noqa: BLE001 - keep serving; the error is visible in the log and the events
            print(f"[lab] tick error: {type(e).__name__}: {e}", flush=True)
        time.sleep(CONFIG["poll_seconds"])


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────

def cmd_status(args):
    con = connect()
    for st in STATUSES:
        rows = [e for e in all_experiments(con) if e["status"] == st]
        if not rows or (st in ("done", "cancelled") and not args.all):
            continue
        print(f"== {st} ({len(rows)})")
        for e in rows:
            prog = done_count(e)
            extra = f" {prog}/{e['done_count']}" if prog is not None and st == "running" else ""
            waits = f" waits on: {e['waits_on']} {e['waits_note']}".rstrip() if st == "future" else ""
            print(f"  {e['name']:34s} {e['location']:7s}{extra}{(' eta ' + e['eta']) if e['eta'] else ''}{waits}")
    unread = con.execute("SELECT COUNT(*) FROM events WHERE read = 0").fetchone()[0]
    print(f"({unread} unread events: lab.py inbox)")


def field_args(args):
    m = {"purpose": args.purpose, "location": args.location, "launch_cmd": args.launch, "cwd": args.cwd,
         "workers": args.workers, "proc_pattern": args.proc_pattern, "done_glob": args.done_glob,
         "done_count": args.done_count, "slurm_jobs": args.slurm_jobs, "pull_cmd": args.pull,
         "analysis_cmd": args.analysis, "results_path": args.results, "waits_on": args.waits_on,
         "waits_note": args.waits_note, "eta": args.eta, "check_at": args.check_at, "notes": args.note,
         "writeup": args.writeup, "priority": args.priority,
         "auto_start": None if args.auto_start is None else int(args.auto_start)}
    return {k: v for k, v in m.items() if v is not None}


def cmd_add(args):
    con = connect()
    add_experiment(con, args.name, args.status, **field_args(args))
    print(f"added {args.name} ({args.status})")


def cmd_set(args):
    con = connect()
    get(con, args.name)
    vals = field_args(args)
    if args.status:
        vals["status"] = args.status
    update(con, args.name, **vals)
    event(con, args.name, "status" if args.status else "edited", f"{args.name} updated: {', '.join(vals)}")
    print(f"updated {args.name}: {', '.join(vals)}")


def cmd_action(args):
    con = connect()
    if args.action == "start":
        start(con, args.name)
    elif args.action == "cancel":
        cancel(con, args.name)
    else:
        status = {"requeue": "queued", "done": "done"}[args.action]
        update(con, args.name, status=status, **({"finished_at": iso(now())} if status == "done" else {}))
        event(con, args.name, "status", f"{args.name} -> {status} (cli)")
    print(f"{args.action}: {args.name}")


def print_event(r):
    print(f"[{r['ts']}] {r['kind']}: {r['message']}")
    if r["detail"]:
        print(r["detail"])


def cmd_inbox(args):
    con = connect()
    rows = con.execute("SELECT * FROM events WHERE read = 0 ORDER BY id").fetchall()
    for r in rows:
        print_event(r)
    if rows and not args.keep:
        con.execute(f"UPDATE events SET read = 1 WHERE id <= {rows[-1]['id']} AND read = 0")
    if not rows:
        print("inbox empty")


def cmd_wait(args):
    """Blocks until an unread finished / failed / timer / overdue / promoted / cluster event exists, prints and
    marks them read. Meant for a Claude session to run in the background (one notification per event batch)."""
    con = connect()
    deadline = time.time() + args.timeout * 3600 if args.timeout else None
    while True:
        rows = con.execute("SELECT * FROM events WHERE read = 0 ORDER BY id").fetchall()
        if rows:
            for r in rows:
                print_event(r)
            con.execute(f"UPDATE events SET read = 1 WHERE id <= {rows[-1]['id']} AND read = 0")
            return
        if deadline and time.time() > deadline:
            print("wait: timeout, no new events")
            return
        time.sleep(20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("@par")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve").set_defaults(fn=serve)
    p = sub.add_parser("status")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_status)
    for name, fn in (("add", cmd_add), ("set", cmd_set)):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.add_argument("--status", choices=STATUSES, default="future" if name == "add" else None)
        for flag in ("purpose", "location", "launch", "cwd", "proc-pattern", "done-glob", "slurm-jobs", "pull",
                     "analysis", "results", "waits-on", "waits-note", "eta", "check-at", "note", "writeup"):
            p.add_argument(f"--{flag}")
        for flag in ("workers", "done-count", "priority"):
            p.add_argument(f"--{flag}", type=int)
        p.add_argument("--auto-start", type=int, choices=(0, 1))
        p.set_defaults(fn=fn)
    for act in ("start", "cancel", "requeue", "done"):
        p = sub.add_parser(act)
        p.add_argument("name")
        p.set_defaults(fn=cmd_action, action=act)
    p = sub.add_parser("inbox")
    p.add_argument("--keep", action="store_true", help="do not mark read")
    p.set_defaults(fn=cmd_inbox)
    p = sub.add_parser("wait")
    p.add_argument("--timeout", type=float, default=0, help="hours (0 = none)")
    p.set_defaults(fn=cmd_wait)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
