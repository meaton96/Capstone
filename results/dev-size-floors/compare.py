"""
@file compare.py
@brief dev-size-floors: summary of every exported floor (machines, bays, zones, footprint), and the new m50 / m100
       exports (current player) against the v4 exports rq3-size-v5 used (../agv-ratio-scale/floors), field by field.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "agv-ratio-scale" / "floors"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def footprint(d):
    xs = [m["pos3"][0] for m in d["machines"] if m.get("pos3")]
    zs = [m["pos3"][2] for m in d["machines"] if m.get("pos3")]
    return (round(max(xs) - min(xs), 1), round(max(zs) - min(zs), 1)) if xs else None


for f in sorted(HERE.glob("floors/m*/des_floor.json"), key=lambda p: int(p.parent.name[1:])):
    d = load(f)
    print(f"{f.parent.name:6s} machines {len(d['machines']):5d}  bays {len(d['agvs']):5d}  zones {len(d['zones']):6d}  "
          f"footprint (x, z) m {footprint(d)}  obs frame {'obs' in d}")
for m in (50, 100):
    a, b = load(HERE / "floors" / f"m{m}" / "des_floor.json"), load(OLD / f"m{m}" / "des_floor.json")
    if a is None or b is None:
        print(f"m{m}: missing ({'new' if a is None else 'v4'})")
        continue
    diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print(f"m{m} new vs v4: differing top-level fields {diff or 'none'}")
    if "machines" in diff:
        n = sum(x != y for x, y in zip(a["machines"], b["machines"]))
        print(f"   machines differing: {n} / {len(a['machines'])}")
