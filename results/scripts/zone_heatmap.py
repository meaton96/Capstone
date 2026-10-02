"""Zone-level AGV traffic heatmaps from segment_congestion.csv.

Each sweep cell (Results/gridlock_sweep/<tag>/agv<N>_<RULE>/segment_congestion.csv) has one row per zone per
episode. Zone positions are not in the CSV, so they are rebuilt schematically from the zone-name grammar that
TrafficZoneManager uses (RowAisle{a}_Dock{c}[_N|_S], LeftVert_Row{a}, TopSpine_Transit{c}, Lane_{k}, ...). The
picture is topologically faithful (same neighbours, same corners), not to scale.

Usage:
  python results/scripts/zone_heatmap.py --tags LF2_A LF2_B LF2_C LF2_D LF2_E --agv 7 12 \
      --metric total_block_time --out docs/figures/heatmap_block_LF2.png
  metrics: traversal_count (travel), total_block_time (seconds spent waiting to enter), block_events

Per-episode mean over every episode (seed) and both rules found for that tag/agv. Also prints the top zones.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "linux_server", "Results", "gridlock_sweep")
# Single-hue sequential ramp (blue 100 -> 700), light = near zero.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
INK, MUTED, MACHINE = "#1f1f1d", "#6b6a66", "#e4e2dc"


def zone_xy(name, rows, cols, gap_counts):
    """Schematic (x, y, w, h) for a zone name; y grows downward (north at top). None if unknown."""
    xL, xR = -1.6, 2 * (cols - 1) + 1.6
    ySpineT, ySpineB = 0.0, 2.0 * rows
    lane = 0.0
    if name.endswith("_N"): lane, name = -0.28, name[:-2]
    elif name.endswith("_S"): lane, name = 0.28, name[:-2]
    h = 0.5 if lane else 0.9
    m = re.match(r"RowAisle(\d+)_(Dock|Transit)(\d+)$", name)
    if m:
        a, kind, c = int(m[1]), m[2], int(m[3])
        return (2 * c + (1 if kind == "Transit" else 0), 2.0 * (a + 1) + lane, 0.9, h)
    m = re.match(r"(TopSpine|BotSpine)_(Dock|Transit)(\d+)$", name)
    if m:
        c = int(m[3])
        return (2 * c + (1 if m[2] == "Transit" else 0), ySpineT if m[1] == "TopSpine" else ySpineB, 0.9, 0.9)
    m = re.match(r"(LeftVert|RightVert)_(TopConn|BotConn|Row(\d+)|Gap(\d+)_(\d+))$", name)
    if m:
        x = xL if m[1] == "LeftVert" else xR
        def jy(s):  # junction s: 0 = TopConn, 1..rows-1 = Row{s-1}, rows = BotConn
            return ySpineT if s == 0 else (ySpineB if s == rows else 2.0 * s)
        if m[2] == "TopConn": return (x, ySpineT, 1.1, 0.9)
        if m[2] == "BotConn": return (x, ySpineB, 1.1, 0.9)
        if m[3] is not None: return (x, 2.0 * (int(m[3]) + 1), 1.1, 0.9 if not lane else 1.0)
        s, k = int(m[4]), int(m[5]); n = gap_counts.get((m[1], s), k) + 1
        return (x, jy(s) + k * (jy(s + 1) - jy(s)) / n, 1.1, 0.9 * (jy(s + 1) - jy(s)) / n)
    m = re.match(r"InSiding_(Exit|Over|N(\d+))$", name)
    if m:  # bypass return strip, north of the top spine
        x = {"Exit": xL - 1.3, "Over": xL}.get(m[1], (int(m[2]) - 1) if m[2] else 0)
        return (x, ySpineT - 1.1, 0.9, 0.7)
    m = re.match(r"(InSiding|OutSiding)_(Entry|Dock)$", name)
    if m:  # I/O sidings (ioDocks=siding): outside the left/right vertical, Dock beside the corner
        top = m[1] == "InSiding"
        n = gap_counts.get(("LeftVert", 0) if top else ("RightVert", rows - 1), 0) + 1
        step = 2.0 / n
        y = (ySpineT if top else ySpineB) + (0 if m[2] == "Dock" else (step if top else -step))
        return ((xL - 1.3) if top else (xR + 1.3), y, 1.1, 0.9 * step if m[2] == "Entry" else 0.9)
    m = re.match(r"Lane_(\d+)$", name)
    if m:
        k, n = int(m[1]), gap_counts.get(("Lane", 0), int(m[1])) + 1
        return (xR - k * (xR - xL) / max(n - 1, 1), ySpineB + 1.4, (xR - xL) / max(n, 2), 0.6)
    return None  # bays, lane exits, parking alcoves: not drawn (parking is off the traffic floor)


def load(tag, agv, clean_only=False):
    """clean_only drops deadlocked episodes (results.csv deadlock_detected): once gridlocked, waiting time keeps
    accumulating for the rest of the episode and swamps the normal-operation contention pattern."""
    frames = []
    for f in glob.glob(os.path.join(BASE, tag, f"agv{agv}_*", "segment_congestion.csv")):
        df = pd.read_csv(f)
        if clean_only:
            res = pd.read_csv(os.path.join(os.path.dirname(f), "results.csv"))
            bad = set(res.loc[res.deadlock_detected == 1, "seed"])
            df = df[~df.seed.isin(bad)]
        df["cell"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    if not frames: return None
    df = pd.concat(frames)
    if df.empty: return None
    episodes = df.groupby(["cell", "seed"]).ngroups
    agg = df.groupby("zone_name")[["traversal_count", "block_events", "total_block_time"]].sum() / episodes
    return agg, episodes


def draw(ax, agg, metric, norm, title):
    names = list(agg.index)
    rows = 1 + max([int(m[1]) + 1 for n in names if (m := re.match(r"RowAisle(\d+)", n))] + [0])
    cols = 1 + max([int(m[1]) for n in names if (m := re.search(r"Dock(\d+)", n))] + [0])
    gaps = {}
    for n in names:
        if (m := re.match(r"(LeftVert|RightVert)_Gap(\d+)_(\d+)", n)):
            key = (m[1], int(m[2])); gaps[key] = max(gaps.get(key, 0), int(m[3]))
        if (m := re.match(r"Lane_(\d+)$", n)):
            gaps[("Lane", 0)] = max(gaps.get(("Lane", 0), 0), int(m[1]))
    # Machines (context only), one grey block per grid cell between aisles.
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((2 * c - 0.45, 2 * r + 0.55), 0.9, 0.9, color=MACHINE, lw=0, zorder=0))
    for n in names:
        g = zone_xy(n, rows, cols, gaps)
        if g is None: continue
        x, y, w, h = g
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=CMAP(norm(agg.at[n, metric])),
                               edgecolor="white", lw=1.0, zorder=1))
    xL, xR = -1.6, 2 * (cols - 1) + 1.6
    ax.annotate("IN", (xL, -0.75), ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
    ax.annotate("OUT", (xR, 2 * rows + 0.75), ha="center", va="top", fontsize=8, color=INK, weight="bold")
    ax.set_xlim(xL - 2.2, xR + 2.2); ax.set_ylim(2 * rows + 2.2, -1.9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--agv", nargs="+", type=int, required=True)
    ap.add_argument("--metric", default="total_block_time",
                    choices=["traversal_count", "block_events", "total_block_time"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--clean-only", action="store_true", help="exclude deadlocked episodes")
    ap.add_argument("--base", default=None, help="results dir holding the tags (default linux_server/Results/gridlock_sweep)")
    a = ap.parse_args()
    global BASE
    if a.base: BASE = a.base

    data = {}
    for agv in a.agv:
        for tag in a.tags:
            got = load(tag, agv, a.clean_only)
            if got is not None: data[(agv, tag)] = got
    vmax = max(d[0][a.metric].max() for d in data.values())
    norm = Normalize(0, vmax)
    fig, axes = plt.subplots(len(a.agv), len(a.tags), figsize=(2.6 * len(a.tags), 3.4 * len(a.agv)), squeeze=False)
    label = {"traversal_count": "zone entries / episode", "block_events": "blocked reservation attempts / episode",
             "total_block_time": "seconds spent waiting to enter / episode"}[a.metric]
    for i, agv in enumerate(a.agv):
        for j, tag in enumerate(a.tags):
            ax = axes[i][j]
            if (agv, tag) not in data: ax.axis("off"); continue
            agg, eps = data[(agv, tag)]
            draw(ax, agg, a.metric, norm, f"{tag} · {agv} AGVs · n={eps}")
            top = agg[a.metric].sort_values(ascending=False).head(a.top)
            share = top.sum() / max(agg[a.metric].sum(), 1e-9)
            print(f"{tag} agv{agv} ({eps} ep): top {a.top} = {share:.0%} of total | " +
                  ", ".join(f"{z} {v:.0f}" for z, v in top.items()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02)
    cb.set_label(label, color=MUTED, fontsize=9); cb.outline.set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
