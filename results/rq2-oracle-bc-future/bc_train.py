"""
@file bc_train.py
@brief rq2-oracle-bc-future (2026-10-15): behaviour cloning of the v5 switching oracle with and without the future job
       table. Port of ../rq2-oracle-bc-v5/bc_train.py (same targets, optimizer, early stopping and per-slot scores) to
       this experiment's data and arms; frozen v5 code + this experiment's code/ (common.py).

One call = one model: --arm and --bc-seed.
  state       SchedulingNetwork (the RL policy network), the observation only (rq2-oracle-bc-v5's "full" model)
  future64    FutureSchedulingNetwork: the same + the first 64 rows of the future job table (2 h horizon)
  future128   the same with 128 rows (the 2 h horizon holds about 98 jobs on average, so 64 rows cover ~78 min)
Data: training labels out/labels/s<seed>.npz of B2 seeds 3000-3999 (collect.py label; --n-train limits them to the
first N seeds), 15% of those seeds held out by seed (drawn from the BC seed) for early stopping. Evaluation, per slot,
on the held-out B2 test seeds 0-39 and fitting seeds 2000-2159 (labels ../rq2-rebase-v5/out/oracle, future tables
out/future): deterministic argmax pair vs the oracle's pair, its job rule, and the hold-to-end regret of the pick.
Targets: soft labels q(pair) ~ exp(-regret% / tau), tau = 1%, marginalized onto the two heads (as rq2-oracle-bc-v5).
Output: bc/bc_<arm>_s<k><tag>.pt (train.py checkpoint; run_settings say the arm and rows) and
bc/scores_<arm>_s<k><tag>.json (scores and every evaluation slot's pick). Skips a model whose scores exist.
@par Usage
@code{.sh}
python results/rq2-oracle-bc-future/bc_train.py --arm future64 --bc-seed 0 --device cuda
@endcode
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                 # noqa: E402  (frozen code first; overlays the future table)

import numpy as np                                 # noqa: E402
import torch                                       # noqa: E402
import torch.nn as nn                              # noqa: E402
import torch.nn.functional as F                    # noqa: E402
from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES, PPOConfig   # noqa: E402
from models.network import SchedulingNetwork       # noqa: E402

ARMS = {"state": 0, "future64": 64, "future128": 128}
STREAMS = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")
FUTURE = "future_job_table"


def load(paths, future_paths, rows):
    """Obs (float16 grids / job tables as stored), tails, seed and slot of each row; the future table's first rows."""
    obs = {k: [] for k in STREAMS}
    fut, tails, seed_of, slot_of = [], [], [], []
    for p, fp in zip(paths, future_paths):
        d = np.load(p)
        for k in STREAMS:
            obs[k].append(d[k])
        tails.append(d["tails"])
        if rows:
            f = np.load(fp)["future"] if fp is not None else d["future"]
            fut.append(f[:, :rows].astype(np.float16))
        seed_of += [int(d["seed"])] * len(d["tails"])
        slot_of += list(range(len(d["tails"])))
        if list(d["pairs"]) != C.V5:
            raise SystemExit(f"{p}: pairs {list(d['pairs'])} are not the v5 pairs")
    out = {k: np.concatenate(v) for k, v in obs.items()}
    if rows:
        out[FUTURE] = np.concatenate(fut)
    return out, np.concatenate(tails), np.array(seed_of), np.array(slot_of)


def head_index():
    j = np.array([JOB_HEAD_RULES.index(p.split("-")[0]) for p in C.V5])
    m = np.array([MACHINE_HEAD_RULES.index(p.split("-")[1]) for p in C.V5])
    return j, m


def soft_targets(tails, tau):
    best = tails.min(axis=1, keepdims=True)
    regret = 100.0 * (tails - best) / np.maximum(best, 1e-9)
    q = np.exp(-regret / tau)
    q /= q.sum(axis=1, keepdims=True)
    j, m = head_index()
    qj = np.stack([q[:, j == i].sum(1) for i in range(len(JOB_HEAD_RULES))], 1)
    qm = np.stack([q[:, m == i].sum(1) for i in range(len(MACHINE_HEAD_RULES))], 1)
    return qj.astype(np.float32), qm.astype(np.float32), regret


def batch(obs, idx, device):
    return {k: torch.as_tensor(v[idx], device=device).float() for k, v in obs.items()}


def logits_of(model, obs):
    return model.actor_critic.actor(model._features(obs))


def loss_fn(logits, qj, qm):
    nj = len(JOB_HEAD_RULES)
    return (-(qj * F.log_softmax(logits[:, :nj], -1)).sum(-1).mean()
            - (qm * F.log_softmax(logits[:, nj:], -1)).sum(-1).mean())


def predict(model, obs, device, bs=256):
    model.eval()
    n = len(obs["global_scalars"])
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            out.append(logits_of(model, batch(obs, np.arange(i, min(i + bs, n)), device)).cpu())
    return torch.cat(out).numpy()


def fit(model, obs, qj, qm, tr, va, device, epochs, patience, lr, log):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    qj_t, qm_t = torch.as_tensor(qj, device=device), torch.as_tensor(qm, device=device)
    rng = np.random.default_rng(0)
    va_obs = {k: v[va] for k, v in obs.items()}
    for ep in range(epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = loss_fn(logits_of(model, batch(obs, idx, device)), qj_t[idx], qm_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        lv = loss_fn(torch.as_tensor(predict(model, va_obs, device), device=device), qj_t[va], qm_t[va]).item()
        log(f"  epoch {ep + 1}: validation loss {lv:.4f}")
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, ep + 1


def picks(logits):
    nj = len(JOB_HEAD_RULES)
    j, m = head_index()
    pj, pm = logits[:, :nj].argmax(1), logits[:, nj:].argmax(1)
    return np.array([np.flatnonzero((j == a) & (m == b))[0] for a, b in zip(pj, pm)])


def scores(chosen, tails, regret):
    if not len(chosen):
        return None
    j, _ = head_index()
    best = tails.argmin(1)
    ref = C.V5.index("MDD-PTECT")
    return {"slots": int(len(chosen)), "pair_match_%": 100 * float((chosen == best).mean()),
            "job_rule_match_%": 100 * float((j[chosen] == j[best]).mean()),
            "mean_regret_%": float(regret[np.arange(len(chosen)), chosen].mean()),
            "mddptect_regret_%": float(regret[:, ref].mean()),
            "share_mddptect_%": 100 * float((chosen == ref).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--bc-seed", type=int, required=True)
    ap.add_argument("--train-seeds", default="3000-3999")
    ap.add_argument("--n-train", type=int, default=0, help="first N training seeds (learning curve); 0 = all")
    ap.add_argument("--tau", type=float, default=1.0, help="soft-label temperature, in regret %%")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-dir", default=str(C.HERE / "out" / "labels"))
    ap.add_argument("--future-dir", default="", help="future tables of the training seeds when the label files have "
                    "none (e.g. --label-dir ../rq2-rebase-v5/out/oracle --future-dir out/future: the 160 fitting seeds)")
    ap.add_argument("--out-dir", default=str(C.HERE / "bc"))
    ap.add_argument("--tag", default="", help="output suffix (e.g. _n250 for a learning-curve point, _smoke)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"bc_{a.arm}_s{a.bc_seed}{a.tag}"
    if (out_dir / f"scores_{a.arm}_s{a.bc_seed}{a.tag}.json").exists():
        print(f"{name}: done, skipped")
        return
    logf = open(out_dir / f"{name}.log", "w")

    def log(msg):
        print(msg, file=logf, flush=True)

    rows = ARMS[a.arm]
    t0 = time.time()
    want = C.sim.parse_seeds(a.train_seeds)[:a.n_train or None]
    train_seeds = [s for s in want if (Path(a.label_dir) / f"s{s}.npz").exists()]
    if len(train_seeds) < len(want):
        raise SystemExit(f"only {len(train_seeds)} of {len(want)} training label files in {a.label_dir}")
    fdirs = [Path(a.future_dir) / f"s{s}.npz" if a.future_dir else None for s in train_seeds]
    obs, tails, seed_of, _ = load([Path(a.label_dir) / f"s{s}.npz" for s in train_seeds], fdirs, rows)
    ev = [s for s in C.EVAL_SEEDS if (C.HERE / "out" / "future" / f"s{s}.npz").exists()]
    eobs, etails, eseed, eslot = load([C.ORACLE_DIR / f"s{s}.npz" for s in ev],
                                      [C.HERE / "out" / "future" / f"s{s}.npz" for s in ev], rows)
    qj, qm, _ = soft_targets(tails, a.tau)
    _, _, eregret = soft_targets(etails, a.tau)
    log(f"{name}: train {len(train_seeds)} seeds / {len(tails)} slots, eval {len(ev)} seeds / {len(etails)} slots, "
        f"rows {rows}, device {a.device}, load {time.time() - t0:.0f} s")
    rng = np.random.default_rng(a.bc_seed)
    va_seeds = rng.choice(train_seeds, size=max(1, int(0.15 * len(train_seeds))), replace=False)
    va = np.flatnonzero(np.isin(seed_of, va_seeds))
    tr = np.flatnonzero(~np.isin(seed_of, va_seeds))
    torch.manual_seed(a.bc_seed)
    model = (C.future_net.FutureSchedulingNetwork() if rows else SchedulingNetwork()).to(a.device)
    t1 = time.time()
    vloss, eps = fit(model, obs, qj, qm, tr, va, a.device, a.epochs, a.patience, a.lr, log)
    train_chosen = picks(predict(model, {k: v[tr] for k, v in obs.items()}, a.device))
    _, _, tregret = soft_targets(tails[tr], a.tau)
    sc_train = scores(train_chosen, tails[tr], tregret)
    va_chosen = picks(predict(model, {k: v[va] for k, v in obs.items()}, a.device))
    _, _, vregret = soft_targets(tails[va], a.tau)
    sc_val = scores(va_chosen, tails[va], vregret)
    chosen = picks(predict(model, eobs, a.device))
    test = eseed < 2000
    redrawn = {}
    rd_paths = [C.HERE / "out" / "future_redrawn" / f"s{s}.npz" for s in ev]
    if rows and all(p.exists() for p in rd_paths):
        # the same slots with the redrawn (unrelated) future in the table: does the pick depend on the table?
        robs = dict(eobs)
        robs[FUTURE] = np.concatenate([np.load(p)["future"][:, :rows].astype(np.float16) for p in rd_paths])
        rchosen = picks(predict(model, robs, a.device))
        redrawn = {"eval_all_redrawn": scores(rchosen, etails, eregret),
                   "redrawn_pick_changed_%": 100 * float((rchosen != chosen).mean())}
    res = {"arm": a.arm, "rows": rows, "bc_seed": a.bc_seed, "tag": a.tag, "n_train_seeds": len(train_seeds),
           "epochs": eps, "val_loss": vloss, "train_minutes": (time.time() - t1) / 60, "device": a.device,
           "train": sc_train, "val": sc_val, "eval_all": scores(chosen, etails, eregret),
           "eval_test": scores(chosen[test], etails[test], eregret[test]),
           "eval_fit": scores(chosen[~test], etails[~test], eregret[~test]), **redrawn,
           "eval_picks": {f"{s}:{k}": C.V5[c] for s, k, c in zip(eseed, eslot, chosen)},
           "stamp": C.stamp()}
    import train as T
    from rewards import load_reward
    T.save_checkpoint(out_dir / f"{name}.pt", model, torch.optim.Adam(model.parameters()), 0, PPOConfig(),
                      load_reward(str(C.FROZEN / "config/rewards/tardiness.json")).name, C.CAPS,
                      settings={"experiment": "rq2-oracle-bc-future", "arm": a.arm, "future_rows": rows,
                                "future_horizon": C.HORIZON, "bc_seed": a.bc_seed, "n_train_seeds": len(train_seeds)})
    C.write_atomic(out_dir / f"scores_{a.arm}_s{a.bc_seed}{a.tag}.json", json.dumps(res))
    short = {k: (round(v["pair_match_%"], 1), round(v["mean_regret_%"], 2)) for k, v in res.items()
             if k in ("train", "val", "eval_test", "eval_fit") and v}
    log(f"{name}: {eps} epochs, val loss {vloss:.4f}, (pair match %, regret %) {short}, {res['train_minutes']:.1f} min")
    if redrawn:
        short["redrawn"] = (round(redrawn["eval_all_redrawn"]["pair_match_%"], 1),
                            round(redrawn["eval_all_redrawn"]["mean_regret_%"], 2),
                            f"{redrawn['redrawn_pick_changed_%']:.1f}% picks changed")
    print(f"{name}: {eps} epochs, val loss {vloss:.4f}, (pair match %, regret %) {short}", flush=True)


if __name__ == "__main__":
    C.check_frozen()
    main()
    C.check_frozen()
