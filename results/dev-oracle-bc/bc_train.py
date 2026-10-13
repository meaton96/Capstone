"""
@file bc_train.py
@brief dev-oracle-bc step 2: behavior cloning of the switching oracle. Can the policy network represent oracle-quality
       slot choices from obs v3 at all? If supervised imitation from the full observation cannot beat MDD-TECT either,
       the limit is the observation / architecture, not RL; if it can, RL is what fails to find it.

Data: data/s<seed>.npz from collect.py (full obs at each slot start along the oracle trajectory, and every pair's
hold-to-end tardiness at that slot). Targets are soft: q(pair) ~ exp(-regret% / tau), regret% = 100 * (tardiness -
best) / best at that slot, marginalized onto the two heads (the policy is factorized: p(job rule) x p(machine rule)).
Near-ties between pairs therefore do not count as errors.

Models, trained on seeds 2000-2159 (15% of them held out by seed for early stopping), tested on held-out 0-39:
  - full:    SchedulingNetwork, fresh init, the same architecture as the RL policy (grid CNN, set encoders, scalars)
  - scalars: an MLP on the global scalars + flags only (what dev-congestion-signal / dev-history-signal used)
Metrics on the test seeds (per slot, deterministic argmax pair): oracle pair matched, job rule matched, regret of the
chosen pair vs the slot's best (hold-to-end tails), against always-MDD-TECT. The full model is saved as a normal
checkpoint (bc_full_s<k>.pt) so evaluate.py can run it as a policy (run.sh step 3).

@par Usage
@code{.sh}
.venv/bin/python results/dev-oracle-bc/bc_train.py --bc-seeds 0,1,2 --device cuda > results/dev-oracle-bc/bc_train.out
@endcode
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES, PPOConfig  # noqa: E402
from models.network import SchedulingNetwork  # noqa: E402

STREAMS = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")


def load(seeds, data_dir, with_futures=False):
    """Obs, realized tails (slots, pairs), seed and slot of each row, pair names; with_futures also returns the
    redrawn-future tails (slots, futures, pairs) of ../dev-oracle-bc-ev/collect_ev.py data as a 6th element."""
    obs = {k: [] for k in STREAMS}
    tails, fut, seed_of, slot_of = [], [], [], []
    pairs = None
    for s in seeds:
        d = np.load(Path(data_dir) / f"s{s}.npz")
        pairs = list(d["pairs"])
        for k in STREAMS:
            obs[k].append(d[k].astype(np.float32))
        tails.append(d["tails"])
        if with_futures:
            if "tails_fut" not in d:
                raise ValueError(f"s{s}.npz has no redrawn-future tails (collect with ../dev-oracle-bc-ev/collect_ev.py)")
            fut.append(d["tails_fut"])
        seed_of += [s] * len(d["tails"])
        slot_of += list(range(len(d["tails"])))
    out = ({k: np.concatenate(v) for k, v in obs.items()}, np.concatenate(tails), np.array(seed_of),
           np.array(slot_of), pairs)
    return out + (np.concatenate(fut),) if with_futures else out


def label_tails(tails, fut, pairs, mode, default="MDD-TECT", z=1.0):
    """Tails the soft targets are built from (2026-10-07, handoff fix 7).

    - "oracle": the realized future's hold-to-end tails (one future: its argmax is about a coin flip in sign against
      the expected best, docs/experiments/review_1007/CREDIT_ASSIGNMENT_TRACE_1007.md);
    - "ev": the mean over the realized and the redrawn futures (an expected-value oracle);
    - "ev-safe": ev, but a pair other than @p default counts only by the lower confidence bound of its gain over the
      default, mean - z * standard error over the futures (paired), so the label switches away from the default only
      when the predicted gain beats its uncertainty (after Wei et al. 2026, arXiv:2605.23957).
    """
    if mode == "oracle":
        return tails
    allf = np.concatenate([tails[:, None, :], fut], axis=1)          # (slots, 1 + futures, pairs)
    ev = allf.mean(1)
    if mode == "ev":
        return ev
    if mode != "ev-safe":
        raise ValueError(f"unknown label mode {mode!r}")
    d = pairs.index(default)
    gain = allf[:, :, d:d + 1] - allf                                # > 0: the pair beats the default on that future
    se = gain.std(1, ddof=1) / np.sqrt(allf.shape[1])
    lcb = gain.mean(1) - z * se
    lcb[:, d] = 0.0
    out = ev[:, d:d + 1] - lcb
    out[:, np.arange(len(pairs)) != d] += 1e-9      # exact ties (pairs that play identically) go to the default
    return out


def head_index(pairs):
    j = np.array([JOB_HEAD_RULES.index(p.split("-")[0]) for p in pairs])
    m = np.array([MACHINE_HEAD_RULES.index(p.split("-")[1]) for p in pairs])
    return j, m


def soft_targets(tails, pairs, tau):
    best = tails.min(axis=1, keepdims=True)
    regret = 100.0 * (tails - best) / np.maximum(best, 1e-9)
    q = np.exp(-regret / tau)
    q /= q.sum(axis=1, keepdims=True)
    j, m = head_index(pairs)
    qj = np.stack([q[:, j == i].sum(1) for i in range(len(JOB_HEAD_RULES))], 1)
    qm = np.stack([q[:, m == i].sum(1) for i in range(len(MACHINE_HEAD_RULES))], 1)
    return qj.astype(np.float32), qm.astype(np.float32), regret


class ScalarMLP(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, len(JOB_HEAD_RULES) + len(MACHINE_HEAD_RULES)))

    def forward(self, obs):
        return self.net(torch.cat([obs["global_scalars"], obs["event_flags"]], -1))


def logits_of(model, obs):
    if isinstance(model, SchedulingNetwork):
        return model.actor_critic.actor(model.fusion(model.encoder(obs)))
    return model(obs)


def batch(obs, idx, device):
    return {k: torch.as_tensor(v[idx], device=device) for k, v in obs.items()}


def loss_fn(logits, qj, qm):
    nj = len(JOB_HEAD_RULES)
    return -(qj * F.log_softmax(logits[:, :nj], -1)).sum(-1).mean() - (qm * F.log_softmax(logits[:, nj:], -1)).sum(-1).mean()


def predict(model, obs, device, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(obs["global_scalars"]), bs):
            out.append(logits_of(model, batch(obs, np.arange(i, min(i + bs, len(obs["global_scalars"]))), device)).cpu())
    return torch.cat(out).numpy()


def fit(model, obs, qj, qm, tr, va, device, epochs, patience, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    qj_t, qm_t = torch.as_tensor(qj, device=device), torch.as_tensor(qm, device=device)
    rng = np.random.default_rng(0)
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
        lv = loss_fn(torch.as_tensor(predict(model, {k: v[va] for k, v in obs.items()}, device), device=device),
                     qj_t[va], qm_t[va]).item()
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if patience < epochs:      # early stopping: keep the best-validation weights; a capacity check keeps the last
        model.load_state_dict(best_state)
    return best, ep + 1


def scores(logits, tails, regret, pairs):
    nj = len(JOB_HEAD_RULES)
    j, m = head_index(pairs)
    pj, pm = logits[:, :nj].argmax(1), logits[:, nj:].argmax(1)
    chosen = np.array([np.flatnonzero((j == a) & (m == b))[0] for a, b in zip(pj, pm)])
    best = tails.argmin(1)
    mdd = pairs.index("MDD-TECT")
    return {"pair_match_%": 100 * (chosen == best).mean(),
            "job_rule_match_%": 100 * (j[chosen] == j[best]).mean(),
            "mean_regret_%": regret[np.arange(len(chosen)), chosen].mean(),
            "mddtect_regret_%": regret[:, mdd].mean(),
            "share_mddtect_%": 100 * (chosen == mdd).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", default="2000-2159")
    ap.add_argument("--bc-seeds", default="0,1,2")
    ap.add_argument("--tau", type=float, default=1.0, help="soft-label temperature, in regret %")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-dir", default=str(HERE / "data"),
                    help="npz folder (rq2-oracle-steady/data for the steady-state setting, added 10-07)")
    ap.add_argument("--test-seeds", default="0-39")
    ap.add_argument("--out-dir", default=str(HERE), help="where bc_scores / bc_full checkpoints go")
    ap.add_argument("--labels", choices=("oracle", "ev", "ev-safe"), default="oracle",
                    help="label tails: the realized future (oracle), the mean over redrawn futures (ev), or ev with "
                         "switches away from --default-pair only past z standard errors (ev-safe); ev* need "
                         "../dev-oracle-bc-ev data (added 10-07)")
    ap.add_argument("--safe-z", type=float, default=1.0, help="ev-safe: standard errors a switch must clear")
    ap.add_argument("--default-pair", default="MDD-TECT", help="ev-safe: the pair kept unless a switch clears z SE")
    ap.add_argument("--tag", default="", help="suffix for the outputs (bc_scores<tag>.csv, bc_full_s<k><tag>.pt), "
                    "e.g. a capacity check with --patience 1000 (added 10-07)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.train_seeds.split("-"))
    train_seeds = [s for s in range(lo, hi + 1) if (Path(a.data_dir) / f"s{s}.npz").exists()]
    tlo, thi = (int(x) for x in a.test_seeds.split("-"))
    test = list(range(tlo, thi + 1))
    out_dir = Path(a.out_dir)
    loaded = load(train_seeds, a.data_dir, with_futures=a.labels != "oracle")
    obs, tails, seed_of, slot_of, pairs = loaded[:5]
    tobs, ttails, tseed, tslot, _ = load(test, a.data_dir)
    ltails = label_tails(tails, loaded[5] if a.labels != "oracle" else None, pairs, a.labels, a.default_pair, a.safe_z)
    if a.labels != "oracle":
        agree = float((ltails.argmin(1) == tails.argmin(1)).mean())
        print(f"labels {a.labels}: the label's best pair is the realized future's best on {100 * agree:.1f}% of slots; "
              f"default {a.default_pair} best on {100 * float((ltails.argmin(1) == pairs.index(a.default_pair)).mean()):.1f}%")
    qj, qm, _ = soft_targets(ltails, pairs, a.tau)
    _, _, tregret = soft_targets(ttails, pairs, a.tau)
    print(f"train {len(train_seeds)} seeds / {len(tails)} slots, test {len(test)} seeds / {len(ttails)} slots; "
          f"tau {a.tau}% regret; mean regret of the slot's 2nd-best pair (test) "
          f"{np.sort(tregret, 1)[:, 1].mean():.2f}%")
    rows = []
    for k in [int(x) for x in a.bc_seeds.split(",")]:
        rng = np.random.default_rng(k)
        va_seeds = set(rng.choice(train_seeds, size=max(1, int(0.15 * len(train_seeds))), replace=False))
        va = np.flatnonzero(np.isin(seed_of, list(va_seeds)))
        tr = np.flatnonzero(~np.isin(seed_of, list(va_seeds)))
        for kind in ("full", "scalars"):
            torch.manual_seed(k)
            model = (SchedulingNetwork() if kind == "full" else ScalarMLP(18 + 6)).to(a.device)
            t0 = time.time()
            vloss, eps = fit(model, obs, qj, qm, tr, va, a.device, a.epochs, a.patience, 3e-4 if kind == "full" else 1e-3)
            sc_tr = scores(predict(model, {kk: v[tr] for kk, v in obs.items()}, a.device), tails[tr],
                           soft_targets(tails[tr], pairs, a.tau)[2], pairs)
            sc = scores(predict(model, tobs, a.device), ttails, tregret, pairs)
            rows.append({"model": kind, "bc_seed": k, "epochs": eps, "val_loss": vloss, "train_pair_match_%": sc_tr["pair_match_%"],
                         **sc, "minutes": (time.time() - t0) / 60})
            print(json.dumps(rows[-1]), flush=True)
            if kind == "full":
                import train as T
                from rewards import load_reward
                ck = out_dir / f"bc_full_s{k}{a.tag}.pt"
                T.save_checkpoint(ck, model, torch.optim.Adam(model.parameters()), 0, PPOConfig(),
                                  load_reward(str(REPO / "env/config/rewards/tardiness.json")).name, (15, 256))
    import pandas as pd
    pd.set_option("display.width", 200)
    df = pd.DataFrame(rows)
    print(f"\n== test seeds {a.test_seeds}, per slot (deterministic argmax pair); mean over BC seeds")
    print(df.groupby("model")[["epochs", "train_pair_match_%", "pair_match_%", "job_rule_match_%", "mean_regret_%",
                               "mddtect_regret_%", "share_mddtect_%"]].mean().round(2).to_string())
    df.to_csv(out_dir / f"bc_scores{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
