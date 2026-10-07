# dev-history-signal: would memory over past slots help the slot policy? (2026-10-06)

Question (user, 10-06): if the policy's ATC-vs-MDD mistakes need temporal information, a recurrent or attention layer
over past slots would be the fix. Cheap check before any architecture change: does the state of the previous slots
predict which family to play beyond the current observation?

**Answer: no.** The observed state of the last slots adds nothing beyond the current observation. With 40 instances
the 72 extra columns cost a little (out-of-fold AUC -0.02 to -0.04). The one large effect (the previous slot's pair) is
leakage from the oracle's hindsight, not a memory signal (see below).

## Method (`analyze.py`; no new simulation)

- **Data:** the 960 per-slot states and targets of `../dev-congestion-signal` (oracle trajectories on B2 seeds
  0-39). Slots 0-1 are dropped, so every feature set has its lags (880 slots).
- **Feature sets:**
  - *now:* obs v3 global scalars + flags
  - *+history:* now + the same columns at slots k-1 and k-2, plus the change since k-1 and since k-4. This is what a
    GRU or temporal attention over the slot sequence could summarize.
  - *+prev pair:* now + the pair played in slot k-1, one-hot
- **Targets:** dev1 / dev4 (ATC-TECT vs MDD-TECT swapped into 1 slot / 1 h of the oracle schedule) and hold (to the
  end).
- **Models and scoring:** the same models and seed-grouped out-of-fold scoring as dev-congestion-signal.

## Results (`analysis.out`)

| target | AUC now (logistic / boosting) | +history: change | +prev pair: change (leaky) |
|---|---|---|---|
| dev1 | 0.674 / 0.668 | -0.043 / -0.020 | +0.145 / +0.144 |
| dev4 | 0.705 / 0.712 | -0.030 / -0.017 | +0.133 / +0.120 |
| hold | 0.697 / 0.704 | -0.034 / -0.019 | +0.205 / +0.203 |

- **History adds no information.** The lagged columns that correlate with the target do so about as strongly as the
  current values (WIP one slot ago +0.26 with dev4, now +0.25). They repeat the present state rather than add to it.
- **The previous pair is leakage.** It is the oracle's choice, and the oracle picks each slot with the future of the
  instance in hand (it tries every pair to the end of the episode). So its last choice carries hindsight about the
  instance that a real policy's own previous action would not. This does show one thing: the right family is
  persistent within an instance. If ATC was right for the last slot, it is likely right now. An instance has a type
  that lasts.

## Reading

- Memory of the observed state is not what is missing, at least at the level of the global scalars. A GRU or temporal
  attention over slot observations would have nothing extra to work with. The architecture change is not supported
  by this evidence.
- The persistence finding points at a different kind of memory: inferring the instance's type from the policy's own
  outcomes, e.g. jobs going late faster than expected under the current pair. That would need the realized
  consequence of the policy's own choices in the input (tardiness accrued in the last slot, the previous action). It
  can only be tested on the policy's own trajectories, not the oracle's, and is left open.
- The check covers only the global scalars. The job table and the grid could hold temporal patterns the scalars miss.
  A check of the whole observation is behavior cloning of the oracle (supervised, full network vs scalars only);
  not run.
