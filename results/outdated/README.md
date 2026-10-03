# Outdated results (moved 2026-10-03)

Everything here was produced on builds before the 10-02/10-03 C# audit fixes and the AGV/rule changes, so none of it
is comparable to runs on the next build. Kept for reference and for the thesis's historical/methods discussion only.

What changed (details: `docs/handoffs/HANDOFF_2026-10-02_csharp_audit_code_fixes.md`, section 7):
- AGV dispatch now picks the nearest of idle AND returning AGVs (was: any idle AGV first).
- A stall that teleports an AGV to parking now costs the estimated drive-back time (was free).
- AGV timers use the episode clock (float32 `Time.fixedTime` drifted after 2^17 s of player life).
- Jobs delivered to a machine as it fails are re-routed; layout A phantom dock removed; layout J junction boxes fixed.
- Logging: penalized flow (G1), zone congestion (G4), trips at dropoff (G13), utilization incl. in-progress op (G10),
  NaN for empty means (G9), new results.csv columns.
- The dispatching rule set is being replaced (`docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md`).

Folder names are unchanged inside `outdated/`, so a path `results/<name>` in an older doc, script or memory note is now
`results/outdated/<name>`. Scripts in `results/scripts/` still default to the old paths.
