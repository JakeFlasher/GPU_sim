# SM86 regime router (2026-09-30)

`tools/sm86_regime_router.py` integrates the committee into an opt-in fast
endpoint path. It selects a source using only an external regime observation:

* low regime: frozen frontier/archetype reference;
* high-clock cold: two-component clock elasticity;
* high-clock warm: same-regime native CUDA-event endpoint ratio.

The threshold is fixed at `active_clock / 1350 >= 1.15`; the source route and
coverage status are emitted for every row. NCU score labels are used only after
route selection for reporting.

## Live high-regime scores

| phase | GPC MAPE | duration MAPE | route |
|---|---:|---:|---|
| cold, 15 rows | **6.637%** | **7.733%** | clock elasticity |
| warm, 15 rows | **10.398%** | **11.702%** | native-event ratio |

Reports:

* `results/alpha/sm86_regime_router_goal_20260930/high_cold.json`
* `results/alpha/sm86_regime_router_goal_20260930/high_warm.json`

This is an endpoint committee router, not a cycle-accurate mechanism model.
Pipe/ABI/cache ownership and unified endpoint semantics remain explicit
unknowns. The native replay and promoted default are unchanged.
