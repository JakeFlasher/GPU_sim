# Archived research reports

This directory is the retained record of dated reports that previously lived
in `docs/`. They are useful when a result, command, or design decision must be
reconstructed, but they are **not** current capability claims, execution plans,
validation gates, or automatic work queues. Read `docs/STATUS.md` first.

The reports mix observations, hypotheses, negative results, and superseded
interpretations. A report can therefore be correct as a historical record and
still be unsuitable as a present-day model contract. In particular:

- reported projection-only timings exclude trace acquisition, replay, and
  native measurement unless the report says otherwise; they are not simulator
  speedups;
- the base/large sets are related cohorts, not an architecture-wide
  out-of-distribution test;
- NCU values are comparison labels, never features or route-selection inputs
  for the same scored row;
- empirical coverage intervals are scenarios, not confidence intervals or
  universal bounds;
- the four physical boundaries in the current status remain open.

The machine-readable campaign records and raw captures remain in their
original `results/`, `data/`, and `traces/` locations (many are intentionally
ignored by Git). The archived Markdown is an indexable explanation, not a
freshness or identity manifest. New work should update the compact entry
points rather than append another dated handoff report.

## Topic map

| Topic | Representative reports |
| --- | --- |
| Simulator architecture and prior art | `gpu_simulator_*`, `accelsim_*`, `huerta_*`, `curry_*`, `architecture_*` |
| Native SM86 capture and replay | `sm86_current_*`, `sm86_replay_*`, `sm86_measurement_*`, `sm86_external_*` |
| NCU metric projection | `sm86_ncu_*`, `sm86_frontier_*`, `sm86_tile_*`, `sm86_shape_*`, `sm86_stage_*` |
| Clock, cache, and endpoint alternatives | `sm86_clock_*`, `sm86_cache_*`, `sm86_endpoint_*`, `sm86_regime_*`, `sm86_interaction_*` |
| NAK, PMA, and CUPTI probes | `sm86_nak_*`, `sm86_cupti_*`, `sm86_native_cupti_*`, `sm86_pm_*` |
| Earlier custom/native probes and audits | `custom_*`, `adversarial_*`, `remaining_*`, `*_probe_*`, `*_audit_*` |

Use Git history for exact pre-consolidation paths. The compact history index at
`docs/HISTORY.md` records why this directory supersedes the former flat report
set.
