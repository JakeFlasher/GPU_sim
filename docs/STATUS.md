# Current status

Updated 2026-09-27. Source-backed capabilities, not a declaration that all
research is complete. Start with [memory](PROJECT_MEMORY.md); use
[history](HISTORY.md) for dated evidence and superseded claims.

## Runnable paths

| Path | Implemented | Boundary |
| --- | --- | --- |
| Native SM86 | NVBit/enhanced protobuf -> SM, L1/L2, event replay -> `sim-run` | Fixed values/control/address stream; physical retirement tails incomplete |
| Native resource override | Carded scheduler; scalar or strict `trace_sass_v1` issue-width override, also in prepared replay | Hypothetical trace-safe width only; no unused latency/resource knobs |
| Custom `simt_u32_v1` | Dependency/resource overlap, resident groups, separate global/shared words, atomic add, broadcast/reduction, bounded CTA barriers | Fixed ISA; no general compiler, weak memory or native cache timing |
| Custom `simt_its_v1` | Per-lane PCs/predicates, reconvergence, indirect loads/stores, optional versioned execution trace | Single group, functional only; not native SASS regeneration |
| `ptx_subset_v1` frontend | Bounded text-to-`simt_its_v1` compiler for lane ID, integer arithmetic, predicates, labels, indirect u32 load/store, exit | Functional subset only; no ABI, calls, atomics, vector types, arbitrary PTX or SASS |
| Dirty-sector writeback queue | Transactional capacity/timing preflight, fixed-point byte service, same-line fetch hazard, tokenized completion in `sim-mem` | Reusable mechanism primitive only; not wired into native SM86 replay or coherence ownership |
| PhaseFuse fast frontier | Trace-derived scalar-region certificate/refusal report via `sim-run --fast-frontier-only` | Conservative analysis only; skips native timing and cannot certify memory, barriers, dynamic pipes or silicon |

Custom barriers require matching lengths/positions/IDs, one use per ID, and
all participants resident. Same-phase shared conflicts and inter-group global
read/write conflicts reject. Full contracts: [programs](../arch_profiles/programs/README.md),
[specs](../arch_profiles/specs/README.md), [configuration](archprofile.md).

## Observability, not new hardware mechanisms

Native reports retain separate issue/event/completion/launch frontiers and
unmodeled-tail flags, trace and lane counts, branch successors, CFG/sharing
observations, occupancy projections, readiness causes, and cache/read-demand
counters. Legacy geometry falls back to the coverage witness when resources
are absent. NCU labels and CUPTI launch metadata remain separate observations.

`native_metrics` has outer schema 1 / metric surface 2. New
`*_completion_frontier_cycles`, `*_completion_scenario_end_cycles`, and
`*_completion_service_width_cycles` rows are **diagnostic formulas**, not
writeback or coherence events. Missing decoded/known topology stays unavailable.
The native pipe policy `min_selected_then_lexical_v1` chooses by cumulative
candidate counts, not issue time, SMSP resources, latency or queue state. The
custom scheduler additionally supports an explicit alternative-resource set
with latency, initiation-interval and in-flight-capacity selection; that is a
hypothetical model and does not recover native SM86 physical-pipe IDs. Details
and limitations: [experimental diagnostics](experimental_diagnostics.md).

ITS trace fields identify `custom_its_dynamic_v2` / version 2. The current
schema validates a broad output shape; its fields are not a native adapter.
Capture is bounded at 100k records and 32 MiB of accumulated serialized record
bytes (not whole-process memory); `lane_blocked_by` and explicit join/resume
PCs keep barrier state visible. [Schemas and inputs](../arch_profiles/README.md).

The PhaseFuse lane is deliberately separate from native replay. It preserves
trace/control/card identity, emits exact candidate/refusal populations for its
declared recurrence, and never consumes NCU values. It is a fast analysis
front-end, not a replacement timing model. [Prototype and SOTA study](sm86_fast_exact_prototype_20260927.md).

The tracer resolves exact module/library owners for stable regular files.
Anonymous/JIT/deleted origins can remain unavailable. Active CUDA graph
launches fail explicitly (`graph_node_capture_unsupported`).

## Evidence and limits

- Native canary, custom executables, Rust tests, and schema checks are engineering
  evidence; commands and latest results are in [testing](testing.md) and [history](HISTORY.md).
- The dated [measurement pair](sm86_measurement_boundary_20260924.md) matches
  197,120 instructions, not complete elapsed timing.
- The [eight-workload table](sm86_holdout_replay_20260926.md) is a retained
  development replay, not an untouched holdout. It matches instruction totals;
  its GPC discrepancy compares an incomplete modeled frontier with a profiler
  endpoint. The L2 domains differ, so no L2 accuracy score is valid.
- [Speed cohort](sm86_speed_cohort_20260927.md) records historical phase timings;
  the filename is retained although the record was committed September 26.
  No universal speedup or external-simulator ranking follows.
- [Issue-width study](sm86_issue_width_design_study_20260923.md) is a negative
  model-relative result. The [tail sensitivity](sm86_tail_sensitivity_20260926.md)
  is arithmetic under assumed coefficients, not a consequential hardware study.
- [PhaseFuse](sm86_fast_exact_prototype_20260927.md) reports bounded fast-only
  trace analysis; its wall times exclude native timing replay by design.

## Open work and scope

Native scheduled writeback/visibility/coherence integration, native physical-pipe routing,
execution-driven SASS regeneration, same-domain independent timing/cache
validation, and general compiler/ISA support remain open. Their absence is not
a proof of impossibility. A bounded `ptx_subset_v1` functional frontend and
custom alternative-resource scheduler plus a transactional dirty-sector queue
are implemented as explicit bounded seams, not as substitutes for those native
mechanisms; see [options](sm86_sota_simulator_options_20260926.md).

No full Accel-Sim/MGPUSim comparison has been performed; the [baseline matrix](baseline_matrix_20260924.json)
is non-scoring. Earlier runs excluded AMD/SM120 and full external execution;
follow the latest explicit user scope rather than silently reviving or
permanently forbidding those tasks. SOL/campaign data and old research remain
retained. [Roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md)
contains research choices, not an automatic endless queue.
