# Architecture and module map

The simulator follows the event-driven separation used by validated GPU
simulators: trace input, SM issue state, sectored memory state, deterministic
events, and post-replay attribution are separate boundaries. The Rust 2021 parent files are small facades declaring real child modules. Child modules use explicit re-exports and visibility (`pub`, `pub(crate)`, or `pub(super)`) so split `impl` blocks retain the original APIs.

## Native replay path

```text
NVBit/SASS + sidecars
        -> sim-trace (validated trace reader)
        -> sim-sm (SMSP issue, warp, barrier, async state)
        <-> sim-mem (L1/L2/DRAM event state)
        -> sim-engine (event ordering)
        -> sim/direct_run (reports and NCU joins)
```

NCU is read only after replay. Architecture cards choose assumptions; they are
not inferred from NCU values.

## Rust crate codemap

### `sim/trace`

* `reader.rs` is the facade for trace lineage and directory loading.
* `reader/error.rs`, `snapshot.rs`, `compact.rs`, `sampling.rs`, and
  `directory.rs` hold validation, artifact snapshots, compact instruction and
  memory records, sampling provenance, and directory loading respectively.
  `reader/preflight_access.rs`, `preflight_allocation.rs`, and
  `preflight_validation.rs` are direct children of `reader`; they keep
  payload, allocation, and semantic validation separate without a pass-through
  module.
* `async_sidecars.rs`, `anchored.rs`, and `obligations.rs` retain producer
  sidecars and trace-contract checks.
* Invariant: malformed or incomplete trace contracts fail closed; no timing or
  cache state is mutated by this crate.

### `sim/sm`

* `issue.rs` is the SMSP issue-model facade. `issue/diagnostics.rs`,
  `request_types.rs`, `core_state.rs`, `frontend.rs`, `pipelines.rs`,
  `async_copy.rs`, `warp_state.rs`, `core_init.rs`, and `core_mem.rs` are
  direct child modules for diagnostics, request records, `SmCore` state, the
  front end, readiness, async copies, warp state, construction, and memory.
  Cycle scheduling is implemented directly in the named children
  `schedule_frontend.rs`, `schedule_barriers.rs`, and `schedule_control.rs`.
* `replay.rs` is the launch/replay facade. `replay/coverage.rs`, `fast_warp.rs`,
  `results.rs`, and `placement.rs` separate coverage, fast-path execution,
  result records, and placement. `session_fast.rs`, `session_l2.rs`,
  `session_results.rs`, and `session_stream.rs` are direct session children.
* `system.rs` coordinates interleaved SM/L2/DRAM events. `system/queues.rs`,
  `dram.rs`, `observers.rs`, `event_loop.rs`, `event_helpers.rs`, and
  `execution_completion.rs` directly isolate pending events, service, observer
  accounting, event processing, helper validation, and completion.
* `metrics.rs` owns `TraceMetrics`; `metrics/locality_mrc.rs` contains
  trace-local Fenwick/HOTL observers, `pipeline_activity.rs` contains activity
  counters, and `tests.rs` contains metric tests.
* Invariant: `SmCore`, residency, barriers, and issue timestamps mutate active
  timing state; locality observers are shadow/report-only and cannot change it.

### `sim/mem`

* `l1.rs` is the sectored L1 facade. `l1/config_counters.rs`,
  `cache_state.rs`, `cache_ops.rs`, and `accessors.rs` group
  configuration/counters, cache state and lookup/fill mutation, visibility and
  merge operations, and access expansion.
* `l2.rs` is the shared L2 facade. `l2/config_counters.rs` groups card/config
  and counter types; `l2/core.rs` contains request, fill, and completion logic.
* `payload.rs` and `l1_visibility.rs` define payload and visibility contracts.
* Invariant: L1/L2/DRAM mutation occurs only at their event timestamps;
  report-only counters are not allowed to change cache state.

### `sim/semantics` and `sim/engine`

* `semantics/state.rs` declares `cache_boundary`, `core_boundary`,
  `typed_events_serialization`, and `typed_events_validation`. These are typed
  report/state contracts, not schedulers.
* `sim-engine` supplies dependency/resource event ordering and does not own
  architecture-specific cache policy.

### `sim/direct_run`

* `lib.rs` is the run/report facade; `reports.rs` holds report structures and
  `runner_config.rs` plus `runner_execute.rs` hold configuration and execution
  entry points directly.
* `metrics.rs` declares `l1`, `l2_dram`, `launch_occupancy`, and `pipeline_ipc`;
  the IPC module further separates SM86 pipeline helpers. All are post-replay,
  report-only attribution.
* `emit.rs`, `prepared.rs`, and `resources.rs` provide serialization, prepared
  execution, and external launch-resource observations.
* Invariant: metric emitters never feed NCU targets into replay.

## State boundaries

1. Trace records and architecture cards are immutable model inputs.
2. `SmCore` and memory models own active cycle/cache state.
3. Event queues define ordering and completion; issue, completion, and launch
   observations are not collapsed into one timestamp.
4. Locality/MRC observers consume declared trace/event streams and expose
   uncertainty without mutating active state.
5. `sim/direct_run` joins replay output with independent NCU profiles only after
   execution and labels exact versus ordered-fallback joins.

See [`COMPONENTS.md`](COMPONENTS.md) for the compact ownership table and
[`TERMINOLOGY.md`](TERMINOLOGY.md) for metric definitions.

## Nested module examples

`sim/sm::issue` directly owns `schedule_frontend`, `schedule_barriers`, and
`schedule_control`; `sim/sm::replay` directly owns stream, fast-path, L2, and
result session modules; `sim/trace::reader` directly owns access, allocation,
and semantic preflight modules; and `sim/direct_run::metrics::pipeline_ipc`
contains the common projection plus descriptively named SM86 helper modules.
These are ordinary Rust modules, not textual source concatenation or
pass-through wrappers.
