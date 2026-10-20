# Architecture and module map

The simulator follows the event-driven separation used by validated GPU
simulators: trace input, SM issue state, sectored memory state, deterministic
events, and post-replay attribution are separate boundaries. The Rust 2021
parent files are small facades; named child sections are included in the parent
scope where private fields and split `impl` blocks must remain compatible.

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
  `preflight.rs` hold validation, artifact snapshots, compact instruction and
  memory records, sampling provenance, and allocation preflight respectively.
* `async_sidecars.rs`, `anchored.rs`, and `obligations.rs` retain producer
  sidecars and trace-contract checks.
* Invariant: malformed or incomplete trace contracts fail closed; no timing or
  cache state is mutated by this crate.

### `sim/sm`

* `issue.rs` is the SMSP issue-model facade. `issue/diagnostics.rs`,
  `requests.rs`, `frontend.rs`, `pipelines.rs`, `async_copy.rs`,
  `warp_state.rs`, `core_init.rs`, and `core_step.rs` are lexical child
  sections for diagnostics, request records, front end, readiness, async
  copies, warp state, construction, and cycle progression.
* `replay.rs` is the launch/replay facade. `replay/coverage.rs`, `fast_warp.rs`,
  `results.rs`, `placement.rs`, and `session.rs` separate coverage, fast-path
  execution, result records, placement, and stream sessions.
* `system.rs` coordinates interleaved SM/L2/DRAM events. `system/queues.rs`,
  `dram.rs`, `observers.rs`, and `execution.rs` isolate pending events,
  service, observer accounting, and event coordination.
* `metrics.rs` owns `TraceMetrics`; `metrics/locality_mrc.rs` contains
  trace-local Fenwick/HOTL observers, `pipeline_activity.rs` contains activity
  counters, and `tests.rs` contains metric tests.
* Invariant: `SmCore`, residency, barriers, and issue timestamps mutate active
  timing state; locality observers are shadow/report-only and cannot change it.

### `sim/mem`

* `l1.rs` is the sectored L1 facade. `l1/config_counters.rs`, `core.rs`, and
  `accessors.rs` group configuration/counters, mutation, and access expansion.
* `l2.rs` is the shared L2 facade. `l2/config_counters.rs` groups card/config
  and counter types; `l2/core.rs` contains request, fill, and completion logic.
* `payload.rs` and `l1_visibility.rs` define payload and visibility contracts.
* Invariant: L1/L2/DRAM mutation occurs only at their event timestamps;
  report-only counters are not allowed to change cache state.

### `sim/semantics` and `sim/engine`

* `semantics/state.rs` re-exports `state/cache_boundary.rs`,
  `typed_events.rs`, and `core_boundary.rs`; these are typed report/state
  contracts, not schedulers.
* `sim-engine` supplies dependency/resource event ordering and does not own
  architecture-specific cache policy.

### `sim/direct_run`

* `lib.rs` is the run/report facade; `lib/reports.rs` holds report structures.
* `metrics.rs` is a compact facade over `metrics/l1.rs`, `l2_dram.rs`, and
  `pipeline_ipc.rs`; all three are post-replay, report-only attribution.
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
