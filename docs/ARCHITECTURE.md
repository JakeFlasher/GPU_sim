# Architecture

## System boundary

The project has two deliberately separate paths: simulation and silicon-label
capture. They share identities and metric contracts at evaluation time, but
NCU values do not enter the simulator.

```text
                              static SASS control bits
                                         |
workload -> NVBit tracer -> protobuf trace + allocation/sampling sidecars
                                         |
                                         v
arch card + microbench cards ------> Rust simulator ------> predictions
                                         |                    |
                                         |                    v
workload ------------------------------> NCU ------------> evaluator
                                                    admitted labels only
```

The prohibited edge is NCU label value -> simulator input. Metric names,
units, aggregation contracts, and output shapes may be shared because they
define what is predicted, not the answer.

## Trace and static inputs

`tools/nvbit_tracer/tracer_tool/` instruments CUDA launches through NVBit and
writes the `traces_enhanced` protobuf format. The trace contract includes SASS
records, active-lane masks, compressed per-lane addresses, memory operation
classes, allocation identity, launch placement, and completeness/sampling
metadata.

`sim-trace` decodes that format directly with committed Prost bindings. It
reconstructs lane addresses, checks the reader-level wire contract, validates
allocation identity needed by replay, and reads thread blocks in launch order.
Protobuf schema changes require regeneration and schema-pin updates in the same
change. The planned C++ protobuf shim is not used.

`tools/sass_control/` extracts static control blocks from cubins. `sim-sm`
loads the resulting sidecars fail-closed, re-decodes the packed high word, and
joins records to trace instructions by function and program counter. Stall,
barrier, yield, and reuse information comes from these sidecars rather than a
reconstructed scoreboard.

Architecture and calibration inputs come from `arch_profiles/cards/` and
microbenchmark-derived cards such as `data/pchase/`. Typed `research_gap`
values stop a configuration from silently falling back to another
architecture.

## Event-driven execution

`sim-engine` owns a binary-heap event queue ordered by `(time, insertion_seq)`.
The clock advances directly to the next event; same-time events remain FIFO and
reproducible. There is no global per-cycle tick.

`sim-sm` builds per-function issue schedules and replays the placement recorded
by the trace. Its actor form models four SMSPs, control-bit-driven eligibility,
barriers, and memory request/response suspension. `replay_kernel` is the
library-level replay entry point.

The current hierarchy has three modes:

- `Flat`: latency-only ablation without cache state;
- `L1Const`: per-SM sectored L1 with constant fill latency;
- `L1SharedL2`: per-SM L1s over a shared sectored L2 and aggregate DRAM
  bandwidth server.

In the full path, L1 misses become timestamped requests. L2 state changes only
when the ordered L2 event is handled; responses clear the originating SM's
fills or barriers. This keeps shared-cache ordering inside the deterministic
event system.

`sim-mem` provides the cache state machines and counters:

- 32-byte sectors and 128-byte lines;
- finite L1 MSHRs and fill merging;
- cache-operator eligibility derived from static SASS;
- a shared sectored, set-associative L2;
- read-sector demand observed by the DRAM bandwidth model.

Undocumented replacement, associativity, hash, and latency choices remain
explicit architecture hypotheses or typed gaps. NCU labels are not used to
fill them per scored cell.

## Output contracts

`sim-emit` converts raw replay counters into typed NCU-shaped predictions. Raw
counts remain in the simulator; rate numerators and denominators are composed
at the output boundary. A real zero, an NCU-style not-applicable rate, and a
metric unavailable from the selected replay mode are distinct states.

Seven descriptors are implemented:

- `smsp_inst_executed_sum`;
- `traffic_gmem_ld_sectors`;
- `traffic_gmem_st_sectors`;
- `cache_hit_rate_l1tex_read_sector`;
- `cache_hit_rate_l1tex_sector`;
- `cache_hit_rate_l2_sector`;
- `traffic_dram_read_sectors`.

Each descriptor is checked against `tools/local_labels/metric_registry.json`.
The registry contains 61 capture targets, but registry membership does not mean
the simulator implements the metric.

The sole publication-binding headline, `exec_cycles_gpc_max` /
`gpc__cycles_elapsed.max`, is required but unimplemented. Cycles MAPE ≤10%
meets the engineering target, >10% and ≤15% is floor-only, and >15% is
publication failure. `KernelReplayResult.max_cycles` is a directional
diagnostic only. Emission is blocked until silicon evidence establishes both
single-launch mapping and multi-launch aggregation semantics. The seven
implemented descriptors remain reported secondary outputs and are ungated.

## Label and evidence path

`tools/local_labels/` owns NCU capture, protocol pins, sidecar validation,
repeatability, typed exclusions, campaign sealing, and evidence verification.
The v5 label authority has three members:

- `workloads/simbench_bw/schemas/local_label_sidecar.schema.json`, the
  per-launch contract;
- `workloads/simbench_bw/schemas/local_label_admitted.schema.json`, the
  per-cell admitted contract;
- `tools/local_labels/metric_registry.json`, the metric identity, unit, and
  61-entry capture-ceiling contract.

Timing labels travel as registry-resolved `unit_tagged_metrics`. The active
`workloads/simbench_bw/schemas/measurement_manifest.schema.json` is a separate
v2 workload-provenance contract; it is not a timing-label contract and does
not migrate to v5. The retired CUPTI dual-source duration v1 contract is frozen
under `workloads/simbench_bw/schemas/historical/` for retained evidence only.

Tracked historical fixtures under `evidence/` are authenticated by canonical
tracked path, SHA-256 manifest membership, and a clean working-tree check.
Historical absolute paths remain byte-exact in ledger rows and may resolve only
through an authority-closed versioned alias table with mandatory hash equality.
`evidence/path_aliases.v2.json` is active for its M1 scope.
`data/local_labels/campaign/frozen_v1_source_aliases.v1.json` is the M6-local
versioned authority for the unavailable
`124ce358733f97ba6e0af808d10ab69ffe4d5a78` registry and sidecar objects; the
`f74ac4e8750f970ab3bb5d7da54ec9db2df0f4ef0f5fb22b5f0f6f90898790ff`
and requires exact canonical-file digest and size equality rather than falling
back to ambient worktree bytes or Git object discovery.
`evidence/path_aliases.v1.json` is retained but currently cites a missing ADR
whose number is occupied by another decision, so its relocation chain is
blocked pending an explicitly accepted versioned repair. POSIX mode `0444` is
advisory, not authority, because Git cannot reproduce it for ordinary files.

## Integration gaps

The core is currently a set of libraries and tests, not a runnable product.
The next integration work must add, without weakening the boundaries above:

- a thin CLI that loads trace, sidecar, and architecture inputs and invokes
  `replay_kernel`;
- aggregation and emission wiring across launches;
- a cycles-semantics study before the primary emitter;
- an evaluation join that compares predictions with admitted NCU labels;
- a performance benchmark and enforceable events-per-second threshold.

These are feature milestones, not claims made by the current implementation.

## Contract ownership

| Contract | Authority | Mutability |
|---|---|---|
| NVBit trace wire format | `tools/nvbit_tracer/traces_enhanced/` plus committed `sim-trace` Prost bindings | Producer and consumer change together |
| Static SASS control | `tools/sass_control/` sidecars and `sim-sm` loader | Fail closed on missing or mismatched records |
| Architecture parameters | `arch_profiles/` and committed microbenchmark cards | Typed gaps instead of guessed fallback values |
| Workload provenance | active measurement-manifest v2 schema | Additively extensible active contract |
| Silicon labels | three-part v5 label authority | Additively extensible, lockstep tested |
| Historical CUPTI duration | frozen v1 schema and hash manifest | Byte-frozen; explicit version dispatch only |
| Simulator predictions | `sim-emit` descriptor set | Seven secondary descriptors implemented; primary cycles absent |
| Migrated Round-28 bytes | `evidence/MANIFEST.sha256`; v1 alias resolution remains blocked on missing decision authority | Hash and clean-tree verified; relocation fails closed |
