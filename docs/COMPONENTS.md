# Components and invariants

| Crate / module | Concrete responsibility | State boundary |
|---|---|---|
| `sim/trace::reader` | Protobuf/JSON trace loading, lineage, compact memory records | immutable input; fail-closed validation |
| `sim/trace::reader::snapshot` | Artifact identity and JSON-shape limits | report-only |
| `sim/trace::reader::compact` | Validated warp/instruction/memory representations | report-only |
| `sim/trace::reader::preflight_access` | Memory-payload and access validation | report-only |
| `sim/trace::reader::preflight_allocation` | Allocation and manifest validation | report-only |
| `sim/trace::reader::preflight_validation` | Trace-shape and semantic validation | report-only |
| `sim/arch` | Architecture cards, scheduler resources, functional instruction domains | card-selected assumptions |
| `sim/sm::issue` | SMSP issue, CGGTY selection, barriers, readiness, async copies | mutates issue/residency timing |
| `sim/sm::issue::frontend` | I-cache, stream buffer, front-end admission | mutates instruction timing |
| `sim/sm::issue::pipelines` | Scalar/memory pipeline readiness and fixed latencies | mutates issue timing |
| `sim/sm::issue::core_mem` | Memory dispatch and observer hooks | mutates request/cache state |
| `sim/sm::issue::schedule_frontend` | Front-end issue and writeback scheduling | mutates issue timing |
| `sim/sm::issue::schedule_barriers` | Barrier, readiness, and admission scheduling | mutates issue/barrier timing |
| `sim/sm::issue::schedule_control` | Control-flow, retirement, and cycle advancement | mutates issue timing |
| `sim/sm::issue::request_types` | Replay request records and decoding | immutable request contracts |
| `sim/sm::issue::core_state` | `SmCore` state and memory-response contracts | mutates issue/cache state |
| `sim/sm::replay` | Launch placement, replay sessions, result records | mutates replay session state |
| `sim/sm::replay::session_fast` | Fast replay session state | mutates the selected replay session |
| `sim/sm::replay::session_l2` | Shared-L2 replay session state | mutates replay/cache timing |
| `sim/sm::replay::session_results` | Replay result extraction | report-only |
| `sim/sm::replay::session_stream` | Trace-stream replay session state | mutates replay timing |
| `sim/sm::system` | Interleaved SM/L2/DRAM event loop | mutates event and service state |
| `sim/sm::system::event_loop` | Interleaved SM/L2/DRAM event processing | mutates event/service state |
| `sim/sm::system::event_helpers` | Event validation and counter helpers | report-only validation |
| `sim/sm::system::execution_completion` | Response and completion scheduling | mutates event/service state |
| `sim/sm::system::queues` | Pending responses and due-time queues | mutates event queue state |
| `sim/sm::system::dram` | Carded DRAM bandwidth/service accounting | mutates DRAM events |
| `sim/sm::system::observers` | L2 source and admission observer records | report-only accounting |
| `sim/sm::metrics::locality_mrc` | Fenwick MRC, HOTL footprint inversion, Tier-2/Tier-3 shadows | report-only |
| `sim/sm::metrics::pipeline_activity` | Warp lifetime and pipeline activity counters | report-only |
| `sim/mem::l1::cache_state` | L1 tags, fills, lookup, and dirty residency | mutates L1 state |
| `sim/mem::l1::cache_ops` | L1 visibility, merge, and flush operations | mutates L1 state |
| `sim/mem::l1` | Sectored L1 facade and access expansion | mutates L1 state |
| `sim/mem::l2` | Shared L2 tags, sector fills, write packets | mutates L2 state |
| `sim/mem::l1_visibility` | Payload/visibility handoff | event-boundary contract |
| `sim/semantics::state` | Typed cache/core/event boundary snapshots | report-only |
| `sim/semantics::state::typed_events_serialization` | Typed-event serialization contracts | report-only |
| `sim/semantics::state::typed_events_validation` | Typed-event validation contracts | report-only |
| `sim/engine` | Dependency/resource event ordering | event ordering only |
| `sim/direct_run::emit` | Stable JSON and counter emission | report-only serialization |
| `sim/direct_run::reports` | Post-replay report structures | report-only |
| `sim/direct_run::runner_config` | Replay configuration inputs | immutable configuration |
| `sim/direct_run::runner_execute` | Replay execution and launch accounting | orchestration state |
| `sim/direct_run::metrics::l1` | L1/TEX counter attribution | report-only |
| `sim/direct_run::metrics::l2_dram` | L2/DRAM and readiness attribution | report-only |
| `sim/direct_run::metrics::pipeline_ipc` | IPC and pipeline projections | report-only |
| `sim/direct_run::metrics::pipeline_ipc::pipeline_sm86_core` | SM86 instruction/warp projections | report-only |
| `sim/direct_run::metrics::pipeline_ipc::pipeline_sm86_readiness` | SM86 readiness projections | report-only |
| `sim/direct_run::metrics::pipeline_ipc::pipeline_sm86_memory` | SM86 memory projections | report-only |
| `sim/direct_run::metrics::pipeline_ipc::pipeline_sm86_tail` | SM86 tail/completion projections | report-only |
| `tools/sm120_corpus_replay.py` | Twelve-worker full-corpus driver | orchestration only |
| `tools/validate_reuse_mrc.py` | MRC structural/error report | post-replay analysis |
| `arch_profiles/cards` | SM86/SM120 architecture parameters | immutable configuration |

## Public interfaces

Parent facades retain the existing crate paths and serde field names. Child
files are genuine Rust `mod` submodules with explicit re-exports. Split `impl`
blocks use `pub(super)` or `pub(crate)` only for sibling access; public items
remain public. Every active mutation is
still performed by the original owner (`SmCore`, `L1Cache`, `L2Model`, or
`DramServer`); observer modules cannot write those structures.

## Validation surfaces

* Full replay: integrated `l1-l2`, card-selected architecture, twelve workers,
  compressed JSONL.
* Hardware comparison: independent NCU profiles joined after replay.
* Locality comparison: exact Fenwick MRC versus HOTL estimator on the same
  within-warp stream.
* Determinism: compare status, identity, metadata, and common numeric fields;
  ignore only wall-clock process timings.
