# Components and invariants

| Crate / module | Concrete responsibility | State boundary |
|---|---|---|
| `sim/trace::reader` | Protobuf/JSON trace loading, lineage, compact memory records | immutable input; fail-closed validation |
| `sim/trace::reader::snapshot` | Artifact identity and JSON-shape limits | report-only |
| `sim/trace::reader::compact` | Validated warp/instruction/memory representations | report-only |
| `sim/arch` | Architecture cards, scheduler resources, functional instruction domains | card-selected assumptions |
| `sim/sm::issue` | SMSP issue, CGGTY selection, barriers, readiness, async copies | mutates issue/residency timing |
| `sim/sm::issue::frontend` | I-cache, stream buffer, front-end admission | mutates instruction timing |
| `sim/sm::issue::pipelines` | Scalar/memory pipeline readiness and fixed latencies | mutates issue timing |
| `sim/sm::issue::core_mem` | Memory dispatch and observer hooks | mutates request/cache state |
| `sim/sm::replay` | Launch placement, replay sessions, result records | mutates replay session state |
| `sim/sm::system` | Interleaved SM/L2/DRAM event loop | mutates event and service state |
| `sim/sm::system::queues` | Pending responses and due-time queues | mutates event queue state |
| `sim/sm::system::dram` | Carded DRAM bandwidth/service accounting | mutates DRAM events |
| `sim/sm::system::observers` | L2 source and admission observer records | report-only accounting |
| `sim/sm::metrics::locality_mrc` | Fenwick MRC, HOTL footprint inversion, Tier-2/Tier-3 shadows | report-only |
| `sim/sm::metrics::pipeline_activity` | Warp lifetime and pipeline activity counters | report-only |
| `sim/mem::l1` | Sectored L1 tags, fills, merges, write lookup | mutates L1 state |
| `sim/mem::l2` | Shared L2 tags, sector fills, write packets | mutates L2 state |
| `sim/mem::l1_visibility` | Payload/visibility handoff | event-boundary contract |
| `sim/semantics::state` | Typed cache/core/event boundary snapshots | report-only |
| `sim/engine` | Dependency/resource event ordering | event ordering only |
| `sim/direct_run::emit` | Stable JSON and counter emission | report-only serialization |
| `sim/direct_run::metrics::l1` | L1/TEX counter attribution | report-only |
| `sim/direct_run::metrics::l2_dram` | L2/DRAM and readiness attribution | report-only |
| `sim/direct_run::metrics::pipeline_ipc` | IPC and pipeline projections | report-only |
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
