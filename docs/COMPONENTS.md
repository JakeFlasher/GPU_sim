# Component inventory

This inventory describes the cleanup boundary as of 2026-07-29. `cargo
metadata` is authoritative for Rust workspace membership; this document is a
reviewable snapshot until a tracked generator is added.

Status vocabulary:

- **active**: required for the simulator or silicon-validation loop;
- **fixture**: retained as a reproducible oracle or historical contract;
- **optional**: evaluation breadth, not required by the default gate;
- **frozen**: evidence retained byte-exact and not modified by cleanup;
- **retire**: obsolete or generated material scheduled for removal.

## Rust workspace

| Package / path | Status | Responsibility | Known gap |
|---|---|---|---|
| `sim-engine` / `sim/engine` | active | Deterministic `(time, seq)` event queue | Timing-wheel/batching optimization not implemented |
| `sim-trace` / `sim/trace` | active | Prost trace decode, lane reconstruction, completeness and identity checks | Machine-local smoke policy still being refit |
| `sim-sm` / `sim/sm` | active | Control-bit join, four-SMSP issue, traced placement, admitted launch replay, and checked full-launch aggregation | Per-launch issue/terminal timing fields remain diagnostic |
| `sim-mem` / `sim/mem` | active | Sectored L1/L2 and DRAM bandwidth model | Several microarchitecture choices remain explicit hypotheses |
| `sim-emit` / `sim/emit` | active | Seven typed NCU-shaped secondary predictions plus the mandatory blocked cycles row | Binding-cycle implementation awaits admitted aggregation semantics |
| `sim-arch` / `sim/arch` | active | Typed architecture identities/capabilities and fail-closed architecture-card loading | Production runner supports `sm_86`; `sm_120` gaps remain closed |
| `sim-runner` / `sim/runner` | active | Strict run-manifest admission, card-derived configuration, replay, checked aggregation, and canonical sealed results | No production launch-manifest producer or admitted cycle semantics |
| `sim-cli` / `sim/cli` | active | One-manifest command-line boundary with canonical result/error streams | Offline execution confers no hardware readiness |
| `sim-ffi` / `sim/ffi` | retire | Three-line unused placeholder | Already excluded from Cargo workspace; leftover path is HEAD-only cleanup, not active architecture |
| `sim/trace/pbgen` | build helper | One-shot Prost binding generation | Not a workspace member; generated target is disposable |

The workspace now has a production simulator binary. No performance benchmark
crate or production evaluator is present yet.

## Core capture and modeling tools

| Path | Status | Responsibility |
|---|---|---|
| `tools/nvbit_tracer/tracer_tool/` | active | NVBit SASS/address trace producer and host-side trace printer |
| `tools/nvbit_tracer/traces_enhanced/` | active | Protobuf schema, generated C++ bindings, and native wire library |
| `tools/nvbit_tracer/metric_extractor/` | active | Required NVBit instrumentation and Tier-2 integrity source/tests |
| `tools/nvbit_tracer/nvbit_release/core/` | active vendored | NVIDIA NVBit SDK, including byte-exact `libnvbit.a` |
| `tools/local_labels/` | active | NCU capture, admission, provenance, campaign, registry, and evidence gates |
| `tools/sass_control/` | active | Static SASS control-bit extraction and validation |
| `tools/trace_sampling/` | active | Typed trace-completeness and sampling contracts |
| `tools/cache_predictor/` | active reference | L1 cache-policy/classification parity source |
| `tools/cache_replay/` | active reference | Sectored L2 replay parity source |
| `tools/pchase/`, `data/pchase/` | active | Microbenchmark collection and committed memory cards |
| `arch_profiles/` | active | Per-architecture geometry, latency, and typed research gaps |

The source and tests under `metric_extractor/` remain in-tree.
`metric_extractor.so` is a grandfathered, SHA-bound H3 artifact and is not a
cleanup target. Its six rebuildable binaries/objects are cleanup targets:
`metric_extractor.o`, `inject_funcs.o`, `test_classifier`,
`test_saturating_counter`, `test_tier2_marker_metadata`, and
`test_tier2_sha256`.

## Workloads and evaluation bundles

| Path / class | Status | Policy |
|---|---|---|
| `workloads/simbench_bw/` | retained core workload/evidence source | Keep as trace, capture, and evaluation input. Its NCU labels are scoring targets only and never derive simulator, architecture-card, or calibration constants; historical CUPTI contracts are fixtures only |
| `workloads/portable_lcd/` | retained core evidence source | Capture and machine-local replay source; its CUTLASS 3.8 subset remains byte-exact |
| `tools/sol_execbench/`, SOL workload corpus | optional | Separate environment and explicit test invocation; not in default pytest collection |
| `tools/solar/` | optional | Evaluation baseline, not simulator input |
| FlashInfer-derived data/tests | optional | Heavy bundle; requires its own dependencies and environment |
| `third_party/cutlass/` | retained active vendor | Official CUTLASS v4.6.1, scoped to `examples/`, `include/`, and `tools/`; provenance and digest are pinned in `third_party/cutlass.provenance.json`. Do not merge with the workload's byte-exact 3.8 subset or relabel frozen cuBLAS evidence |

Unclear workload provenance resolves to retain-and-flag, never delete by
inference.

## Evidence and data

| Path | Status | Policy |
|---|---|---|
| `evidence/round28/` | fixture/frozen | Migrated Round-28 audits, adjudications, reconstruction, session excerpt, and retained consultation/decision provenance |
| `evidence/MANIFEST.sha256` | active authority | Hash authority for tracked migrated fixtures |
| `evidence/path_aliases.v1.json` | blocked historical authority | Its rows cite a missing ADR whose number is now occupied; retain bytes, but do not claim authority closure until an accepted versioned repair exists |
| `evidence/path_aliases.v2.json` | active authority | Versioned M1 read-time relocation with mandatory source-manifest and byte equality |
| `data/local_labels/campaign/` | frozen | Canonical 344-cell authority and historical runnability v1; undeletable in this cleanup pass |
| `data/annex/` | frozen | Possibly unique; undeletable in this cleanup pass |
| External Task7/H3 capture-defect archive | frozen external | Zero campaign credit, non-authoritative, and reuse-forbidden under ADR 0001 |
| External stopped 49-row campaign revision | frozen external | Zero campaign credit, non-authoritative, and reuse-forbidden |
| External interrupted 53/201 SOL runnability archive | frozen external | Separate non-authoritative preservation; canonical v1 is restored |
| Supplemental launch-set v3 | active offline contract | Separate cohort; live-disabled until a pinned canary and never canonical campaign credit |
| `/home/jakeshea/sm86_reference_capture_r27/` | frozen external | Parent campaign/preparation-failure and non-authoritative attempt archive; never mutate in place |
| Original repository `.humanize/` copy | frozen external | Read-only historical authority; never chmod or mutate |
| Local `.humanize/` | retire after gates | Delete only after all consumers migrate, fresh-worktree tests pass, accepted ADRs are reflected, and archive identity is rechecked |
| `tools/sol_execbench/data/` | optional, retained | Possibly unique until content-addressed replication is hash-verified |
| `references/` cited subset | active | Keep items directly used by architecture/tooling documentation |
| `references/` bulk | externalize | Manifest source, retrieval date, license, size, and SHA-256 before removal from HEAD |

Git history is not rewritten, so tracked HEAD deletions remain recoverable from
the existing commit graph. That does not authorize deletion of ignored or
external evidence.
