---
Use when: reading parent-repo nvasim context as PRIOR-ART reference for the ACTIVE simulator build (re-founded 2026-07-02) or understanding why simulator-related reference cards exist.
Primary raw source: simulator/README.md
Supporting sources:
- simulator/Cargo.toml (workspace structure)
- simulator/src/main.rs (entrypoint, CLI, config loading)
- simulator/src/simulator.rs (SM mini-loop, memory hierarchy wiring)
- simulator/src/sim_config.rs (TOML config schema, env-var overrides)
Repo touchpoints:
- simulator/configs/rtx_3090_ti.toml (hardware config)
- simulator/crates/event-engine/ (priority-queue event dispatch)
- simulator/crates/trace-parser/ (protobuf trace ingestion + SASS opcode table)
- simulator/crates/sm-model/ (SubCore, scheduler, register file)
- simulator/crates/memory/ (L1, L2, shared memory, DRAM, interconnect)
Out of scope: porting nvasim source verbatim (design from cards instead; this card is PRIOR-ART reference for the ACTIVE simulator build, re-founded 2026-07-02), NCU capture procedure (see measurement/pinned-protocol-v21.md), CUDA/PTX instruction semantics (see ptx/ cards), and NVBit tracer internals (see tools/nvbit_tracer/README.md).
Version family: Historical nvasim SM120 / Blackwell snapshot from the parent project, 2026-04-16.
---

# nvasim Architecture Overview

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

Historically, nvasim was an event-driven, single-SM GPU architectural simulator
targeting SM120. It read NVBit protobuf traces and modeled SM timing with
SubCore scheduling, shared execution units, a banked register file, and a memory
hierarchy. That implementation is no longer an active objective in this fork.

## Workspace Structure

Four Rust crates in a cargo workspace (`simulator/Cargo.toml`, resolver v2,
edition 2024):

| Crate | Path | Role |
|-------|------|------|
| **event-engine** | `crates/event-engine/` | Binary-heap priority queue, tagged enum dispatch (`SimEvent`), next-event clock advancement, SHA-256 determinism proof. Benchmarked at 258M events/sec. |
| **trace-parser** | `crates/trace-parser/` | Prost protobuf trace reader, `enhanced_execution_info.json` parser, 166+ Blackwell SASS opcodes with branch target extraction, control bit parser, kernel exclusion filter, trace audit. |
| **sm-model** | `crates/sm-model/` | 4 SubCores per SM, GTO and CGTY warp schedulers, shared execution unit pipelines (FP32/INT32/FP64/SFU/TC/LS/BR/Uniform), banked register file with RF cache hypothesis. |
| **memory** | `crates/memory/` | Sectored/banked L1 with MSHRs, shared memory bank conflict model, memory coalescing, flow-controlled interconnect channels, partitioned L2, DRAM controller with row buffer timing. |

The top-level `nvasim` binary (`simulator/src/main.rs`) wires the four
crates together via `simulator.rs` (the SM mini-loop).

## Entrypoint Flow

1. **CLI** (clap derive): `--config`, `--trace`, `--determinism-check`, `--trace-audit`, `--report-json`, `--max-cycles`, `-v`.
2. **Config loading**: TOML deserialization → `NVASIM_*` env-var overrides → `config.validate`.
3. **Trace parsing**: `TraceReader::read_trace_dir` → version validation (must be 120) → kernel exclusion filter (contested atomics, spin-loop detection via branch targets).
4. **Simulation**: `run_simulation` → event-engine loop → SM mini-loop per cycle → JSON report.
5. **Determinism proof**: optional SHA-256 hash of the full event log for replay verification.

## SM Timing Model

The SM mini-loop in `simulator.rs` drives the core simulation:

- **4 SubCores** per SM, each with an independent `WarpScheduler` (GTO or CGTY policy).
- **Shared execution units**: Pipeline units (FP32, INT32, FP64, SFU, TC, LS, BR, Uniform) are shared among all warps in a SubCore, modeling real resource contention.
- **Register file**: Banked with per-cycle bank access tracking. 8-entry LRU RF cache hypothesis bypasses bank access for recently-used registers.
- **Control-bit-driven**: 6-entry scoreboard barriers, stall counters, yield flags extracted from SASS control bits.
- **CTA-at-a-time**: SubCore warp slots populated per-CTA; completed CTAs release SubCore slots for new ones.

## Memory Hierarchy

| Component | Key parameters |
|-----------|---------------|
| **L1 cache** | Sectored (32B sectors), banked, configurable size, MSHRs for miss handling. Supports global, local, and constant loads. |
| **Shared memory** | Banked (32 banks on SM120), bank conflict detection, configurable size. |
| **L2 cache** | Partitioned, configurable associativity, handles L1 misses. |
| **DRAM** | Row buffer timing (hit/miss/precharge latencies), channel interleaving. |
| **Interconnect** | Abstract flow-controlled channel with configurable bandwidth and latency. |

## Configuration System

Hardware parameters live in TOML files (e.g., `simulator/configs/rtx_3090_ti.toml`):

- `[gpu]` — SM count, clock frequencies, compute capability
- `[sm]` — SubCore count, warps/SM, registers, scheduler policy
- `[sm.hypothesis]` — Tunable hypotheses (RF cache, uniform unit, L0 icache)
- `[memory.l1]` — L1 size, sectors, banks, MSHRs
- `[memory.shared_memory]` — shared memory size, banks
- `[memory.l2]` — L2 size, partitions, associativity
- `[memory.dram]` — channels, row hit/miss/precharge timing
- `[memory.interconnect]` — bandwidth, latency

A small allow-list of fields supports runtime `NVASIM_*` env-var overrides
(validated before simulation). The known overrides are:

| Env var | Target field |
|---------|-------------|
| `NVASIM_SCHEDULER_POLICY` | `sm.scheduler_policy` (Gto or Cgty) |
| `NVASIM_IBUFFER_ENTRIES` | `sm.instruction_buffer_entries` |
| `NVASIM_REGISTER_FILE_BANKS` | `sm.register_file_banks` |
| `NVASIM_KERNEL_REGISTERS_JSON` | Per-kernel register count sidecar (consumed in main.rs, not sim_config) |
| `NVASIM_KERNEL_REGISTERS_ALLOW_NO_MATCH` | Downgrade no-match bail to warning (consumed in main.rs) |
| `NVASIM_DUMP_KERNEL_ACCESS_PATTERN` | Calibration-doc authoring helper; dumps per-kernel access pattern for offline analysis (consumed in the simulation loop). |

Any `NVASIM_*` env var not in this allow-list is a hard error, preventing
silent typo-driven no-ops.

## Trace Format

> **Active-reader note:** the ACTIVE trace consumption lives in `sim/trace/src/reader.rs` + `lanes.rs` (admission owned by `tools/nvbit_tracer/import_trace_contract.py`); the version-120 gate and mandatory enhanced_execution_info.json below were nvasim-only rules — do NOT add them to the active reader. The ACTIVE sim now ALSO consumes enhanced_execution_info.json via `sim/trace/src/exec_info.rs`, but ONLY as the function-id→name/pc join surface; its embedded per-instruction control_bits are garbage (88/5,736 agreement with the bit-validated sidecar) and are never parsed — the Phase-1E control-bits sidecar is the sole control source.

NVBit protobuf traces in the MICRO 2025 enhanced format:

```
traces/
  dynamic_trace.pb                          # Kernel metadata
  extra_info/enhanced_execution_info.json   # Static ISA info (mandatory)
  threadblocks/device_0/stream_0/
    kernel_1/d_0_s_0_k_1_0,0,0.pb          # Per-threadblock traces
```

Binary version must be 120 (SM120/Blackwell). The trace parser supports 166+
Blackwell SASS opcodes including cluster barriers (`UCGABAR_ARV`,
`UCGABAR_WAIT`) and DSMEM load/store aliases. The `MBARRIER_*` family
(arrive, wait, try_wait, init, arrive_drop, arrive_expect_tx) remains
unsupported — these are deferred behind async-copy / TMA modeling.
`CLUSTERLAUNCH` and `CLUSTERLAUNCHCONTROL` are also unsupported. Unknown
opcodes not in the opcode table are rejected with diagnostics.

## JSON Report

When `--report-json` was set, nvasim emitted per-kernel metrics that the
historical parent project compared against profiler output. Key field
categories from that historical contract are preserved here only to explain old
card references:

- **Core**: `cycles`, `instructions`, `events`, `ipc`
- **L1**: `l1_hits`, `l1_misses`, `l1_mshr_full`, `l1_hit_rate_unified`, `l1_global_load_hit_rate`
- **L2**: `l2_hits`, `l2_misses`, `l2_read_transactions`, `l2_write_transactions`, `l2_total_transactions`
- **DRAM**: `dram_accesses`, `dram_total_transactions`
- **Op-class counters**: `op_class_global_load_requests`, `op_class_global_store_sectors`, etc. (15 per-op-class fields)
- **Occupancy**: `block_limit_registers`, `block_limit_shared_memory`, `block_limit_warps`, `block_limit_sm`, `theoretical_occupancy`
- **Tensor**: `tensor_inst_executed`, `tensor_inst_hmma`, `tensor_inst_qmma`, `tensor_pipe_active_cycles`
- **Provenance**: `kernel_registers_json_sha256`, `git_commit`, `version`

The report is currently constructed via `format!` string building (see
`main.rs:860-907`). Future target: `#[derive(Serialize)]` struct.

## Determinism Guarantee

Every event dispatched by the engine is hashed (SHA-256). The `--determinism-check` flag prints the cumulative hash at exit. Identical inputs produce identical hashes across runs — this is the replay verification mechanism.

## Modeling context (papers)

- [Need for Speed (NVArchSim)](references/papers/knowledge_base/active/2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim.pdf) — nvasim is architecturally inspired by NVArchSim; the validation discipline and event-driven design descend from this work.
- [Accel-Sim](references/papers/knowledge_base/active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf) — the SASS-trace-driven simulation paradigm that nvasim's trace format follows.

## See also

- `.claude/knowledge/simulator/nvasim-dev-patterns.md` — historical Rust conventions in nvasim.
- `.claude/knowledge/measurement/nvarchsim-vs-this-project.md` — why simulator material is now prior art.

## Escalate to raw when

- You need an exact historical detail from the parent simulator snapshot.
- You are writing prior-art context and need to distinguish simulator validation from this fork's non-root measurement goal.
