---
name: designing-gpu-simulator
description: ACTIVE design skill for the event-driven, trace-driven GPU simulator (sim/, Rust workspace) that predicts NCU metric vectors on Ampere sm_86 + Blackwell sm_120. Use when designing or implementing the sim engine, SM/warp-scheduler model, memory hierarchy (L1 sectored+MSHR, L2 slices, DRAM), NCU-emission layer, calibration layers, or when consulting simulator prior art (NVArchSim, Accel-Sim, GPGPU-Sim, UPC MICRO'25, MGSim, ClusterSim) and Rust DES patterns. Building the simulator IS the project (CLAUDE.md mission, re-founded 2026-07-02); the old "no simulator revival" rule is retired.
---

# Designing the Event-Driven GPU Simulator (ACTIVE)

The repo's mission (CLAUDE.md, re-founded 2026-07-02) is to build `sim/` — a fast,
event-driven, trace-driven Rust simulator of Ampere sm_86 + Blackwell sm_120 that
reproduces NCU metric vectors at 100–1000× cycle-level-simulator speed, validated
per-metric against silicon NCU labels. This skill routes its design sources.

**Founding design note:** `references/papers/notes/draft_new_gpu_simulator.md`
(NVArchSim-inspired: event engine, per-SM actor mini-loops, fidelity knobs,
phases A–E). Execution plan: `plans/micro2027_ncu_sim_execution_plan.md` §6.
Companion audit: `plans/deep_research_audit_and_extraction_plan_2026-07-06.md`.

## Current implementation (Phase 2, 2026-07-06)

- `sim/trace` — `reader.rs` trace walk + wire re-checks + launch-order block API;
  `lanes.rs` lockstep decoder; `exec_info.rs` static-metadata join surface
  (function id → name/pc table ONLY — its embedded control bits are garbage,
  88/5,736 sidecar agreement, never parsed); committed prost bindings +
  `regen_pb.sh`.
- `sim/sm` — `control_bits.rs` fail-closed sidecar loader; `program.rs`
  (function, pc) join (pc-set equality both directions); `issue.rs` v0
  4-sub-core CGGTY issue model driven by control bits, now the
  RESUMABLE `SmCore` actor (advance-to-horizon + on_response; barrier
  PENDING counts for in-flight responses); `system.rs` kernel-level
  event system (SmWake/L2Access/MemResponse over the engine queue —
  the §6.3 request/response inversion; per-SM response-due horizon,
  never other SMs' events; per-source L2 split lives in the DRIVER
  observer `L2SourceSplit`, not the cache model); `replay.rs`
  traced-placement replay (`sm_id`/`warpid_sm`; ONE typed entry point
  `replay_kernel(.., ReplayMemMode::{Flat, L1Const, L1SharedL2})`).
  `SmCore` carries a typed `MemModel` (shared-mode state inside the
  variant); the L1 fill-timing discipline is latched per instance.
  Causality contract (gate-3 hardened): horizon = min(own scheduled
  responses, per-SM UNPRICED request bounds) — a request is
  self-affecting from EMISSION until L2 pricing; equivalence-proven
  vs the constant path (event-vs-const differential 1000/1000 exact
  in eviction-free regimes). DRAM: carded aggregate byte server
  (`DramBw` — vendor 1008 GB/s at the locked point, ~747 B/SM-cycle
  from `ArchCard.dram_bandwidth`; G/D/1 fixed-point, read-miss
  sectors only, `max(latency arm, server completion)`; sweep finding:
  ONLY ac4_dram engages it — the broad cycles undershoot is upstream,
  per-SM port/MSHR/async-copy structure). Result:
  `smsp__inst_executed.sum` EXACT 15/15 vs the frozen v2.1 labels;
  cycles ratio table in the progress log (geomean 0.358).
- `sim/mem` — `l1.rs` sectored L1 core (lockstep port of the Phase-1
  warp-pool walker's cache state machine + event-completion mode for
  the shared L2); `l2.rs` shared L2 (lockstep port of
  `sector_cache_replay.py`: LRU/SRRIP/BIP, 6 slice hashes with
  Python-floored NEGATIVE-id semantics, 72/72 counter-exact
  differential + committed parity pair
  `tools/cache_replay/test_l2_rust_parity.py` ↔
  `python_parity_corpus_pins`; geometry vendor-carded; assoc/hash/
  policy documented defaults gated on `L2_DEFAULTS_REVIEWED_ARCHES`
  and the card's write policy — the Phase-1 LABEL-FITTED calibration
  is never consumed); `access.rs` shared MemRole/AccessKind
  classification (token-preserving); `cache_op.rs` token rules from
  sidecar opcode text (the Python classifier's OWN manifests are
  cross-arch-contaminated — never consume them); `pchase_card.rs`
  fail-closed geometry resolvers. Result: L1 eligible-load hit rate
  EXACT (MAE 0.0003 pp) vs NCU op_ld on 10/10 scorable cells. L2 §6.4:
  per-access-class hit rule (`L2OpClass{Read, WriteAtomic}`: loads hit
  on sector validity — the ported Python rule; stores/atomics/
  reductions on line-TAG residency), COUNTING split from PRICING
  (`L2Outcome`: responses price on physical `sector_present`, worst
  sector of a multi-sector fill); `MemRequest.sector_mask` makes the
  L2 see a SECTOR stream (batch merges fold into the fill's mask,
  late merges send one follow-up fetch) — fill/store sector counts
  now EXACTLY match NCU's op_ld/op_st on the witness cell. L2 unified
  MAE 45 pp → 13.96 pp with accounting contract-exact (the interim
  8.96 rested on the fill undercount compensating). Load-bearing
  residuals: L2 FETCH GRANULARITY on streaming cells (modeled 37.5 /
  64 B-pair 62.5 / measured ~55 — close by microbench, arch-gated);
  RED/ATOM addresses excluded at capture (histogram L2 label
  unscorable until tracer extension + re-capture).
- `sim/calib` — ArchProfile loader; research_gap = typed error; block-level
  Carded gaps.
- `sim/engine` — (time,seq) deterministic event queue, live under
  `sim/sm/src/system.rs` for the shared-L2 co-simulation.
- Progress log: `plans/phase2_sim_core_progress.md`.

## Cards (nvasim background + Rust DES)

| Card | Topic |
|---|---|
| `.claude/knowledge/simulator/nvasim-architecture.md` | 4-crate Rust workspace; event-driven SM timing; memory hierarchy; config; trace format |
| `.claude/knowledge/simulator/nvasim-dev-patterns.md` | Error handling, serde/TOML config, CLI, testing, dependency management |
| `.claude/knowledge/simulator/rust-type-safety.md` | Newtypes, Option vs magic constants, checked arithmetic, visibility, 5 audit gaps |
| `.claude/knowledge/simulator/rust-testing-quality.md` | proptest/insta/cargo-fuzz/Loom/MIRI as future targets, CI design |
| `.claude/knowledge/simulator/rust-des-patterns.md` | Priority queue determinism, tagged enum dispatch, EventHandler trait |
| `.claude/knowledge/simulator/rust-performance.md` | Profiling workflow, allocation discipline, memory layout, iterator optimization |

## Rust reference material

- `references/Rust_refs/README.md` — navigator for NeXosim DES framework, Effective Rust, Async Rust, Programming Rust 3rd ed.

## Design ground rules (from the execution plan §6 + F1–F10 fix map)

- Engine: binary-heap event queue → timing-wheel hybrid later; tagged-enum dispatch
  (no virtual dispatch on the hot path); per-SM actor mini-loop batching several
  issue cycles per wake — never one event per warp pick. Target ≥3–10 M events/s.
- SM: 4 sub-cores, CGGTY issue; dependence stalls READ from SASS control bits
  (`tools/sass_control/` sidecar), not simulated scoreboards (layout contract
  lockstep: extractor.py PARSER_CONTRACT == sim/sm EXPECTED_LAYOUT — parity test
  tools/sass_control/test_control_layout_rust_parity.py; cache-op vocabulary
  lockstep: tools/cache_predictor/test_cache_op_rust_parity.py). Unified INT32/FP32
  pipes on sm_120 (per-arch card field), split 64+64 on sm_86.
- Memory: L1 sectored + finite MSHR + carveout-aware capacity (F1–F3), with the
  access interleaving supplied by the SM issue model (the Phase-1 warp-pool FIFO
  scheduler was a stand-in; only its cache core is ported); L2 slice actors with
  `tools/cache_replay/` semantics (32 B
  sector / 128 B line; widen the slice hash for GB202's **128 MB** L2); DRAM
  GDDR6X/GDDR7 bandwidth-latency.
- Emission (`sim/emit`): every metric implements its
  `tools/local_labels/metric_registry.json` v2 `ncu_observation_contract`,
  including NCU sector-accounting semantics and cold-cache-per-kernel defaults
  (`--cache-control all` parity).
- Targets-not-features (CLAUDE.md Rule 1): NCU values never enter sim inputs or
  per-cell-fitted calibration constants. Per-arch constants come from
  microbenchmarks (`tools/pchase/` et al.) or vendor docs, cross-checked only.
- Undocumented microarchitecture (RF cache, uniform unit, L0 stream buffer) =
  tunable hypotheses behind fidelity knobs, never baseline spec.
- sm_120 tensor path = extended warp-synchronous `mma.sync` (HMMA/QMMA/OMMA) —
  **no tcgen05/TMEM** (those are sm_100-only; out of scope for paper 1).

## Cluster modeling

- `.claude/knowledge/cuda/cuda-cluster-modeling.md` — modeling contract: DSMEM latency, barrier state machine, traffic patterns
- `workloads/simbench_bw/l2/cluster_remote_reduction/cluster_remote_reduction.cu` — cluster microbench in this repo
- Paper: `references/papers/knowledge_base/active/2025__clustersim_modeling_thread_block_clusters_in_hopper_gpus.pdf` (first dedicated cluster execution model)

## Prior art routing

- `references/papers/knowledge_base/problem_playbook.md` + `selected_index.md` +
  `active/*.pdf` — simulator/model literature (Accel-Sim ISCA'20, NVArchSim
  HPCA **2021** Villa et al., UPC MICRO'25 modern-core rewrite, GCoM, MDM,
  Nugteren HPCA'14, ClusterSim).
- `references/papers/notes/gpu-simulator-non-privileged-research.md` —
  simulator-validation contract methodology (salvaged from wall_clock_sims).
- Per-generation must-model objects: `.claude/knowledge/patents/problem-playbook.md`
  (Blackwell TMEM/tcgen05 rows apply to sm_100 only — not the sm_120 targets).

## Cross-references

- Hardware semantics → `interpreting-ptx-isa` (PTX), `analyzing-sass-nvbit-traces` (SASS).
- Why each modeled object exists → `reasoning-about-nvidia-architecture-history`.
- Blackwell resource ceilings → `tuning-blackwell-sm120-kernels`.
- Measurement/label protocol → root `CLAUDE.md`, the active local-label schemas,
  `tools/local_labels/`, and the owner runbook (the historical
  `validating-nonroot-gpu-measurement` skill is retired).

## Cite consulted files

End every reply with `## Knowledge Consulted`.
