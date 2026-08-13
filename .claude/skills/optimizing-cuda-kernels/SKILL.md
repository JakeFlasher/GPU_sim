---
name: optimizing-cuda-kernels
description: Generic CUDA kernel optimization — coalescing, occupancy/launch config, register-file budgeting, warp scheduling/divergence, shared-memory bank conflicts, async-copy patterns, sync/memory-ordering primitives. Use when prompts mention coalescing, occupancy, __launch_bounds__, register pressure/spilling, bank conflicts, warp divergence, ITS, warp scoreboard, GTO/CGTY scheduling, L1/L2 cache, async copy (cp.async/TMA), multi-stream concurrency, cluster/DSMEM ergonomics, memory-consistency model, fence/release/acquire scopes, or "make my CUDA kernel faster" non-Blackwell-specific. Hand off Blackwell tuning numbers to `tuning-blackwell-sm120-kernels`.
---

# Optimizing CUDA Kernels

Owns the cross-architecture CUDA performance surface. 7 cuda/ cards + 3 best-practices/ cards.

## Cards

### Performance (best-practices/)

| Topic | Card |
|---|---|
| Global-mem coalescing, bank conflicts, L1/L2, shared-mem, pinned memory, async copy / TMA | `.claude/knowledge/best-practices/perf-coalescing-memory.md` |
| Occupancy, `__launch_bounds__`, register-file budgeting, block-size heuristics, multi-stream | `.claude/knowledge/best-practices/perf-occupancy-launch.md` |
| Arithmetic throughputs, warp divergence, predication, sync semantics, math-lib tradeoffs | `.claude/knowledge/best-practices/perf-instruction-controlflow.md` |

### Microarchitecture (cuda/)

| Topic | Card |
|---|---|
| Memory hierarchy (L1/L2/DRAM/interconnect, sector sizes, MSHRs, generation deltas) | `.claude/knowledge/cuda/cuda-memory-hierarchy.md` |
| Warp scheduling (GTO/CGTY, scoreboard, stall taxonomy, SubCore) | `.claude/knowledge/cuda/cuda-warp-scheduling.md` |
| Warp divergence (SIMT stack vs convergence barriers, ITS) | `.claude/knowledge/cuda/cuda-warp-divergence.md` |
| Register file (banked organization, port conflicts, RF cache hypothesis) | `.claude/knowledge/cuda/cuda-register-file.md` |
| Sync & memory ordering (PTX consistency model, scope hierarchy, CUDA C++ → PTX mapping) | `.claude/knowledge/cuda/cuda-sync-memory-ordering.md` |
| Cluster programming model, DSMEM, `cudaLaunchKernelEx`, cooperative-groups cluster APIs | `.claude/knowledge/cuda/cuda-clusters-dsmem.md` |
| Cluster measurement features and prior-art routing | `.claude/knowledge/cuda/cuda-cluster-modeling.md` |

## Decision tree

1. **Memory-bound?** → coalescing → `perf-coalescing-memory.md` → memory-hierarchy backing → `cuda-memory-hierarchy.md`.
2. **Compute-bound?** → occupancy + register pressure → `perf-occupancy-launch.md` → register-file → `cuda-register-file.md`.
3. **Latency-bound?** → warp divergence → `cuda-warp-divergence.md` → scheduling → `cuda-warp-scheduling.md`.
4. **Synchronization issue?** → `cuda-sync-memory-ordering.md` (PTX-side syntax in `interpreting-ptx-isa` skill).
5. **Multi-block cooperation?** → cluster + DSMEM → `cuda-clusters-dsmem.md` → modeling → `cuda-cluster-modeling.md`.
6. **Anything Blackwell-specific?** → hand off to `tuning-blackwell-sm120-kernels`.

## Quick rules

- **Coalescing**: 32-byte sector size (Volta+); a warp's 32 loads should land in 1 sector if perfectly aligned. Misalignment → 2 sectors.
- **Bank conflicts**: 32-bank shared memory; thread-i accesses bank `(i % 32)`. Conflicting access patterns serialize.
- **Occupancy ceiling per-arch (current targets)**:
  - SM120 (RTX PRO 6000): 48 warps/SM ← **the cloud target**
  - SM100 (B200): 64 warps/SM
  - sm_86 (3090 Ti, local smoke-test only): 48 warps/SM
- **Register pressure**: first lever is `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` or `-maxrregcount=N`.
- **Async copy idiom**: `cp.async` (Ampere+) → `mbarrier` wait; `cp.async.bulk.tensor` (Hopper+, TMA) → transaction barrier (`mbarrier.complete_tx`); `tcgen05.cp` (Blackwell) → TMEM-resident.
- **Cluster.sync REQUIRED before any DSMEM peer access** of another block's shared memory.
- **Compute capability matters**:
  - cp.async = sm_80+; structured 2:4 sparsity = sm_80+; BF16/TF32 = sm_80+
  - FP8 = sm_89/sm_90+; cluster + DSMEM + TMA + WGMMA + transaction barrier = sm_90+
  - tcgen05/TMEM/FP4/MX = sm_100/sm_120+

## Generation-spanning quirks

- **ITS (sm_70+)**: `_sync` mask-suffixed warp intrinsics REQUIRED. Pre-Volta intrinsics without mask deadlock under cooperative scheduling.
- **CUDA core split (sm_70+)**: FP32 + INT32 cores execute concurrently; integer arithmetic is "free" against FP32 throughput in mixed code.
- **L1/SMEM unification (sm_70+)**: configurable carveout up to 256 KB combined per SM (varies by gen).
- **Hopper introduces clusters** but blocks still don't move between GPCs mid-execution.
- **Blackwell replaces WGMMA encoding** with `tcgen05.mma` — see `tuning-blackwell-sm120-kernels`.

## Cross-references

- PTX syntax for any sync/atomic/load/store → `interpreting-ptx-isa`.
- SASS-level performance evidence → `analyzing-sass-nvbit-traces` (instruction-mix from NVBit traces).
- Blackwell-specific tuning numbers → `tuning-blackwell-sm120-kernels`.
- Why a feature exists at all (patent reasoning) → `reasoning-about-nvidia-architecture-history`.
- Measurement/label protocol → root `CLAUDE.md`, active label schemas,
  `tools/local_labels/`, and the owner runbook; the historical
  `validating-nonroot-gpu-measurement` skill is not an operational authority.

## Scientific boundary

For scored cells, NCU values are prediction targets/labels, never simulator
features. Follow the current root `CLAUDE.md`, active schemas, and owner runbook
for capture and replay state; the historical v2.1 settings are relevant only
when reproducing that dataset.
