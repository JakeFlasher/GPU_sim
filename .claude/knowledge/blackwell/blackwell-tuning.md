---
Use when: choosing block / cluster dims, register budget, or shared-mem carveout for an SM120 (cloud RTX PRO 6000 Blackwell, GB202) or SM100 (B200) kernel; deciding whether an optimization carries over from Hopper or must be re-tuned under the non-root measurement contract.
Primary raw source: references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md.
Supporting sources: references/NVIDIA_Manuals/best_practices/09_performance-metrics.md, references/NVIDIA_Manuals/cuda_guide/02-02-07_kernel-launch-and-occupancy.md.
Repo touchpoints: workloads/simbench_bw/, tools/cloud/preflight.sh, tools/nvbit_tracer/, CLAUDE.md.
Out of scope: NCU workflows, hardware-counter tuning, Blackwell PTX instruction semantics (in `ptx-isa-sm100-blackwell.md`); Blackwell forward-compat / JIT rules (in `blackwell-compatibility.md`); generic CUDA tuning not specific to Blackwell.
Version family: CUDA 13.x; Blackwell compute capability 10.0 (B200) and 12.0 (SM120 consumer). Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# Blackwell Tuning — SM120 / SM100 Cheat Sheet

Compact view of the Blackwell tuning guide's hard numbers plus the
Blackwell-era features that change how existing Hopper-era code should
be tuned. The validation target for this repo is **SM120 (cloud RTX PRO 6000
Blackwell, GB202)**; SM100 numbers for B200 are included only where they
differ. (The local smoke-test host on this branch is RTX 3090 Ti / sm_86 —
Ampere, three architectural generations older — and cannot launch any of
the SM120-specific paths; only sm_86-portable kernels run locally.)

## Occupancy hard numbers (raw guide §1.4.1.1)

| Quantity | SM100 (B200, CC 10.0) | SM120 (GB206, CC 12.0) |
|----------|------------------------|-------------------------|
| Max concurrent warps / SM | 64 | **48** |
| Register file | 64K × 32-bit / SM | 64K × 32-bit / SM |
| Max registers / thread | 255 | 255 |
| Max thread blocks / SM | 32 | 32 |
| Shared memory capacity / SM | 228 KB | **128 KB** |
| Max shared memory / thread block | 227 KB | **99 KB** |

**Repo observation:** the 48-warp cap on SM120 means a kernel optimized
for 64-warp Hopper/B200 occupancy will hit a different occupancy floor
on cloud RTX PRO 6000 Blackwell. Re-measure with CUDA events, CUPTI Activity,
NVBit/static features, or read-only covariates rather than NCU counters.

## Thread block clusters (raw guide §1.4.1.2)

- Clusters supported on Blackwell (introduced on Hopper).
- **Portable cluster size**: 8.
- **Nonportable opt-in**: B200 supports cluster size 16 after setting `cudaFuncAttributeNonPortableClusterSizeAllowed`. Other Blackwell parts' nonportable ceiling must be probed.
- Larger clusters reduce the number of active blocks visible across the GPU — use `cudaOccupancyMaxActiveClusters`, not the per-block variant.
- DSMEM accesses should follow the same **32-byte coalescing rules as global memory** when aligned (raw guide §1.4.1.2 explicit recommendation).

**Repo policy:** for cluster-using kernels on SM120, **do not trust `cudaOccupancyMaxPotentialClusterSize` alone**. The runtime can report a potential above the actual device ceiling. Always launch-probe with `cudaLaunchKernelEx` — this repo's `cluster_support_probe.cu` encodes the protocol.

## Memory hierarchy (raw guide §1.4.2)

### L2 cache

- B200 (GB200): L2 capacity **126 MB** (up from 96 MB on H100).
- L2 persistence APIs same as Ampere+:
  - Set-aside portion size: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`. Use `cudaGetDeviceProperties` to read `l2CacheSize` and `persistingL2CacheMaxSize` first.
  - Per-stream access-policy window: `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)` where `attr.accessPolicyWindow` carries `base_ptr`, `num_bytes`, `hitRatio`, `hitProp`, `missProp`.
  - (Not `cudaDeviceSetCacheConfig` / `cudaFuncSetAttribute`; those are for a different L1/shared carveout decision — see raw guide §4.13.)

### Unified L1/Texture/Shared

- B200: combined L1 + texture + shared capacity per SM = **256 KB** (matches Hopper).
- Carveout selectable at runtime via `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, N)`. Supported carveouts on B200 and H100: **0, 8, 16, 32, 64, 100, 132, 164, 196, 228** KB.
- CUDA reserves **1 KB** shared / block → the "228 KB per SM" limit yields max **227 KB per block** on B200.
- **Static shared-memory allocations stay capped at 48 KB**; dynamic allocations above 48 KB require explicit opt-in via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N)`.

**Repo observation:** SM120 has a smaller combined L1/shared pool (128 KB/SM) and correspondingly smaller per-block max (99 KB). Any kernel using the B200-scale 228 KB shared path will fail launch on SM120 without a carveout strategy.

### HBM3 / HBM3e (B200 only)

- B200 supports HBM3 and HBM3e memory up to **180 GB** capacity.
- SM120 is a consumer part; uses GDDR instead. Memory-bandwidth-bound kernels need retuning between SM100 and SM120.

## NVLink (raw guide §1.4.3)

- B200 implements **5th-generation NVLink** (more links per GPU, higher bandwidth than H100).
- Transparent to the CUDA model: `cudaDeviceEnablePeerAccess` and `cudaDeviceCanAccessPeer` unchanged.
- Peer transfers automatically route over NVLink when available, else PCIe.
- Multi-GPU is out-of-scope for the current single-SM validation track.

## High-priority tuning recommendations

The raw guide §1.2 re-emphasizes these general rules; they still apply to Blackwell:

- Parallelize sequential code.
- Minimize host↔device transfers.
- Tune launch config (block size, shared mem, register budget) for device utilization.
- Coalesce global memory accesses on 32-byte segments.
- Avoid redundant global-memory accesses.
- Avoid long diverged sequences within a warp.

These are generic — the Blackwell-specific part is the numbers above. Don't over-index on the list.

## Tensor-core pathway (one-line handoff)

The raw guide §1.4 does not cover tcgen05 directly. For tuning the
tensor-core path, consult the PTX and SASS cards (see the See also block), then
collect allowed non-root timing and static-instruction evidence before making a
quantitative claim.

## Common tuning decisions

| Decision | SM120 hint | SM100 (B200) hint |
|----------|------------|-------------------|
| Block size for full occupancy | `warps × 32 / max_blocks_per_SM`, block count bounded by 48-warp SM ceiling | 64-warp SM ceiling; larger blocks feasible |
| Shared-mem budget per block | ≤ 99 KB | ≤ 227 KB (need opt-in above 48 KB static) |
| Cluster size default | Start at 2–4, probe up to 8 portable, launch-test beyond | Start at 4–8, opt-in for 16 if needed |
| Register budget / thread | ≤ 255; watch spill with `ptxas -v` | Same |

## Dynamic shared-memory opt-in (the 48-KB cliff)

Any dynamic shared-memory allocation above 48 KB requires an explicit opt-in per kernel:

```cpp
size_t bytes = 96 * 1024;                                          // 96 KB
cudaFuncSetAttribute(my_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     bytes);
my_kernel<<<grid, block, bytes>>>(...);                            // launch with dynamic smem
```

Static `__shared__` allocations stay capped at 48 KB regardless; only the dynamic path opts in above the line.

## Register spill vs occupancy tradeoff

**Repo policy:** when `ptxas -v` reports register spills on SM120, the first lever is `-maxrregcount` (or per-kernel `__launch_bounds__`). Do not immediately drop to a smaller block size — the SM120 48-warp cap already restricts how much slack the block-size axis has. Measure occupancy change first with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

## Modeling context (papers)
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf) — Blackwell-specific microbenchmark measurements for SM120; pair with this card for hardware-validated tuning parameters.

## See also

- `.claude/knowledge/blackwell/blackwell-compatibility.md` — forward-compat / JIT / `-gencode` rules.
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md` — cluster programming model and DSMEM discipline.
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` — PTX-level Blackwell deltas (tcgen05 family).
- `.claude/knowledge/sass/sass-instruction-reference.md` — per-arch SASS mnemonic provenance (Blackwell-new vs Hopper-era).
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need per-CC-specific capability flags not in this card (e.g. CC 10.0a vs 12.0a architecture-conditional feature gates) — see raw guide §1.3 and `blackwell_compatibility/01_blackwell-architecture-compatibility.md`.
- You need GA200 / GB202 / GB203 / GB205 / GB207 consumer-SKU deltas (this card covers GB206 and B200 explicitly) — NVIDIA's per-SKU numbers ship only in product datasheets, not the tuning guide.
- You need the full carveout-to-register mapping for a specific occupancy scenario — use `cudaOccupancyMax*` APIs directly with the target kernel; the guide does not close-form this.
- You need Independent-Thread-Scheduling compat details — `blackwell_compatibility/01_blackwell-architecture-compatibility.md` §1.4.3 has the `-gencode=arch=compute_60,code=sm_100` opt-in for Pascal scheduling.
