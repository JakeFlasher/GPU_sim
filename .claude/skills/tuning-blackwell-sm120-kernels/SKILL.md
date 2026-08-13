---
name: tuning-blackwell-sm120-kernels
description: Tunes CUDA kernels for sm_120 (RTX PRO 6000 Blackwell GB202, cloud target) and sm_100 (B200 datacenter); resolves Blackwell compatibility / forward-compat / -gencode questions. Use when prompts mention sm_120, sm_120a, sm_100, sm_100a, RTX PRO 6000, GB202/GB200/B200, NVL72, NV-HBI, TMEM, tcgen05.mma, OMMA, QMMA, NVFP4, FP4/FP6/MX, Blackwell occupancy / SMEM ceiling, L2 persistence, dual-die unified GPU, cluster size 16 nonportable opt-in, or Hopper→Blackwell migration. Corrects the forum misconception that consumer/workstation Blackwell (sm_120) lacks clusters/DSMEM/TMA — it does not.
---

# Tuning Blackwell SM120 / SM100 Kernels

The primary cloud target is **sm_120 (RTX PRO 6000 Blackwell, GB202, CC 12.0)**. Measure directly on the selected device; do not assume Hopper-tuned launch configs carry over.

## Cards

| Topic | Card |
|---|---|
| Occupancy / SMEM carveout / cluster ceilings / L2 persistence / HBM | `.claude/knowledge/blackwell/blackwell-tuning.md` |
| Cubin vs PTX survival / -gencode recipes / Pascal-scheduling JIT opt-in | `.claude/knowledge/blackwell/blackwell-compatibility.md` |
| Historical tensor-core latency / II context and re-measurement cautions | `.claude/knowledge/blackwell/blackwell-tensor-perf.md` |
| RTX PRO 6000 specifically — feature surface verified against forum FUD | `.claude/knowledge/blackwell/rtx_pro_6000_blackwell_capabilities.md` |
| PTX-level Blackwell deltas (`tcgen05.mma`, TMEM, FP4/MX) | `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` |
| Cluster programming model + DSMEM discipline | `.claude/knowledge/cuda/cuda-clusters-dsmem.md` |
| Cluster measurement features and prior-art routing | `.claude/knowledge/cuda/cuda-cluster-modeling.md` |
| SASS-level Blackwell mnemonics (LDT/STT/UTCCP/UTCSHIFT/OMMA/QMMA) | `.claude/knowledge/sass/sass-instruction-reference.md` |
| Blackwell historical context (TMEM, dual-die unified GPU) | `.claude/knowledge/patents/arch-blackwell.md` |

## SM120 vs SM100 — these are not the same

| Quantity | SM100 (B200, CC 10.0) | SM120 (GB206/GB202, CC 12.0) |
|---|---|---|
| Max warps / SM | 64 | **48** |
| SMEM / SM | 228 KB | **128 KB** |
| Max SMEM / block | 227 KB | **99 KB** |
| TMEM | 256 KB | smaller (consumer-tier) |
| Memory | HBM3e | GDDR (consumer/workstation) |
| Cluster size 16 nonportable | Yes (opt-in) | Probe via launch-test |

A kernel tuned for B200 64-warp occupancy will hit a different floor on SM120's 48-warp cap. Re-measure with NCU, CUDA events, CUPTI Activity, NVBit/static features, and useful covariates as appropriate.

## Critical rules for Blackwell-specific work

1. **`tcgen05.mma` REPLACES `wgmma.mma_async`** — Hopper SASS dumps with `WARPGROUP`/`*GMMA` will not run on Blackwell as-is.
2. **NVFP4 / FP4 block-scaled MMA is sm_120a-conditional** — needs `-gencode=arch=compute_120a,code=sm_120a` opt-in. sm_120 alone is insufficient.
3. **B200 is two reticle-limited dies but ONE CUDA GPU** — `cudaGetDeviceCount` returns 1 per B200; do NOT treat as two GPUs.
4. **Single-thread issue ≠ single-thread compute** — `tcgen05.mma` is initiated by one thread; the hardware still runs the matrix op in parallel.
5. **Cluster size 16 on B200 needs explicit `cudaFuncAttributeNonPortableClusterSizeAllowed`** — for SM120 / other Blackwell parts, **launch-test** rather than trust `cudaOccupancyMaxPotentialClusterSize` which can over-report.
6. **DSMEM accesses follow 32-byte coalescing rules** like global memory.

## Performance numbers (B200, marketing; NVIDIA quotes sparse-mode)

| Metric | Sparse peak | Dense (≈ half of sparse) |
|---|---|---|
| FP8 / FP6 | 20 PFLOPS | ~10 PFLOPS |
| FP4 (NVFP4) | 40 PFLOPS | ~20 PFLOPS |
| HBM bandwidth | 5× Hopper | — |
| NVLink 5.0 + SHARP | 4× Hopper all-reduce | — |

These are platform-level numbers (NVL72 multiplies again). Match scope when quoting.

## Workflow for "tune this kernel for Blackwell"

1. Identify the SM target (SM100 datacenter vs SM120 consumer/workstation). Defaults to **SM120** in this repo.
2. Open `blackwell-tuning.md` for hard ceilings.
3. If kernel uses Hopper `wgmma`, plan migration to `tcgen05.mma` (PTX) / `OMMA/QMMA/UTC*` (SASS) — `ptx-isa-sm100-blackwell.md` + `sass-instruction-reference.md`.
4. If kernel uses FP4 / MX, ensure `-gencode=arch=compute_120a,code=sm_120a` is set — `blackwell-compatibility.md`.
5. If kernel uses clusters, re-validate with launch-test on SM120 — `rtx_pro_6000_blackwell_capabilities.md`.
6. Re-evaluate occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and elapsed-time sweeps. Do NOT scale Hopper-tuned launches.
7. Measure on the target device, repeat noisy cases, and record enough host and tool context to interpret the result.

## Common pitfall: "I have an RTX 5090, does it support clusters?"

**Yes.** All consumer Blackwell parts (sm_120 / GB202–GB207) support clusters / DSMEM / TMA / WGMMA-class operations. A widely-cited forum post claiming the contrary was incorrect (likely driver/toolchain misconfiguration). See `rtx_pro_6000_blackwell_capabilities.md` for the Codex-verified errata.

## Cross-references

- PTX-level instruction syntax → `interpreting-ptx-isa`.
- SASS-level mnemonic decoding → `analyzing-sass-nvbit-traces`.
- General CUDA tuning (occupancy, coalescing) that applies to all generations → `optimizing-cuda-kernels`.
- Why TMEM exists at all (patent provenance) → `reasoning-about-nvidia-architecture-history`.
- Historical sm_86 v2.1 measurement context → `measurement/pinned-protocol-v21.md`.
- nvcc / `-gencode` decisions → `compiling-cuda-artifacts`.
