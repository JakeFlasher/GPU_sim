---
Use when: confirming the cloud target's feature surface (RTX PRO 6000 Blackwell, GB202, sm_120, CC 12.0); verifying sm_120 supports clusters / DSMEM / TMA / WGMMA-class operations (it does — corrects a common forum-post misconception); placing per-SM resource ceilings, TMEM size, FP4/FP6/FP8 availability on consumer/workstation Blackwell; deciding whether a kernel needs sm_120a opt-in for FP4 block-scaled MMA; sizing cluster launches against sm_120's portable-vs-nonportable cluster-size limits.
Primary raw source: references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md (§1.4.1.1 Occupancy, §1.4.2 Memory System — CC 12.0 callouts inline); references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md.
Supporting sources: references/NVIDIA_Manuals/cuda_guide/ (programming guide, target-specific clauses); .claude/knowledge/blackwell/blackwell-tuning.md; .claude/knowledge/blackwell/blackwell-compatibility.md; .claude/knowledge/blackwell/blackwell-tensor-perf.md; .claude/knowledge/ptx/ptx-isa-sm100-blackwell.md; .claude/knowledge/patents/arch-blackwell.md.
Repo touchpoints: tools/cloud/preflight.sh + tools/cloud/preflight_manifest.py (capability probing); workloads/simbench_bw/l2/cluster_remote_reduction/cluster_remote_reduction.cu (cluster workload that exercises the corrected sm_120 feature surface).
Out of scope: SM100 (B200 datacenter) occupancy/SMEM deltas (in `blackwell-tuning.md`); `-gencode` / cubin-vs-PTX forward-compat rules (in `blackwell-compatibility.md`); per-precision tensor-core II/latency numbers (in `blackwell-tensor-perf.md`); per-SKU pricing / thermal / power data; non-CUDA features (DisplayPort, RT Core, NVENC).
Version family: GB202 silicon, sm_120 / CC 12.0, CUDA 13.x, driver 610.x current (595.x is legacy). Historical verification notes were preserved from the parent repo; re-check raw NVIDIA docs before making a new external claim.
---

# RTX PRO 6000 Blackwell — Feature Surface on sm_120 (Consumer / Workstation)

**Status**: historical verification notes preserved from the parent repo. The
load-bearing current rule is to probe compile, launch, and numeric behavior on
the actual target before relying on a feature in this fork.

**Why this card exists**: initial KB + web research suggested sm_120 consumer/workstation Blackwell lacked clusters / DSMEM / TMA. This is **wrong**. The authoritative NVIDIA CUDA Programming Guide + PTX ISA target specifications say otherwise. A developer-forum post claiming "RTX 5090 ... does not support clusters" was incorrect (likely a driver / toolchain misconfiguration, not an architectural absence).

## Device identity

- **Silicon**: GB202 (monolithic die, 750 mm², 92.2 B transistors, 192 SMs in full die; 188 SMs enabled on RTX PRO 6000).
- **Memory**: 96 GB GDDR7 ECC, 128 MB L2 cache (full GB202), 4x DisplayPort 2.1b.
- **Compute capability reported at runtime**: `12.0` (not `12.0a`). `sm_120a` is a compiler target (`-arch=sm_120a`), not a separate reported CC.
- **Family**: sm_12x consumer/workstation family. Same silicon as GeForce RTX 5090 (also GB202 at sm_120 with fewer SMs enabled and less L2 — the 5090 cut is 96 MB vs the PRO 6000's full 128 MB).
- **GeForce-vs-PRO tensor-rate asymmetry (added 2026-07-06, RTX Blackwell whitepaper-verified)**: GeForce Blackwell SKUs run tensor ops with **FP32 accumulate at HALF the FP16-accumulate rate**; RTX PRO 6000 runs FP32 accumulate at **full rate**. Same ISA, same CC 12.0 — a documented per-SKU throughput divergence the GB206-vs-GB202 cross-SKU generalization experiment (plan D7) must reproduce. Model it as a per-SKU ArchProfile card field (`tensor_fp32_acc_rate_factor`: 0.5 GeForce / 1.0 PRO), never as a shared sm_120 constant.
- **NOT datacenter**: sm_10x datacenter Blackwell (B200 / GB200 superchip) is a separate family with HBM3e + MCM dual-die + tcgen05 + TMEM. sm_100a cubins do NOT run on sm_120, and vice versa.

## Feature matrix (sm_120 vs sm_90a Hopper vs sm_100a datacenter Blackwell)

| Feature | sm_90a Hopper (H100) | sm_100a Datacenter Blackwell (B200) | sm_120 Consumer/Workstation (RTX PRO 6000) | Evidence |
|---|---|---|---|---|
| **Thread-block clusters** (`barrier.cluster`, `%cluster_ctarank`, cluster launch) | ✅ | ✅ | ✅ (from CC 9.0+ per CUDA Programming Guide) | Codex verification CLAIM 8; web forum claim of "no clusters on RTX 5090" was wrong |
| **DSMEM** (`.shared::cluster` state space, `mapa`, `getctarank`) | ✅ | ✅ | ✅ (enabled by clusters per CUDA docs) | Codex verification CLAIM 9 |
| **TMA** (`cp.async.bulk`, `cp.async.bulk.tensor`) | ✅ (sm_90+) | ✅ | ✅ (same PTX interface as Hopper) | Codex verification CLAIM 7; empirically confirmed on RTX 5090 by CloudRift blog |
| **TMA cluster multicast** (`cp.async.bulk.tensor.multicast::cluster`) | ✅ (sm_90a) | ✅ | ✅ (baseline cluster multicast; not the tcgen05 2-CTA cooperative form) | Codex + PTX ISA |
| **`mbarrier.*` CTA-scoped** (`arrive.expect_tx`, `try_wait`) | ✅ (sm_90+) | ✅ | ✅ | PTX ISA target notes |
| **`mbarrier.*` cluster-scoped** (`release.cluster`, `acquire.cluster`) | ✅ (sm_90+) | ✅ | ✅ | Codex verification CLAIM 10 |
| **`wgmma.*`** (warpgroup MMA, `wgmma.fence`) | ✅ (sm_90a ONLY) | ❌ | ❌ (`ptxas error: Instruction 'wgmma.fence' not supported on .target 'sm_120'`) | Codex CLAIM 6; PTX target note "Requires sm_90a" |
| **`tcgen05.*`** (`tcgen05.mma/ld/st/cp/alloc/dealloc`) | ❌ | ✅ (sm_100a / sm_100f / sm_101a / sm_110a) | ❌ | Codex CLAIM 4; NVIDIA/cutlass issue #2800 restricts BlockScaledMmaOp to sm_100a |
| **TMEM** (Tensor Memory, 256 KB per SM) | ❌ | ✅ (accessed only via tcgen05) | ❌ (accumulators live in registers, like Ampere) | Codex CLAIM 5; Colfax CUTLASS tutorial; gau-nernst tcgen05 blog |
| **2-CTA cooperative MMA** (`tcgen05.mma.cta_group::2`) | ❌ | ✅ (sm_100a) | ❌ | tcgen05 is sm_100-only |
| **Warp-level `mma.sync.kind::f8f6f4/mxf4/mxf4nvf4`** | ❌ | ✅ (both tcgen05 and warp-level) | ✅ (warp-level, sm_120a target) | Codex CLAIM 11; CUTLASS SM120 docs |
| **Block-scaled MMA** (`.block_scale`, `.scale_vec::1X/2X/4X`) | ❌ | ✅ (full set including `.block16/.block32`) | ✅ (`scale_vec::2X`, `.scale_vec::4X` supported on sm_120a; `.block16`/`.block32` are tcgen05-only and NOT on sm_120) | Codex CLAIM 12 CONTRADICTION correction |
| **Sparse 2:4 MMA** (`mma.sp`) | ✅ (sm_80+ BF16/FP16/INT8; sm_89 FP8) | ✅ | ✅ (dense BF16/FP16/FP8/INT8 baseline; sparse FP4 `mma.sp.kind::mxf4nvf4` is sm_120a-gated, unverified on RTX PRO 6000) | Codex CLAIM 13 (probe-gated for sparse FP4) |
| **FP4 (E2M1) dense MMA** | ❌ | ✅ | ✅ via warp `mma.sync` (sm_120a) | — |
| **FP6 (E3M2, E2M3) dense MMA** | ❌ | ✅ | ✅ via warp `mma.sync` (sm_120a) | — |
| **FP8 (E4M3, E5M2) dense MMA** | ✅ | ✅ | ✅ (sm_120 baseline) | — |
| **NVFP4 (block-scaled FP4, E4M3 micro-block × FP32 tensor)** | ❌ | ✅ | ✅ (hardware; dense GEMM works via CUTLASS C++; MoE grouped GEMM broken in CUTLASS as of late 2025 without FlashInfer SM120 patches) | Codex CLAIMS 17-18 |
| **NVFP4 KV cache** | ❌ | Partial (TRT-LLM / FlashInfer 2026) | ❌ in major inference stacks as of late 2025 | Codex CLAIM 19 (UNCERTAIN — version-pinned) |

## The sm_120 vs sm_120a vs sm_120f distinction (load-bearing)

This mirrors the sm_90/sm_90a and sm_100/sm_100a patterns:

| Target | Meaning | Forward-compat | Use for |
|---|---|---|---|
| `-arch=compute_120 -code=sm_120` | Baseline feature set | Forward-compatible via PTX JIT | Kernels that do NOT use FP4/FP6/NVFP4 or block-scaled MMA |
| `-arch=compute_120f -code=sm_120f` | Family feature set (CUDA 12.9+) | Forward-compatible within sm_12x family (sm_120 + sm_121) | Dense NVFP4 GEMM, block-scaled MMA, `cvt.e2m1`, FP6 `cvt` — this is the recommended default on CUDA 13+ per CUTLASS |
| `-arch=compute_120a -code=sm_120a` | Architecture-accelerated, not forward-compatible | NOT forward-compatible (runs only on CC 12.0 exactly) | Sparse FP4 `mma.sp.kind::mxf4nvf4` and a few other sm_120a-only opcodes |

**Runtime behavior**: the hardware reports CC `12.0` regardless of which target was used to compile. The `a` / `f` suffix is purely a compiler contract, not a runtime distinction.

**Empirical ecosystem issue (late 2025)**: NVIDIA/cutlass issue #3096 reports that forcing `-arch=compute_120a` for FP4 MoE kernels SEGFAULTS on RTX PRO 6000. The workaround is `-arch=compute_120f` + CUDA 13.0+. Treat `sm_120a` features as probe-gated until the software ecosystem stabilizes.

## Recommended build flags (Codex verification)

For a benchmark suite targeting both H100 (sm_90a) and RTX PRO 6000 Blackwell (sm_120 / sm_120a):

```bash
# Hopper-only kernels (wgmma microbenches, Hopper-specialized FA-3):
-gencode arch=compute_90a,code=sm_90a

# Baseline SM120 kernels (no FP4/FP6/NVFP4, no block-scaled MMA):
-gencode arch=compute_120,code=sm_120

# Dense FP4/FP6/NVFP4 block-scaled kernels on CUDA 13+ (recommended path):
-gencode arch=compute_120f,code=sm_120f

# sm_120a-only kernels (sparse mxf4 / mxf4nvf4 — probe-gated):
-gencode arch=compute_120a,code=sm_120a
```

**Do NOT** include `sm_100` / `sm_100a` in the RTX PRO 6000 build path. They do not run on sm_120.

## Benchmark-suite implications

**Kernels that work on both H100 (sm_90a) and RTX PRO 6000 (sm_120) — portable tier**:
- HMMA FP16 / BF16 / TF32 (baseline warp `mma.sync`)
- FP8 QMMA (E4M3, E5M2)
- Sparse 2:4 BF16, FP16, FP8, INT8
- TMA-driven GEMM (`cp.async.bulk.tensor`)
- Thread-block clusters, DSMEM, cluster barrier, DSMEM reduction
- `mbarrier`-based producer/consumer pipelines, warp-specialized kernels
- Async copy (`cp.async.commit_group` / `cp.async.wait_group`)

**Kernels that work only on H100 (sm_90a) — Hopper-only tier**:
- `wgmma.*` warpgroup MMA microbenches
- `wgmma`-based FlashAttention-3 kernels
- Any `setmaxnreg` users targeting Hopper's warp-specialization ceiling

**Kernels that work only on RTX PRO 6000 (sm_120a) — Blackwell-only tier**:
- FP4 (E2M1) dense MMA
- FP6 (E3M2, E2M3) dense MMA
- NVFP4 block-scaled GEMM (dense — works; MoE grouped GEMM — broken without CUTLASS patches)
- Sparse FP4 `mma.sp.kind::mxf4nvf4` (probe-gated via sm_120a)

**Kernels NOT VIABLE on either target (datacenter B200 only)**:
- `tcgen05.mma`, `tcgen05.ld/st/cp/alloc/dealloc`
- Any kernel requiring TMEM (tensor memory allocation)
- 2-CTA cooperative MMA (`tcgen05.mma.cta_group::2`)
- tcgen05-specific `.block16` / `.block32` block-scaled variants

## Known software ecosystem gotchas (late 2025)

1. **CUTLASS Python DSL `BlockScaledMmaOp`** restricts FP4 to sm_100a only (issue #2800); workaround is patching the admissible architectures list.
2. **NVFP4 MoE grouped GEMM** on sm_120 produces garbage output without FlashInfer SM120 patches (issue #3096).
3. **vLLM backend selection** pre-0.13.x falls back to Marlin instead of native NVFP4 kernels on sm_120.
4. **SGLang blockwise FP8** on sm_120 was not supported through August 2025 (issue #9233).
5. **PyTorch mainline** pre-2.5 rejects sm_120 as unsupported ("sm_120 is not compatible with the current PyTorch installation"); requires TORCH_CUDA_ARCH_LIST="sm_120" source build or nightly.
6. **Driver pin**: CUDA 13.x recommended for reliable sm_120 / sm_120f feature surface.

## Contrasts commonly confused on web sources

- "TMA = tcgen05" — **WRONG**. TMA (Tensor Memory Accelerator) = `cp.async.bulk.tensor` = sm_90+. `tcgen05` = Blackwell 5th-gen tensor-core instructions including `tcgen05.mma` + TMEM allocation = sm_100-only.
- "RTX 5090 / RTX PRO 6000 doesn't support clusters" — **WRONG**. This claim appeared on NVIDIA Developer Forums in a CUTLASS-kernel-debug thread but is contradicted by the CUDA Programming Guide's Thread Block Cluster feature table (supported from CC 9.0+) and by the PTX ISA target notes for `barrier.cluster` (sm_90+). Likely a driver/toolchain misconfiguration or mis-phrased developer comment.
- "Consumer Blackwell is architecturally equal to Ada + FP4" — **WRONG**. sm_120 has clusters + DSMEM + TMA + mbarrier (all Hopper-era features) plus FP4/FP6/NVFP4 via extended `mma.sync` and block-scaled MMA. It's closer to Hopper + Blackwell-specific low-precision formats than to Ada.

## References

- CUDA Programming Guide — Thread Block Cluster section, Distributed Shared Memory section.
- PTX ISA Reference — `barrier.cluster`, `mbarrier`, `cp.async.bulk.tensor`, `wgmma`, `tcgen05`, `mma.sync.kind::f8f6f4 / mxf4 / mxf4nvf4`.
- NVIDIA/cutlass issues #2800 (BlockScaledMmaOp restriction), #3096 (sm_120a FP4 MoE segfault), #9233 (SGLang blockwise FP8).
- [CloudRift "Surfacing a 60% performance bug in cuBLAS"](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090) — empirical TMA usage on RTX 5090 sm_120.
- [gau-nernst "tcgen05 for dummies"](https://gau-nernst.github.io/tcgen05/) — sm_100 tcgen05 tutorial; confirms sm_12x absence.
- [Colfax CUTLASS TMEM tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — TMEM is sm_100-family only.
- [SemiAnalysis "Dissecting Nvidia Blackwell"](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor) — explicit sm_100 vs sm_12x split.
- [NVIDIA Blog "Introducing NVFP4"](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — NVFP4 format spec.
- Local KB: `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md`, `.claude/knowledge/ptx/ptx-isa-sm90-hopper.md`, `.claude/knowledge/blackwell/blackwell-compatibility.md`.

## Verification record

- Web searches performed 2026-04-24 (6 queries on device identity, TMA, tcgen05/TMEM, cluster multicast, NVFP4, sm_120 vs sm_120a).
- Codex verification of 20 claims on 2026-04-24 in the parent repo; the transcript is not present in this fork, so treat the inlined claims as historical notes backed by the cited public docs.
- Codex CONTRADICTED web claims that sm_120 lacks clusters / DSMEM / TMA / cluster-mbarrier. Those 4 corrections are the load-bearing content of this card.
- Current project caveat: pin CUDA 13.x where `compute_120f` support matters; probe compile, launch, and numeric correctness for every FP4/NVFP4/sparse/TMA/cluster kernel; do not reuse SM100 profiler-counter assumptions or TMEM/tcgen05 assumptions on SM120; NCU is the primary measurement instrument (pinned protocol v2.1, see `measurement/pinned-protocol-v21.md`), corroborated by CUPTI Activity, Nsight Systems timelines, CUDA events, NVBit, and static SASS/PTX.
