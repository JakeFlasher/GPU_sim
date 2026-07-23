---
Use when: explaining the consumer-side RTX 40 architecture; placing FP8 in the consumer Tensor Core; understanding why Ada has Hopper's Transformer Engine but not Hopper's cluster / DSMEM / TMA; placing the L2 cache scaling jump (6 MB → 96 MB).
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md does not have a dedicated Ada section; Ada is referenced as a consumer Hopper-era part with FP8 Tensor Core but without the cluster/TMA/WGMMA features.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/5.ada.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: no Ada-specific patent in the corpus; Ada inherits Volta-era HMMA datapath patent (US10338919B2) and extends it to FP8.
Out of scope: Hopper's data-center features (`arch-hopper.md`); RT-Core 3rd gen / DLSS 3 / SER (parallel graphics line, not in this corpus).
Version family: Ada Lovelace architecture, 2022. Representative product: AD102 (RTX 4090, 2022-10). Compute capability 8.9.
---

# Ada Lovelace (2022, AD102) — consumer Hopper-era flagship

Ada is the **consumer counterpart to Hopper**. Same Tensor Core
generation (4th gen, with FP8), same Transformer Engine support,
same TSMC 4N process node (per gpu-arch-wiki; NVIDIA_patentd.md does
not discuss process nodes) — but **no cluster, no DSMEM, no TMA, no
WGMMA**. Ada is positioned as gaming + light compute; Hopper is
positioned as data-center training.

For full architectural rewrites, the relevant card is `arch-hopper.md`.
Ada's role here is to mark **what features got into the consumer line
and what didn't**.

## SM organization (AD102)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | 64 |
| INT32/FP32 Core | **64** (mixed, like GA102) |
| **Tensor Core (4th gen)** | **4** (per SM) |
| RT Core (3rd gen) | 1 (per SM) |
| LD/ST Unit | 16 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory + L1 (combined) | 99 KB / SM |
| L2 cache (whole chip, AD102) | **96 MB** (16× GA102's 6 MB) |
| Memory controllers | 12 × 32-bit GDDR6X |
| SM count (RTX 4090) | 128 (out of 144 in full AD102) |

**Notes:**

- Same dual-FP32 path as GA102 (consumer Ampere): one pure-FP32 lane
  + one mixed-FP32/INT32 lane = 128 FP32 cores effective.
- 12 GPCs in AD102, each with 6 TPCs → 144 SMs total in full die.
- **L2 jumps from 6 MB (GA102) to 96 MB (AD102), 16× growth.** Most
  consequential change for memory-bound workloads.

## Tensor Core 4.0 (consumer flavor)

Same Tensor Core generation as Hopper, with FP8 (E4M3 / E5M2)
support and the same Transformer Engine acceleration. But **without**:

- WGMMA (warpgroup-level MMA)
- Distributed Shared Memory (DSMEM)
- Thread Block Clusters
- Tensor Memory Accelerator (TMA)

So FP8 inference on Ada is real and works, but the staging /
cooperation infrastructure that Hopper exposes for FP8 *training*
is not on Ada. Ada is the consumer FP8 inference platform.

## Other Ada-specific features

### RT Core 3rd gen
- **Opacity Micro-Map (OMM)**: hardware-accelerated alpha test for
  ray tracing.
- **Displaced Micro-Mesh Engine**: hardware-accelerated geometry
  micro-tessellation.
- ~2× ray-tracing throughput vs Ampere.

### DLSS 3 — frame generation
- Generates intermediate frames using the Optical Flow Accelerator
  (OFA) hardware unit.
- Net effect: rendering at lower base framerate, frame-generated
  upscaled output. Doubles perceived FPS in many titles.

### Shader Execution Reordering (SER)
- Hardware feature for ray-tracing pipelines to reorder shader
  invocations and reduce divergence.
- Programmer-visible via the SER API in Vulkan / DirectX.

These are graphics-pipeline features that don't extend the compute
architectural-evolution thesis this corpus tracks. Mentioned for
completeness.

## What Ada does NOT do

- Does not introduce Thread Block Clusters — those are sm_90+ (Hopper
  data-center, and later Blackwell SM100/SM120 with subtleties).
- Does not introduce DSMEM — Hopper-only (and Blackwell forward).
- Does not introduce TMA — Hopper-only (and Blackwell forward).
- Does not introduce WGMMA — Hopper-only.
- Does not introduce Tensor Memory (TMEM) — Blackwell-only.

If a CUDA workload uses any of these features and you need it on
Ada, **it won't work**. Use Hopper or Blackwell.

## Why the L2 jump matters

GA102 → AD102: 6 MB → 96 MB L2. Reasoning:

- L2-resident working sets are essentially "free" (no DRAM round
  trip).
- 96 MB is enough to keep many activations / weight slabs in L2 for a
  full forward pass on common inference batch sizes.
- Memory-bandwidth-bound inference workloads tuned for AD102 are
  much harder to port back to GA102 — the L2 hit-rate model breaks.

## Why a future agent cares

- "FP8 on consumer GPU" first lands on Ada (sm_89). Hopper data-center
  is sm_90.
- An Ada workload that works on `cp.async` will not work with TMA;
  TMA is sm_90+. If a workload needs TMA, it's not Ada.
- L2 size step (6 MB → 96 MB) means tuning numbers from GA102 are
  not transferable to AD102.
- Compute capability **8.9** is the Ada code; Hopper is **9.0**.
- Ada's SM is closer to GA102 than to GH100 — 4 PBs, mixed
  FP32/INT32 lane, no FP64 in SM (FP64 only via Tensor Core, very low
  ratio).

## See also

- `arch-ampere.md` — GA102 is the prior consumer-line generation;
  many design decisions descend from there.
- `arch-hopper.md` — Hopper data-center counterpart with the cluster
  / DSMEM / TMA / WGMMA features Ada lacks.
- `arch-blackwell.md` — successor; SM100 (datacenter B200) and SM120
  (consumer GB206) split happens here.
- `evolution-precision.md` — FP8 timeline (Ada is consumer FP8 entry).
- `patent-family-map.md` — full patent index (Ada has no entry of its
  own).

## Escalate to raw when

- You need AD102 vs AD103 vs AD104 deltas — gpu-arch-wiki has only
  AD102; the NVIDIA Ada whitepaper is canonical.
- You need RT Core 3 internals — not in this corpus; consult NVIDIA's
  Ada whitepaper or the OptiX 7 documentation.
- You need DLSS 3 / Frame Generation behavior — NVIDIA's DLSS SDK
  and tech-blog posts are canonical.
- You need SER programming model — the Vulkan / DirectX SER spec.
