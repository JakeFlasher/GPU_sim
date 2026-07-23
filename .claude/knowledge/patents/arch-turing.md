---
Use when: explaining why INT8/INT4 Tensor Core support exists; placing the first RT Core; understanding when the Tensor Core became an inference-class object (not just training); reasoning about Turing-era SM organization.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.3 (Turing as the precision-fork follower of Volta).
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/3.Turing.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: no Turing-specific patent in the corpus; Turing extends Volta's HMMA datapath (US10338919B2) with new precisions.
Out of scope: Volta's HMMA / ITS rewrite (`arch-volta.md`); Ampere's `cp.async` (`arch-ampere.md`); RT-Core internals beyond the headline.
Version family: Turing architecture, 2018. Representative product: TU102 (RTX 2080 Ti, 2018-09). Compute capability 7.5.
---

# Turing (2018, TU102) — Tensor Core forks for inference; uniform datapath emerges

Turing is **not a breakpoint** — it is the first maturation step on
Volta's matrix datapath. The architectural significance is that the
**Tensor Core stops being a half-precision training accelerator and
starts forming an independent inference branch** (INT8 / INT4 / Binary).

Turing also introduces the first **RT Cores** for hardware ray
tracing. The RT-Core line is parallel to the compute-architecture
line that this corpus tracks; we mention it but do not deep-dive.

## SM organization (TU102)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | 64 |
| INT32 Core | 64 |
| **Tensor Core (2nd gen)** | **8** (per SM) |
| RT Core | 1 (per SM, new) |
| LD/ST Unit | 16 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory / SM | 64 KB |
| L1 cache / SM | 4 KB |
| L2 cache (whole chip, TU102) | 5.5 MB |
| Memory controllers | 12 × 32-bit GDDR6 |

**Notes:**

- Same 4-PB SM structure as Volta.
- **Tensor Core 2.0**: Volta-shape (4×4×4 GEMM), now adds **INT8** and
  **INT4** input types (with INT32 accumulate). Also Binary (1-bit)
  mode for some shapes.
- **First-generation RT Core** for ray-triangle intersection and BVH
  traversal — used by RTX raster pipelines (DXR / Vulkan RT) and by
  CUDA via OptiX.
- L1 per-TPC bandwidth is ~2× Pascal. Note "TPC" returns as a layout
  concept in the Turing memory diagrams.
- 12 × 32-bit memory controllers using GDDR6 (no consumer HBM).

## What Turing extends on Volta's HMMA datapath

Same datapath (`RF → crossbar → operand collectors → matrix datapath
→ result queue → RF writeback`) — Turing adds **precision branches**:

- INT8: `m8n8k16`, `m16n8k16`, `m16n8k32` shapes
- INT4: `m8n8k32`, `m16n8k32`, `m16n8k64` shapes
- Binary (1-bit): `m8n8k128`, `m16n8k128`, `m16n8k256` shapes (with
  `xor.popc` / `and.popc` reduction modes)

These all live in the same `IMMA` / `BMMA` SASS instruction families
that descend from Volta's `HMMA`. See `sass-evolution-timeline.md`.

## Inference vs training fork

Volta's first-gen Tensor Core was a **training accelerator**: FP16
input is the training-favored format. Turing's INT8/INT4 are the
**inference-favored** formats — quantized weights, no need for
gradient precision, throughput-per-watt matters more than numerical
fidelity.

This is the first time the Tensor Core line forks by intended
workload:

- Volta: **training** (FP16 + FP32 accumulate)
- Turing: **+ inference** (INT8, INT4, Binary)
- Ampere: **training-friendly extension** (BF16, TF32, structured
  sparsity)
- Hopper: **mixed-precision training + low-bit inference** (FP8 with
  Transformer Engine)
- Blackwell: **block-scaled FP4/FP6** (NVFP4/MX) for both training
  and inference

For the format-by-format timeline see `evolution-precision.md`.

## Uniform datapath emerges (the warp-uniform datapath)

Turing introduces the first batch of **uniform-path SASS instructions**.
These handle warp-uniform values — values where all 32 threads in a
warp would compute the same result. Storing one copy and broadcasting
saves register space and arithmetic bandwidth.

Turing adds (visible in SASS):

- `R2UR`, `S2UR` — move from vector register / special register to a
  uniform register
- `UIADD3`, `UIMAD`, `UISETP` — uniform integer add3, MAD,
  set-predicate
- `ULDC`, `ULEA`, `USEL`, `USHF` — uniform load-from-const, LEA, select,
  funnel-shift
- `VOTEU` — voting with uniform destination

This is the "URF / UDP" path mentioned in `arch-volta.md`'s discussion
of "split paths by data form, not just by numeric type." Turing is
when this hardware becomes visible in the ISA.

Other Turing SASS additions:

- `LDSM` — load matrix from shared memory (matrix data supply for
  Tensor Cores)
- `BMMA` — bit (binary) matrix MMA
- INT-MMA variants

See `sass-evolution-timeline.md` for the full per-generation table.

## Patents on the Turing line

The corpus has **no Turing-specific patent**. Turing builds on:

- US10338919B2 (Volta HMMA datapath) — extended with INT8/INT4/Binary
  precision branches.
- US10067768B2 (convergence barrier) — ITS still default, no change.
- US9612836B2 (software scoreboarding) — same machinery.

Turing's main architectural innovation outside Tensor-Core forking is
the **RT Core**, which the patent corpus tracked here does not cover
(it is its own patent line, not in `NVIDIA_patentd.md`).

## What Turing does NOT do

- Does not introduce `cp.async` — that's Ampere.
- Does not change ITS — Volta's default.
- Does not introduce thread block clusters — Hopper.
- Does not introduce the BF16/TF32 training-friendly precisions —
  Ampere.

## Why a future agent cares

- "INT8 Tensor Core" first hits Turing. If a workload targets INT8
  Tensor Core, the floor is sm_75.
- "Binary Tensor Core" (1-bit `xor.popc`) is also Turing+. Rare in
  practice; mostly used by binary-network research.
- Turing TU102 has only **4 KB L1** per SM — much smaller than Volta's
  combined-128KB. If trace replay shows L1 hit-rates inconsistent
  with Volta numbers, check generation.
- TU102 uses **GDDR6**, not HBM. Memory-bandwidth-bound kernel tuning
  numbers transfer poorly between Volta and Turing.
- The first batch of `U*` (uniform-path) SASS instructions are
  Turing — pre-Turing, no uniform datapath in ISA.

## See also

- `arch-volta.md` — Volta's HMMA datapath that Turing extends.
- `arch-ampere.md` — next generation; cp.async + BF16/TF32 +
  structured sparsity.
- `evolution-precision.md` — INT8/INT4 in the precision timeline.
- `sass-evolution-timeline.md` — full per-generation SASS instruction
  history; uniform-path family begins here.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need RT Core internals — outside this corpus; consult NVIDIA's
  Turing whitepaper and OptiX documentation.
- You need exact PTX `mma.sync` shape × precision tables — see
  `.claude/knowledge/ptx/ptx-isa-tensor-cores.md`.
- You need TU102 vs TU104 vs TU106 deltas — gpu-arch-wiki has only
  TU102; consult the NVIDIA Turing whitepaper.
