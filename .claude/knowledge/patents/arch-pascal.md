---
Use when: explaining when FP16 first hit a CUDA core; placing NVLink 1.0; understanding Compute Preemption; reasoning about Unified Memory and HBM2 transition.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md does not have a dedicated Pascal section; Pascal is referenced as "FP16 precursor" in §2.3 and the precision-evolution discussion.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/1.pascal.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: no new defining patent in the corpus; Pascal is on the consolidation track.
Out of scope: Volta's Tensor Core / HMMA datapath (`arch-volta.md`); Volta's ITS (also `arch-volta.md`); Tensor-Core data types beyond FP16.
Version family: Pascal architecture, 2016. Representative product: GP100 (Tesla P100, 2016-04). Compute capability 6.x.
---

# Pascal (2016, GP100) — FP16 precursor and NVLink 1.0

Pascal is **not a breakpoint** — like Kepler and Maxwell it is a
maturation step. The two facts about Pascal that future generations
build on are:

1. **FP16 in the CUDA core** — the first hint of what Volta later
   makes a dedicated Tensor Core path. Format precedes datapath.
2. **NVLink 1.0** — interconnect starts answering "how far does the
   cooperation boundary extend" rather than just "is the bandwidth
   enough."

**Source attribution note**: NVIDIA_patentd.md only references Pascal
as "FP16 precursor" (§2.3). The HBM2, Compute Preemption, Unified
Memory, and NVLink 1.0 details below come from gpu-arch-wiki, not
from the patent corpus. They are facts about Pascal but are not
patent-grounded in this corpus.

## SM organization (GP100)

| Component | Count |
|---|---|
| Warp Scheduler | 2 |
| Dispatch Unit | 4 |
| INT32+FP32 Core | 64 (2 × 32) |
| Independent FP64 Unit | 32 (2 × 16) |
| LD/ST Unit | 16 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory / SM | 64 KB or 96 KB depending on SKU |
| L1 cache / SM | 32 KB |
| L2 cache (whole chip, GP100) | 4 MB |
| Memory | HBM2 (GP100) / GDDR5X (consumer) |

**Key changes from Maxwell:**

- SM is **smaller** — only 2 PBs (not 4). 64 cores per SM instead of 128.
  But total SM count is much higher (60 in GP100 vs 16 in GM204), so
  total compute-per-chip and shared-memory bandwidth go up.
- **FP64 returns**: 32 independent FP64 units per SM (2 × 16).
- **FP16 newly supported in CUDA cores** (compute capability 6.0). One
  FP16 op per cycle but as a 2-wide vec2, so peak FP16 throughput is
  2× FP32 throughput on GP100. (On consumer GP102 etc. the FP16 ratio
  was deliberately gimped to 1:64 for market segmentation.)
- **HBM2 first appears** on the data-center GP100 SKU. Consumer parts
  stick to GDDR5X.
- 8 × 512-bit memory controllers, each with 512 KB L2 = 4 MB total L2.
- Shared memory per SM × SM count > Maxwell, so effective shared-mem
  capacity goes up substantially even though per-SM number is similar.

## What's new beyond pure SM organization

### NVLink 1.0
- **Dual-direction bandwidth ~160 GB/s** between two GP100 GPUs.
- First proprietary NVIDIA high-bandwidth GPU↔GPU interconnect.
- Replaces PCIe as the primary data path for multi-GPU compute.
- Starting point for the line: NVLink 1 → 2 (Volta) → 3 (Ampere) →
  4 (Hopper) → 5 (Blackwell) + NVL72.

### Unified Memory
- CPU and GPU share virtual address space.
- Page faults trigger automatic migration; programmer doesn't
  manually `cudaMemcpy`.
- Foundation for the host-device data-handling story that runs through
  modern CUDA.

### Compute Preemption
- Kernels can be preempted at **instruction level**, not at thread
  block boundaries.
- Critical for interactive workloads (debugging, GUI rendering) — a
  long compute kernel doesn't lock up the GPU.
- Pre-Pascal (Kepler), preemption was at thread-block boundary;
  hitting a debugger breakpoint inside a block locked everyone else
  out.

### FP16 — the precursor
Pascal's FP16 is **dual-FP16 in the standard FP32 path**, not a
dedicated Tensor Core. It is the format that arrives first; the
datapath that consumes it as a tile is Volta's HMMA. So:

- Pascal: FP16 input/output to ALU.
- Volta: dedicated `HMMA datapath` for FP16 tiles with FP32
  accumulator.

Reading any Pascal or Volta material, this distinction matters. See
`evolution-precision.md`.

## What Pascal does **not** do

- Does not introduce Tensor Cores. That's Volta.
- Does not change execution semantics — implicit warp lockstep still
  the default.
- Does not introduce async copy / TMA — that's Ampere / Hopper.
- Does not change the Maxwell-style synchronization mechanisms — they
  exist as compiler-inserted, but ITS is not yet the default.

## Patents on the Pascal line

The corpus has **no Pascal-specific patent**. Pascal builds on existing
patents:

- US9639479B2 (Fermi) — memory side / cache modifiers unchanged.
- US7339592B2 (Tesla collector) — data supply unchanged from Kepler.
- US9612836B2, US10067768B2 (Maxwell) — synchronization machinery
  available, not yet flipped to default.

The **next** major patent on the architectural timeline is Volta's
HMMA datapath patent (US10338919B2, 2017). See `arch-volta.md`.

## Why a future agent cares (pitfalls)

- **"FP16 supported" without a Tensor Core means Pascal.** A tuning
  hint that references FP16 throughput **and** a Tensor Core in
  scope cannot be answered on Pascal — floor is Volta.
- **Compute Preemption is a Pascal+ feature.** Don't infer it on
  Kepler / Maxwell — debugger interaction will differ.
- **Pascal's SM is 2 PBs, not 4.** Volta returns to 4 PBs. Don't
  infer Volta layout from Pascal.
- **Consumer GP102 deliberately gimps FP16** to 1:64 vs FP32 (only
  GP100 has 2× FP16 in the FPU). Trace-replay numbers from a
  consumer Pascal SKU don't transfer to GP100.
- **HBM2 is GP100-only.** Consumer Pascal uses GDDR5X. Memory-
  bandwidth-bound tuning numbers don't transfer between SKUs.

## See also

- `arch-maxwell.md` — the prior synchronization rewrite Pascal builds on.
- `arch-volta.md` — Volta turns FP16 into a dedicated Tensor Core path.
- `evolution-precision.md` — Pascal's dual-FP16 → Volta HMMA → Turing
  INT8/4 → Ampere BF16/TF32 → Hopper FP8 → Blackwell FP4/MX.
- `evolution-interconnect.md` — NVLink 1.0 starts here.
- `patent-family-map.md` — full patent index (Pascal has no entry of
  its own; this card explains why).

## Escalate to raw when

- You need GP100 vs GP102 vs GP104 memory deltas — NVIDIA's Pascal
  whitepaper is the canonical source; gpu-arch-wiki only has GP100.
- You need Compute Preemption microarchitecture — not in this corpus;
  consult NVIDIA's GP100 whitepaper.
- You need NVLink 1 PHY-level details — NVLink is a proprietary
  spec; partial public material in NVIDIA's NVLink whitepaper.
- You need Unified Memory page-fault costs / driver behavior — the
  CUDA Programming Guide §"Unified Memory" is the canonical source.
