---
Use when: explaining TMEM (Tensor Memory); placing `tcgen05.mma` and its single-thread issue semantics; reasoning about FP4/MX/NVFP4 block-scaling formats; understanding the dual-die unified-GPU model; placing the SM100 (datacenter) vs SM120 (consumer) split; understanding why Hopper's WARPGROUP/WGMMA SASS encoding was replaced.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.5.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/7.blackwell.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row); references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md; references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md; references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md (Blackwell SASS).
Patents anchored here: no Blackwell-specific patent in the verified corpus; Blackwell extends Hopper's DSM (US20230289189A1), TMA (US12141082B2), and transaction barrier (US20230289242A1) and adds TMEM as a SASS-visible object.
Out of scope: Hopper's cluster/DSMEM/TMA/WGMMA introduction (`arch-hopper.md`); future Rubin (not enough public material).
Version family: Blackwell architecture, 2024–. Datacenter: GB200/B200 (SM100, compute capability 10.0). Consumer: GB202/GB206 (SM120, compute capability 12.0).
---

# Blackwell (2024, GB200 / GB202 / GB206) — TMEM, `tcgen05.mma`, dual-die unified GPU

Blackwell is **not a new fourth rewrite** of the same kind as Fermi /
Volta / Hopper. It pushes two more boundaries that hadn't been fully
made explicit before:

1. **Tensor Core near-end storage** moves from implicit
   implementation layer to **explicit ISA / toolchain object** —
   `Tensor Memory (TMEM)`.
2. **A single die's physical implementation** no longer naturally
   equals the **single GPU that software sees** — Blackwell datacenter
   SKUs are **dual-die** but appear as one CUDA GPU.

It also pushes precision down to **FP4 (NVFP4) / FP6 / MX**, with
block-scaling becoming a first-class part of the matrix description.

## SM organization (consumer SM120 — GB206 + B200 datacenter SM100 differences)

NVIDIA_patentd.md does not break out a per-SM table for Blackwell;
the gpu-arch-wiki entry shows core counts in flux. The
Blackwell tuning guide gives the official numbers:

| Quantity | SM100 (B200, CC 10.0) | SM120 (GB206, CC 12.0) |
|---|---|---|
| Max concurrent warps / SM | 64 | **48** |
| Register file | 64K × 32-bit / SM | 64K × 32-bit / SM |
| Max registers / thread | 255 | 255 |
| Max thread blocks / SM | 32 | 32 |
| Shared memory capacity / SM | 228 KB | **128 KB** |
| Max shared memory / thread block | 227 KB | **99 KB** |
| **TMEM (new)** | 256 KB | (smaller; consumer-tier) |
| L2 cache (whole chip) | up to 140 MB on GB200 | smaller on GB206 |
| Memory | HBM3e (B200) | GDDR (consumer) |

**SM120 vs SM100 — careful**: the consumer SM120 is **not a scaled-down
SM100**. It has a 48-warp cap (vs 64) and only 128 KB SMEM (vs 228 KB).
Kernels tuned for B200 occupancy targets need re-tuning for SM120 —
don't assume the SM-level model carries over.

## Performance numbers (B200, NVIDIA marketing — sparse-mode peak)

Per NVIDIA's compute-platform comparison slide:

- **FP8 / FP6 peak**: 20 PFLOPS sparse (~2.5× Hopper).
- **FP4 peak (new)**: 40 PFLOPS sparse (~5× Hopper).
- HBM model size: 6× Hopper.
- HBM bandwidth: 5× Hopper.
- NVLINK All-Reduce with SHARP: 4× Hopper.

**Caveats**:
- NVIDIA quotes sparse-mode peak; dense mode is half. Using dense FP8
  on a workload won't see the 20 PFLOPS number.
- These are platform-level numbers, combining low-bit Tensor Core,
  HBM capacity/bandwidth, NVLink collective, and software stack —
  **not** the fixed multiplier of any one datapath.
- A common confusion: GB200 (Grace + 2× B200 superchip) doubles many
  of these relative to a single B200 GPU. NVL72 (rack of 72 GPUs)
  multiplies again. Match the right scope when quoting.

## Three big new lines

### Line 1: single-die → unified multi-die single GPU

Blackwell datacenter SKUs (B200) are **two reticle-limited dies** with
**10 TB/s chip-to-chip interconnect** (NV-HBI), packaged as a single
CUDA GPU.

```text
software view:
  one CUDA GPU

physical implementation:
  die 0  <->  package chip-to-chip fabric  <->  die 1
```

NVIDIA's [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html)
explicitly says CUDA programmers should **not** treat B200 as two
GPUs — the existing programming model is extended, not bifurcated.

The architectural meaning: two things that used to be tied together
(software-visible GPU and physical die) are now decoupled.

### Line 2: FP4 / MX / Transformer Engine Gen2 — block-scaled formats

Reading "FP8 → FP4" as just halving payload bits misses the bigger
picture. Blackwell's low-bit story is a **combined contract**:

```text
payload bits
  + scale factor
  + block granularity
  + metadata / packing
  + Transformer Engine format policy
```

NVFP4 (per NVIDIA's [NVFP4 blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/))
is built jointly from:

- Micro-block scaling
- E4M3 scale factor
- Hadamard reshape
- 2D block quantization
- Stochastic rounding

So scale, metadata, block size, packing, and format choice all enter
the matrix-task description space. The Tensor Core no longer just
consumes "an FP4 tile"; it consumes "an FP4 tile with these scale
factors at this block granularity."

PTX surface uses `.kind::mxf4` / `.kind::mxf8f6f4` block-scaled MMA
forms (sm_120a). See `evolution-precision.md`.

### Line 3: implicit operand residency → explicit Tensor Memory object

Per the [CUDA Binary Utilities Blackwell instruction set](references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md):

- `tmem[URX]` is a **valid operand location**.
- `LDT/LDTM` — load matrix from **Tensor Memory** into the register file.
- `STT/STTM` — store matrix from register file back to **Tensor Memory**.
- `UTCCP` — async copy from shared memory into Tensor Memory.
- `UTCSHIFT` — shift elements within Tensor Memory.
- `OMMA` (FP4 MMA across a warp), `QMMA` (FP8 MMA across a warp).
- `UTCHMMA / UTCOMMA / UTCQMMA` — uniform forms.

Blackwell **removes / replaces** the Hopper-style explicit
`WARPGROUP` / `*GMMA` / `WGMMA` SASS encoding. Public SASS history no
longer lists `WARPGROUP/WARPGROUPSET` and parts of `*GMMA`. The new
main thread is the object set: `tcgen05.mma`, `OMMA/QMMA`, `TMEM`,
`UTC*`.

## `tcgen05.mma` — single-thread issue semantics

Per [PTX 9.0+](references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md):

- **5th-gen Tensor Core dedicated Tensor Memory** is defined as a
  **per-CTA 2D on-chip storage**.
- `D` lives in **TMEM**.
- `A` can live in TMEM or shared.
- `B` lives in shared.
- Operands take addresses or descriptors:
  - `[d-tmem]`, `[a-tmem]` for TMEM-resident matrices
  - `a-desc`, `b-desc` for shared-mem-described matrices
  - `idesc` (instruction descriptor)
  - `[scale-A-tmem]`, `[scale-B-tmem]` for block-scale factors
- **Single-thread issue semantics**: one thread can launch the entire
  matrix operation. (Compare to Hopper `wgmma.mma_async`, which
  requires all 128 threads in the warp group to execute the same
  `.aligned` instruction.)

### "Single-thread semantics" — what it means and doesn't

It describes **who initiates the computation**, not that one thread
does the actual matrix multiply-accumulate. The hardware still does
the matrix work; one thread just signals "go" with a descriptor.

It also breaks the surface constraint that "the entire warpgroup must
execute the same MMA instruction in lockstep."

### Hopper vs Blackwell shape

```text
Hopper:
  warpgroup collective issue
    -> WGMMA consumes SMEM / DSM resident tile
    -> result path returns through the group-level MMA protocol

Blackwell:
  one issuer
    -> TMEM-resident MMA task
    -> tcgen05.mma / OMMA-QMMA consume task descriptors
    -> TMEM-resident accumulator / D update
```

The "semantic center" of the matrix instruction migrates from a
register fragment held by the warpgroup → matrix state in TMEM +
descriptor + the CTA / CTA-pair instruction protocol of `tcgen05`.

## CTA-pair protocol

`tcgen05.mma` introduces **CTA-pair** semantics: even-numbered CTAs in
a CTA-pair partner with their odd peer for some matrix operations.
PTX `tcgen05` references the [CTA-Pair](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair)
and [peer-CTA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-peer-cta)
concepts; access restrictions apply.

## NVLink and multi-GPU (system level)

Blackwell continues pushing external interconnect into a larger GPU
domain:

- **5th-gen NVLink** (more links per GPU, higher bandwidth).
- **NVLink Switch** (NVSwitch successor).
- **NVL72**: rack-scale system with 72 GPUs in one NVLink fabric.

The architectural meaning: **in-package die-to-die unified view inside
one GPU and multi-GPU NVLink/NVSwitch domains begin to jointly serve
one problem** — keeping larger models, more GPUs, and finer low-bit
matrix tasks at sufficiently high interconnect efficiency.

## Cluster cooperation: extended on Blackwell

Hopper introduced clusters; Blackwell continues to support them with
some changes:

- B200 supports **cluster size 16** (vs portable 8) via
  `cudaFuncAttributeNonPortableClusterSizeAllowed`.
- Other Blackwell parts' nonportable ceilings must be probed via
  launch-test (not via `cudaOccupancyMaxPotentialClusterSize` alone —
  it can over-report).
- DSMEM accesses follow **32-byte coalescing rules** like global
  memory.

## What Blackwell does NOT do

- Does **not** introduce a brand-new architectural rewrite at
  Fermi/Volta/Hopper scale (per the three-rewrites thesis).
- Does **not** unify SM100 and SM120 — the consumer / datacenter
  split is real and must be checked per kernel.
- Does **not** keep Hopper's `WARPGROUP/WGMMA` SASS encoding — those
  are replaced. Hopper SASS dumps will not run on Blackwell SM as-is.

## Why a future agent cares

- "Tensor Memory" (TMEM) is **Blackwell-only**. If a kernel uses
  `LDT`/`STT`/`UTCCP`/`tcgen05.mma`, it requires sm_100/sm_120.
- "FP4" / "NVFP4" / block-scaled MMA is **sm_120a-conditional**
  (architecture-conditional opt-in features). Not all sm_100 implies
  full block-scaled MMA.
- Hopper code that uses `wgmma.mma_async` or `WARPGROUP` SASS will
  need to migrate to `tcgen05.mma` for Blackwell — these are not
  binary-compatible.
- Treating B200 as two GPUs in software is an error; it is one CUDA
  GPU with a die-internal fabric.
- Single-thread issue ≠ single-thread compute. Don't model
  `tcgen05.mma` as "one thread does the work."
- SM120 occupancy is half SM100 (48 vs 64 warps). Don't carry over
  Hopper-tuned launch configs.

## See also

- `arch-hopper.md` — Hopper introduces cluster + DSM + TMA + WGMMA;
  Blackwell extends DSM/TMA, replaces WGMMA with tcgen05.
- `arch-ada.md` — consumer Hopper-era part; Blackwell SM100 / SM120
  are the successor split.
- `three-rewrites.md` — Blackwell as "extends rewrite #3, doesn't
  replace it."
- `competitor-tensor-storage.md` — how Microsoft/Intel/Meta/Micron/AMD
  approach the same tensor-near-storage pressure under different
  names.
- `evolution-data-supply.md` — TMA → TMEM continuity.
- `evolution-precision.md` — FP4 / NVFP4 / MX in the timeline.
- `evolution-execution-model.md` — single-thread issue as the latest
  step in execution-granularity scope.
- `sass-evolution-timeline.md` — full Blackwell SASS deltas.
- `.claude/knowledge/blackwell/blackwell-tuning.md` —
  occupancy / shared-mem carveout / cluster ceiling.
- `.claude/knowledge/blackwell/blackwell-compatibility.md` —
  forward-compat / `-gencode` recipes.
- `.claude/knowledge/blackwell/blackwell-tensor-perf.md` —
  per-precision latency / II / unit contention from D6 microbench.
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` — PTX-
  level Blackwell deltas.

## Escalate to raw when

- You need claim-level reading of Hopper-era patents that Blackwell
  inherits — Google Patents US20230289189A1 / US12141082B2 /
  US20230289242A1.
- You need exact `tcgen05.mma` PTX syntax — see PTX 9.0+ §9.7.16
  "Tensor Core 5th Generation Instructions"; raw at
  `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md`.
- You need exact NVFP4 numerical recipe — NVIDIA's NVFP4 blog and the
  Transformer Engine Gen2 documentation.
- You need GB200 / GB202 / GB203 / GB205 / GB206 / GB207 SKU deltas —
  consult NVIDIA's per-SKU datasheets; not in this corpus.
- You need ITS forward-compat for Pascal-style scheduling on
  Blackwell — `references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md`
  §1.4.3.
