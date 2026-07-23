---
Use when: explaining the first Tensor Core; reasoning about why the matrix path is split out of the FPU; placing Independent Thread Scheduling (ITS); understanding `__syncwarp()` and the `_sync` family; placing the second of the three architectural rewrites.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.3.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/2.volta.md (SM config, "milestone architecture"); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: US10338919B2 (generalized acceleration of MMA / HMMA datapath, priority 2017-05-08, filed 2017-11-29); US10067768B2 (convergence barrier, Maxwell-era patent that becomes the default model in Volta).
Out of scope: Turing's INT8/INT4 split (`arch-turing.md`); Ampere's `cp.async` / BF16/TF32 (`arch-ampere.md`); Hopper's cluster (`arch-hopper.md`).
Version family: Volta architecture, 2017–2018. Representative product: GV100 (Tesla V100, 2017-06). Compute capability 7.0.
---

# Volta (2017, GV100) — rewrite #2: matrix path peels out, ITS replaces lockstep

Volta is **rewrite #2** of the three-rewrites thesis (`three-rewrites.md`).
Two simultaneous rewrites:

1. **Execution path**: matrix multiply-accumulate gets its own
   datapath — `HMMA` is no longer just a mode of the general FPU.
2. **Execution model**: implicit warp lockstep stops being the
   default; threads have independent PCs (Independent Thread
   Scheduling, ITS).

The matrix-path rewrite is the more cited one (it kicks off everything
that becomes Tensor Core in later generations). The ITS rewrite is
quieter but at least as deep — it changes what "warp" actually means.

## SM organization (GV100)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | 64 |
| INT32 Core | 64 |
| FP64 Core | 32 |
| **Tensor Core (1st gen)** | **8** (4 × 2) |
| LD/ST Unit | 32 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory + L1 (combined) | 128 KB / SM |
| L2 cache (whole chip) | 6 MB (64 slices × 96 KB) |
| Memory | HBM2, 8 × 512-bit memory controllers |

**Key changes from Pascal:**

- **CUDA core split**: no longer one combined FP+INT core. Now 64 FP32
  + 64 INT32 cores per SM, **executing concurrently** for ~2×
  effective throughput in mixed code.
- **Tensor Core 1.0 introduced**: 8 per SM, each performs a 4×4×4 GEMM
  per cycle (FP16 input + FP32 accumulate), equivalent to ~64 FP32
  ALUs of throughput.
- **Warp Scheduler:Dispatch ratio is back to 1:1** (Maxwell/Kepler
  was 1:2). With L0 instruction cache.
- **Back to 4 Processing Blocks per SM** (Pascal had 2).
- **Shared memory + L1 reunited**, configurable carveout up to 128 KB.
- **L0 ICache per Processing Block** — first time a per-PB icache.
- L2 organization: 64 slices × 96 KB, each slice reads 32 B/cycle =
  2048 B/cycle aggregate L2 read bandwidth.

## Rewrite #1: HMMA datapath peels out — US10338919B2

- **Priority** 2017-05-08, **filed** 2017-11-29, NVIDIA. Inventors:
  Ming Y. Siu, Jack H. Choquette, Jonah M. Alben.
- **Title**: "Generalized acceleration of matrix multiply accumulate
  operations."

### The new chain

```text
RF banks
  -> crossbar
  -> operand collectors
  -> HMMA datapath
  -> result queue
  -> RF writeback
```

Compare to Tesla/Kepler's `RF → crossbar → collector → ALU`. The
matrix path:

- Adds a **dedicated `HMMA datapath`** between collectors and result.
- Adds a **result queue** that buffers writes; writeback to RF is
  arbitrated, not immediate.
- Lets one set of operands be reused across multiple dot-product
  passes without re-reading from RF/crossbar.

### Why peel matrix out of the general FPU?

Three reasons:

1. **Throughput target mismatch**: matrix multiply does many
   dot-products on a fixed tile, with throughput far higher than
   ordinary FMA. Sharing one path starves both.
2. **Data-supply pattern mismatch**: matrix tile data should be
   staged in groups near the operand collector and reused across
   sub-computations. General FPU traffic doesn't have this reuse
   structure.
3. **Numeric-type fork**: V100's first Tensor Core is **FP16 input +
   FP32 accumulate** — already more than just "another FPU mode."
   Pulling it out lets later generations (BF16 / TF32 / FP8 / FP4)
   land on the same dedicated path without polluting the general FPU.

### What the patent does NOT prove

**The patent does NOT prove that HMMA is fully detached from the
original floating-point machine** (NVIDIA_patentd.md §2.3, this is a
load-bearing limit on what can be claimed about Volta's matrix path).

Evidence:

- `Fig. 9` shows `HMMA datapath 930` and `FP64 datapath 940` **sharing
  operand collectors and result queue**.
- `Fig. 14` says some HMMA implementations may share pipeline stages
  with the FP64 FMA datapath.

So the more accurate reading is:

- **Volta specializes the front+middle of the matrix dot-product.**
- It does **not** rebuild data supply, completion, and writeback —
  those are partly shared with FP64.
- The patent supports the chain `RF → crossbar → collectors → HMMA →
  result queue` as the main path, but **not** a definitive read on
  warp/lane fragment mapping, collector allocation policy, or
  per-stage pipeline timing.

Future agents reading "the Volta Tensor Core works like X" should
treat anything beyond this main chain as **unsubstantiated by the
patent alone** and source it elsewhere.

### What this enables forward

- **BF16, TF32, FP8, FP4** all descend along this peeled-off path.
  Every later format addition is on the dedicated matrix datapath,
  not on the general FPU.

## Rewrite #2: ITS — implicit warp lockstep gone

The **convergence barrier** patent (US10067768B2, Maxwell-era) becomes
the default execution model in Volta. The change is twofold:

1. **Per-thread PC**: each thread has independent program counter and
   call stack. The hardware no longer assumes "all threads of a warp
   are on the same PC right now."
2. **Reconvergence is explicit**: divergence and reconvergence are
   programmed objects via convergence barriers (compiler-inserted
   `BSSY`, `BSYNC`, `BMOV`, `BREAK`).

### Before vs after

**Pre-Volta (implicit warp lockstep):**
- All active threads in a warp share one forward-progress point.
- Divergence handled by `active mask + reconvergence stack`.
- Threads on different paths run serially; reconverge at hardware-
  determined IPDom point.

**Volta+ (ITS):**
- Each thread has its own PC.
- Same-PC threads in a warp run together by scheduler choice, but
  the "all together" property is no longer guaranteed.
- Reconvergence via convergence barriers — compiler controls where
  it happens.

### What this means for programmers

- `_sync`-suffixed warp intrinsics (`__shfl_sync`, `__ballot_sync`,
  etc.) now require an explicit thread mask. Pre-Volta, you could
  rely on "all 32 threads of the warp." Post-Volta, you must say
  which.
- `__syncwarp()` becomes important — explicitly tells the hardware
  "this set of threads must converge here."
- CPU-style mutexes become safe, because the YIELD mechanism in the
  convergence-barrier patent (US10067768B2) provides forward-progress
  guarantees no thread can indefinitely block another.

## Reading the diagrams

The NVIDIA_patentd.md document includes two contrasting figures:

- "Pre-Volta: implicit warp lockstep" — all threads at one PC, one
  active mask.
- "Post-Volta: ITS execution split" — multiple PCs interleaved
  through scheduler.

If a future agent wants the visual, see NVIDIA_patentd.md §2.3
(images `v2-5fcd952fbe034e7e554ddab304ce57e0_1440w.png` and
`v2-95733279e6dd59e5850a202cd2990a0e_1440w.png`).

## Why this is rewrite #2 (vs Pascal/Maxwell consolidation)

It's a true breakpoint because **two foundational defaults** are
broken simultaneously:

1. The general-purpose FPU **stops being the only matrix path**.
2. The warp **stops being a lockstep unit by default**.

Neither was true on any prior generation. Both remain true on every
later generation. That makes Volta the second of three rewrites.

## Reference reading

- "Tensor Core microarchitecture deep dive" —
  https://zhuanlan.zhihu.com/p/2028548125518755193
- "Modern GPU control-flow design" —
  https://zhuanlan.zhihu.com/p/1985794546739720618
- Re-converging control flow on NVIDIA GPUs (Collabora, 2024-04-25).

## Why a future agent cares (pitfalls)

- **"Warp acts as one" is only correct pre-Volta.** Volta+ requires
  explicit `__syncwarp(mask)` to recover that property. Trace replay
  and simulator modeling that assume the pre-Volta default on Volta+
  hardware will miss interleaved divergence and break.
- **HMMA is the FIRST matrix instruction in SASS.** Anything
  matrix-shaped pre-Volta (manually-tiled GEMM running on CUDA cores)
  is not HMMA. Don't conflate "Tensor Core path" with "did GEMM."
- **Volta's first-gen Tensor Core is ONLY FP16 input + FP32
  accumulate.** Programs targeting BF16/TF32/FP8/FP4/etc. won't run
  on V100.
- **Convergence-barrier patent is Maxwell-era; ITS default is Volta.**
  Mechanism vs default semantics — keep the distinction. Pre-Volta
  hardware HAS the barrier mechanism but doesn't make it the default,
  so CPU-style mutexes deadlock.
- **The HMMA patent does NOT prove the matrix path is fully detached
  from the FPU.** Don't infer pipeline structure / collector
  allocation / lane mapping from this patent alone — it's not
  pinned down. (See "What the patent does NOT prove" above.)

## See also

- `three-rewrites.md` — Volta as rewrite #2.
- `arch-maxwell.md` — convergence-barrier mechanism that Volta makes
  default.
- `arch-pascal.md` — FP16 precursor that Volta promotes to a Tensor Core.
- `arch-turing.md`, `arch-ampere.md` — the Tensor Core generations
  that mature on Volta's HMMA chain.
- `evolution-execution-model.md` — implicit warp lockstep → ITS →
  cluster.
- `evolution-data-supply.md` — RF/collector → HMMA datapath → cp.async
  → TMA → TMEM.
- `evolution-precision.md` — Volta first stable Tensor Core types
  (FP16 + FP32 accumulate).
- `evolution-interconnect.md` — NVLink 2.0 lands here.
- `patent-family-map.md` — full patent IDs and dates.

## Escalate to raw when

- You need precise warp/lane fragment mapping or operand-collector
  allocation policy for V100 — the patent doesn't pin these down;
  consult academic microbenchmark papers ("Dissecting the NVIDIA Volta
  GPU Architecture via Microbenchmarking", arXiv:1804.06826).
- You need exact PTX-level `mma.sync` syntax — see
  `.claude/knowledge/ptx/ptx-isa-tensor-cores.md`.
- You need a claim-level read of US10338919B2 — Google Patents.
- You need the formal PTX memory consistency model that ITS demands —
  `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md`.
