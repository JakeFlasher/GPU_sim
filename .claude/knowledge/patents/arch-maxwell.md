---
Use when: explaining the SMM split into Processing Blocks; reasoning about why software scoreboarding and convergence barriers exist; placing the synchronization rewrite that prepared Volta's Independent Thread Scheduling.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.2 (Maxwell discussed as the second post-Fermi consolidation step, on the synchronization axis).
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/1.maxwell.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: US9612836B2 (software-based scoreboarding, filed 2014-02-03), US10067768B2 (convergence barrier for divergent threads, earliest priority 2014-07-18).
Out of scope: Pascal's FP16 / NVLink (`arch-pascal.md`); Volta's ITS realization (`arch-volta.md`); Tensor Core.
Version family: Maxwell architecture, 2014–2016. Representative product: GM204 (GTX 980, 2014-09). Compute capability 5.x.
---

# Maxwell (2014, GM204) — synchronization rewritten toward Volta

Maxwell is **not a breakpoint** — like Kepler, it is a post-Fermi
consolidation phase. What it does is **rewrite synchronization**:
push from busy-bit / hardware scoreboard toward programmable
barriers and explicit convergence-barrier mechanisms. This is the
hardware groundwork Volta's Independent Thread Scheduling needs.

## SM organization (GM204 — "SMM")

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 8 |
| INT32+FP32 Core | **128** (4 × 32) |
| LD/ST Unit | 32 |
| SFU | 32 |
| Register file / SM | 64 KB |
| Shared memory / SM | **96 KB** (no longer shared with L1) |
| L1 cache / SM | 32 KB (shared with texture cache) |
| L2 cache (whole chip) | 2 MB |
| Memory | GDDR5 |

**Key structural changes from Kepler:**

- **Renamed SMX → SMM**.
- **GPC layer reintroduced** (Kepler had removed it).
- **Independent FP64 units removed** — back to FP32 cores doing
  emulated FP64 at much lower ratio. Maxwell is intentionally
  consumer-focused.
- CUDA core count drops 192 → 128 per SM.
- **Compute units split into 4 Processing Blocks (PBs)**, each with
  its own warp scheduler, instruction buffer, and register file. This
  PB structure persists through Volta / Turing / Ampere / Hopper —
  reading any later SM diagram, you'll see four PBs.
- **L1 and shared memory no longer share a pool** (Fermi/Kepler had
  configurable carveout; Maxwell separates them).
- **L2 cache changes from write-back to write-through**.

Per-CUDA-core efficiency is reportedly **1.4× Kepler**; total
**per-watt efficiency 2× Kepler** (per gpu-arch-wiki, sourced from
NVIDIA's GM204 launch material; not in NVIDIA_patentd.md). Maxwell
is the architecture that re-establishes NVIDIA as the consumer-
graphics efficiency leader.

## The synchronization rewrite — what Maxwell prepares

Maxwell pushes synchronization from hardware-implicit dependency
checking toward **programmable barriers**. Two patents bound this:

### US9612836B2 — software-based scoreboarding

- **Filed** 2014-02-03, NVIDIA. Inventors include Choquette and Giroux
  (both names that recur in Volta/Hopper patents).
- **Title**: "System, method, and computer program product for
  implementing software-based scoreboarding."
- **Claim**: a `dependency barrier instruction` carrying an immediate
  value and a register identifier. Comparing the immediate to the
  register's value gates whether to dispatch the subsequent
  instruction. Scoreboard registers are 6-bit counters (0–63); the
  scheduler checks whether the value is at-or-below the immediate.
- **Why it matters**: prior schemes implemented scoreboarding **inside
  the hardware** — software did not control it. The patent's
  innovation is **moving control to software/compiler**: the compiler
  emits explicit "wait on this dependency" instructions. Cheaper
  hardware (less circuit complexity, less power) and more flexible
  scheduling.
- **PTX/SASS realization**: SASS instructions like `DEPBAR`,
  `LDGDEPBAR` (post-Maxwell for async loads) descend from this idea.

### US10067768B2 — convergence barrier for divergent threads

- **Earliest priority** 2014-07-18 (filed 2015-07-13), NVIDIA.
  Inventors: Choquette, Tirumala.
- **Title**: "Execution of divergent threads using a convergence
  barrier."
- **Claim**: replace stack-based / priority-based reconvergence with
  **explicit barrier objects**. Threads entering a divergence region
  register on a barrier; threads at the convergence point wait;
  threads still executing carry on; once all participants arrive,
  the barrier releases all of them.
- **Critical innovation**: the divergence-management mechanism is
  **decoupled from thread scheduling**. Either can change without
  the other. This is the precondition for Volta's
  **Independent Thread Scheduling (ITS)**.
- **YIELD mechanism**: provides a forward-progress guarantee — no
  thread can indefinitely block any other. Without this, ITS would
  be incorrect.
- **SASS-level realization**: `BSSY`, `BSYNC`, `BMOV`, `BREAK` are
  the compiler-inserted instructions that implement convergence
  barriers. Not visible in CUDA C++ or PTX — the compiler emits them.

## Why Maxwell is preparation, not realization

Maxwell has the hardware mechanisms but the **public execution model
hasn't changed yet**. Threads in a warp are still expected to advance
together by default. The convergence-barrier patent is the
**hardware substrate**; Volta is when ITS becomes the
**default execution semantics** and per-thread PCs replace one PC
per warp.

Reading order:

1. Maxwell — **mechanism enabled** (compiler can emit explicit
   barriers; software scoreboard available).
2. Pascal — adds FP16 and NVLink but does **not** flip the execution
   model.
3. Volta — flips the execution model: implicit warp lockstep is no
   longer the default; threads have independent PCs.

Without the Maxwell-era patents, Volta's rewrite would have nothing
to land on.

## What Maxwell sustains from Fermi

Memory side: still Fermi-style L2/partition with cache modifiers.
Data supply: still Kepler-style RF/crossbar/collector. Maxwell does
not rewrite either axis — its scope is the synchronization axis only.

## Why a future agent cares

- A pre-Maxwell GPU **cannot** support a software-scoreboard-style
  trace; trying to model `DEPBAR` semantics on Tesla/Fermi/Kepler is
  wrong.
- The 4-PB SM split starts here and continues forward. If you see a
  trace from a 4-PB-style SM, it's Maxwell or newer.
- The convergence-barrier patent's **YIELD** semantics is what makes
  CPU-style mutexes safe on Volta+. If a question asks why a mutex
  loop deadlocks pre-Volta but works on Volta, the answer is "YIELD
  + convergence barrier + ITS, all introduced through this Maxwell
  patent + Volta's adoption of it as the default execution model."
- L2 changes write-back → write-through, so Maxwell-era write-amplification
  characteristics differ from Fermi/Kepler.

## Reference reading

- "In-depth analysis of Modern GPU control-bits design" —
  https://zhuanlan.zhihu.com/p/1985083294845510474 (Chinese; the
  source NVIDIA_patentd.md cites for the software-scoreboard
  micro-architecture).

## See also

- `arch-fermi.md` — the prior memory-side rewrite Maxwell builds on.
- `arch-kepler.md` — Kepler's data-supply loop that Maxwell inherits.
- `arch-volta.md` — Volta is when these mechanisms become the default
  execution semantics (ITS).
- `evolution-synchronization.md` — Maxwell's scoreboard + convergence
  barrier is one node; full line: busy-bit → scoreboard → ITS →
  mbarrier → transaction barrier.
- `evolution-execution-model.md` — how Maxwell prepares Volta's ITS.
- `patent-family-map.md` — full patent IDs and dates.

## Escalate to raw when

- You need exact GM200 / GM204 / GM206 deltas — gpu-arch-wiki has only
  GM204; the GM200 has no official whitepaper, only third-party
  analysis.
- You need claim-level patent reading — open US9612836B2 / US10067768B2
  on Google Patents.
- You need the precise per-warp barrier machinery NVIDIA built — see
  the convergence-barrier patent text and "Re-converging control flow
  on NVIDIA GPUs" by Collabora (https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/).
