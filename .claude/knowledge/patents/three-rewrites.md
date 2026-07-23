---
Use when: framing a high-level discussion of NVIDIA's GPU architectural evolution; explaining why Tesla → Blackwell is not a "more features each year" story; placing any single architectural change on the spine of the corpus; teaching another agent why Fermi / Volta / Hopper are the load-bearing breakpoints.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §5 (the three-architectural-rewrites table) plus the cross-cutting framing in §1 Overview.
Supporting sources: per-generation cards (`arch-tesla.md` → `arch-blackwell.md`).
Patents anchored here: US9639479B2 (Fermi rewrite), US10338919B2 (Volta rewrite), US20230289189A1 (Hopper rewrite).
Out of scope: the per-generation specifics that follow each rewrite (those are in the per-arch cards).
Version family: Tesla (2006) through Blackwell (2024). Mirror snapshot: 2026-05.
---

# The Three Architectural Rewrites — corpus thesis

The **central thesis** of this corpus: nearly two decades of NVIDIA
architectural evolution are not a continuous "more features per year"
story. They are **three rewrites** at three different levels of the
machine, separated by long consolidation phases.

**Caveat — read before applying**: this three-rewrites framing is the
**source document's own analytic synthesis** (NVIDIA_patentd.md §1).
It is the architectural pressure chain jointly supported by patent
text, NVIDIA's public docs, and inference. **It is not equivalent to
any internal NVIDIA design roadmap** and is not officially declared.
Read it as the document's lens for organizing patents and SASS deltas,
not as NVIDIA's stated history.

## The three rewrites

| Patent | Generation | Year | What it rewrites |
|---|---|---|---|
| `US9639479B2` | Fermi | 2010 | **Memory side**: L2 / partition becomes a software-observable, software-tunable hardware domain |
| `US10338919B2` | Volta | 2017 | **Execution path**: HMMA datapath peels out of the general FPU |
| `US20230289189A1` | Hopper | 2022 | **Cooperation boundary**: shared-memory-style CTA cooperation extends from one SM to a cluster (DSM) |

Each rewrite breaks **at least one** default that prior generations
took for granted.

## What each rewrite changes

### Rewrite 1 — Fermi: memory side becomes a hardware domain

**Before**: the memory side (L2 / partition / DRAM) is just "where
addresses end up." Software has no clean interface to control or
interpret cache behavior, write-back paths, or atomic execution
location.

**After**: a stable per-address processing domain. Same global
address always lands in a consistent L2 slice / partition. Cache
modifiers (`.cg/.cs/.cv/.wt`) let software express access intent.
Atomic ordering, write-back timing, and completion paths now have a
common location.

**Patent**: `US9639479B2` (priority 2009-09-23, filed 2010-09-22). The
patent's importance is not "introduce L2" — it's **fixing four things
into one set of hardware boundaries**:

- `partition unit`
- `L2 slice`
- `FB DRAM interface`
- `.cg / .cs / .cv / .wt` modifiers

For details: `arch-fermi.md`.

### Rewrite 2 — Volta: matrix path peels out, ITS replaces lockstep

**Before**: the general-purpose FPU is the only path for major
operators. Implicit warp lockstep is the default execution semantics —
all 32 threads of a warp are at the same PC by default.

**After**: matrix multiply-accumulate gets its own dedicated datapath
(`HMMA`). Threads have independent PCs (Independent Thread
Scheduling, ITS); divergence and reconvergence are explicit programmable
objects.

**Two simultaneous changes**:

1. **Execution path**: `RF banks → crossbar → operand collectors →
   HMMA datapath → result queue → RF writeback`. Matrix is no longer
   one mode of the general FPU.
2. **Execution model**: per-thread PC + convergence barriers. `__syncwarp()`
   and `_sync` family become important.

**Patent**: `US10338919B2` (priority 2017-05-08, filed 2017-11-29) for
the HMMA datapath. ITS rests on the Maxwell-era convergence-barrier
patent `US10067768B2` (earliest priority 2014-07-18) but becomes the
default in Volta.

For details: `arch-volta.md`.

### Rewrite 3 — Hopper: cooperation boundary crosses the SM

**Before**: a single SM is the default boundary for shared-memory-
style hardware cooperation. Blocks exchange state through global
memory or coarser kernel boundaries.

**After**: Thread Block Clusters + Distributed Shared Memory + TMA +
transaction-aware barriers + WGMMA close a producer-consumer loop
**across multiple SMs in a single GPC**.

**Critical reading**: Hopper is rewrite #3 not because clusters were
added, but because **five interlocking interfaces form a closed
loop**. Reading any one in isolation misreads the rewrite as "module
X got stronger."

1. **Numerical contract** — FP8 (E4M3 / E5M2) + Transformer Engine
2. **Consumer granularity** — `WGMMA` (warp-group MMA, 128 threads)
3. **Transport path** — TMA (`cp.async.bulk.tensor`)
4. **Shared scope** — Thread Block Cluster + DSM
5. **Async + barriers** — transaction-aware mbarrier
   (`mbarrier.complete_tx`)

The full chain:

```text
global / L2 tensor tile
  -> TMA descriptor-driven bulk copy
  -> local SMEM or DSM resident tile
  -> transaction-aware barrier confirms data arrival
  -> scheduler releases the waiting consumer group
  -> warp-group / WGMMA consumes the prepared matrix
```

**Patents**: `US20230289189A1` (DSM), `US12141082B2` (TMA),
`US20230289242A1` (transaction barrier). All filed 2022-03-10.

For details: `arch-hopper.md`.

## Generations as consolidation phases

Between rewrites, generations consolidate the prior rewrite's machinery:

| Generation | Year | Phase |
|---|---|---|
| Tesla | 2006 | Baseline machine |
| **Fermi** | **2010** | **Rewrite #1** (memory side) |
| Kepler | 2012 | Consolidation: data supply (`RF→crossbar→collector` loop) |
| Maxwell | 2014 | Consolidation: synchronization (software scoreboard + convergence barrier prepared, but ITS not yet default) |
| Pascal | 2016 | Consolidation: FP16 in CUDA cores + NVLink 1.0 (Unified Memory + Compute Preemption per gpu-arch-wiki, not in source §5) |
| **Volta** | **2017** | **Rewrite #2** (HMMA datapath + ITS default) |
| Turing | 2018 | Consolidation: precision fork (INT8/INT4/Binary) + uniform datapath + RT Core |
| Ampere | 2020 | Consolidation: `cp.async`, BF16/TF32, structured sparsity, MIG |
| Ada | 2022 | Consolidation: consumer FP8, L2 jump, RT Core 3 |
| **Hopper** | **2022** | **Rewrite #3** (cluster + DSM + TMA + transaction barrier + WGMMA) |
| Blackwell | 2024 | Extends rewrite #3 (TMEM, `tcgen05.mma`, dual-die unified GPU, FP4/MX) |

## Three layers of pressure

The same evolution can be read as **three layers of pressure** that
keep returning:

| Pressure | What it forces |
|---|---|
| **Memory-side control / semantics** | When `L2`, atomic, residency, coherence get promoted from background details, the memory side must become a real hardware domain (Fermi rewrite). |
| **Datapath specialization** | When matrix computation, low-bit formats, and warp-uniform data flows can no longer share the general path, execution and data supply must split (Volta rewrite, plus the entire precision-fork line through Turing/Ampere/Hopper/Blackwell). |
| **Coordination** | When `cp.async`, TMA, WGMMA, cluster cooperation become primary, sync primitives and interconnect must upgrade (Hopper rewrite, plus the interconnect-evolution line). |

## Three axes of expansion

Stitching the three rewrites together, the corpus's main thread is:

- **Processing ownership and control semantics** expand outward from
  inside the SM toward the memory side. (Fermi rewrite + the
  evolution-memory-side card.)
- **Execution and data-supply paths** split from a general-purpose
  machine toward a specialized machine. (Volta rewrite + the
  evolution-data-supply and evolution-precision cards.)
- **The cooperation boundary** keeps pushing outward from a single
  SM to the cluster, the interconnect, and the package. (Hopper
  rewrite + the evolution-execution-model and evolution-interconnect
  cards.)

## What Blackwell adds — not a fourth rewrite

Blackwell pushes two more boundaries that hadn't been fully made
explicit before:

1. Tensor Core near-end storage moves from implicit implementation
   layer into **explicit ISA / toolchain object** (TMEM).
2. Single die's physical implementation no longer naturally equals
   the **single GPU that software sees** (dual-die unified GPU on
   B200).

Both of these matter, but neither is at the same level as the three
rewrites — they extend rewrite #3's cooperation-boundary
expansion rather than introducing a new rewrite type.

## Why this framing matters for downstream work

- **Patent research**: pick one of the three rewrites and the rest of
  the corpus is organized around it. Don't read NVIDIA's 17,000+
  patents linearly.
- **Simulator modeling**: each rewrite tells you which hardware
  state must be modeled as a first-class object. Fermi: per-address
  partition + cache policy. Volta: dedicated matrix datapath +
  per-thread PC + barriers. Hopper: cluster scheduling + DSMEM
  address mapping + transaction barriers + warp-group consumer
  semantics.
- **Architecture explanation / teaching**: the three rewrites are the
  spine. A future agent explaining why Volta exists at all should
  reach for "Volta is rewrite #2 — matrix path peeled out + ITS
  default" before any implementation detail.

## Reading order for a future agent

If you have to brief someone on architectural evolution and have only
a few minutes:

1. Read this card.
2. Read the three patents' abstracts (or the relevant paragraphs in
   `arch-fermi.md`, `arch-volta.md`, `arch-hopper.md`).
3. Skim the consolidation generations between rewrites.
4. Explain the three axes of expansion and the three layers of
   pressure.

## See also

- `arch-fermi.md`, `arch-volta.md`, `arch-hopper.md` — the rewrites
  themselves.
- `arch-blackwell.md` — what Blackwell extends.
- `evolution-memory-side.md` — Fermi onward.
- `evolution-execution-model.md` — Volta onward.
- `evolution-data-supply.md` — Tesla collector → Volta HMMA →
  Hopper TMA → Blackwell TMEM.
- `evolution-synchronization.md` — Maxwell scoreboard → Hopper
  transaction barrier.
- `evolution-precision.md` — Volta FP16 → Blackwell FP4.
- `evolution-interconnect.md` — Tesla crossbar → Blackwell NVL72.
- `patent-family-map.md` — full patent index.
- `INDEX.md` — corpus router.

## Escalate to raw when

- You need to argue for or against this framing in detail — read
  NVIDIA_patentd.md §5 in full and the three patents on Google Patents.
- A future generation (Rubin, etc.) makes the framing creak — update
  this card with a fourth row in the table only after a defining
  patent + clear architectural-default-broken story emerges. Do not
  add speculative fourth rewrites.
