---
Use when: you have a specific question and need the shortest path to the right card; you want a problem-oriented index instead of a topic-oriented one. Sister card to `INDEX.md`.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md as a whole, organized by problem type.
Supporting sources: INDEX.md (the topic-oriented router); the per-arch and thematic cards.
Patents anchored here: (router card; full patent index is in `patent-family-map.md`).
Out of scope: content; this card only routes.
Version family: corpus snapshot 2026-05-02.
---

# Problem Playbook — question-driven router

If `INDEX.md` is the topic-oriented router (organized by GPU
generation and thematic axis), this is the **problem-oriented**
router. Match your question against the headings below; open the
listed card.

## "What changed between generation X and Y?"

Open the per-generation cards in sequence:

- `arch-tesla.md` → `arch-fermi.md` → `arch-kepler.md` →
  `arch-maxwell.md` → `arch-pascal.md` → `arch-volta.md` →
  `arch-turing.md` → `arch-ampere.md` → `arch-ada.md` →
  `arch-hopper.md` → `arch-blackwell.md`.

Each card's "Notes" section frames the diff vs the prior generation.

## "Why does feature X exist?"

| Feature | Card |
|---|---|
| Cache modifiers (`.cg/.cs/.cv/.wt`) | `arch-fermi.md` |
| L2 partition / `L2 slice` / `FB DRAM interface` | `arch-fermi.md` |
| Software scoreboard (`DEPBAR`) | `arch-maxwell.md` |
| Convergence barrier (`BSSY/BSYNC`) | `arch-maxwell.md` (mechanism) + `arch-volta.md` (default) |
| Per-thread PC / Independent Thread Scheduling | `arch-volta.md` |
| `__syncwarp()` / `_sync` family | `arch-volta.md` |
| Tensor Core (HMMA) | `arch-volta.md` |
| INT8 / INT4 / Binary Tensor Core | `arch-turing.md` |
| Uniform-path SASS instructions (`U*`) | `arch-turing.md` + `sass-evolution-timeline.md` |
| `cp.async` | `arch-ampere.md` |
| BF16 / TF32 | `arch-ampere.md` + `evolution-precision.md` |
| Structured 2:4 sparsity | `arch-ampere.md` |
| MIG (Multi-Instance GPU) | `arch-ampere.md` |
| Thread Block Cluster | `arch-hopper.md` + `.claude/knowledge/cuda/cuda-clusters-dsmem.md` |
| Distributed Shared Memory (DSMEM) | `arch-hopper.md` |
| TMA / `cp.async.bulk.tensor` | `arch-hopper.md` + `evolution-data-supply.md` |
| Transaction barrier / `mbarrier.complete_tx` | `arch-hopper.md` + `evolution-synchronization.md` |
| WGMMA | `arch-hopper.md` |
| FP8 (E4M3 / E5M2) | `arch-hopper.md` (datacenter) + `arch-ada.md` (consumer) + `evolution-precision.md` |
| Transformer Engine | `arch-hopper.md` |
| Memory synchronization domains | `arch-hopper.md` + `evolution-memory-side.md` |
| Tensor Memory (TMEM) | `arch-blackwell.md` + `evolution-data-supply.md` |
| `tcgen05.mma` | `arch-blackwell.md` + `evolution-execution-model.md` |
| FP4 / FP6 / NVFP4 / MX | `arch-blackwell.md` + `evolution-precision.md` |
| Dual-die single GPU (B200) | `arch-blackwell.md` + `evolution-interconnect.md` |
| NVL72 | `arch-blackwell.md` + `evolution-interconnect.md` |

## "Which patent introduced X?"

Open `patent-family-map.md`. It has both an alphabetical patent
index and a by-generation lookup table.

## "Was this a major architectural breakpoint?"

Open `three-rewrites.md`. The three rewrites are:

- Fermi (memory side becomes a hardware domain)
- Volta (matrix path peels out + ITS default)
- Hopper (cooperation boundary crosses the SM)

Everything else is consolidation.

## "What's the difference between `bar.sync` / `__syncthreads` / `mbarrier` / `mbarrier.complete_tx` / `cluster.sync`?"

Open `evolution-synchronization.md`. Quick reference:

| Primitive | Generation | Scope | What is being waited on |
|---|---|---|---|
| `bar.sync` / `__syncthreads()` | Tesla+ | Thread block | All threads of the CTA |
| `__syncwarp(mask)` | Volta+ | Warp (or subset) | Masked threads of the warp at this PC |
| `mbarrier` | Ampere+ | Software-defined thread set | Arrival count target reached |
| `mbarrier.complete_tx` (transaction barrier) | Hopper+ | Software set + async transactions | Arrivals AND byte-count target both met (used with TMA) |
| `barrier.cluster.{arrive,wait,sync}` / `cluster.sync()` | Hopper+ | All blocks in a cluster | All blocks resident; required before DSMEM peer access |
| `wgmma.commit_group` / `wgmma.wait_group` | Hopper | Warpgroup | All `wgmma.mma_async` instructions in the group complete |
| `tcgen05.commit` / `tcgen05.wait` | Blackwell | CTA / CTA-pair | TMEM-resident matrix operation complete |

Pre-Volta `bar.sync` was sufficient because warps were lockstep. ITS
broke that → `__syncwarp` for warp-level convergence. Async copies
broke "all threads arrived = all data arrived" → `mbarrier` +
transaction count. Cooperation crossing the SM broke "block-local
sync = full coordination" → `cluster.sync()`. WGMMA / tcgen05 add
async-MMA-completion as a separate object.

## "How did synchronization evolve?"

Open `evolution-synchronization.md`. The line is:

```
register dependencies (Tesla)
  -> software scoreboard (Maxwell)
  -> convergence barrier (Maxwell mechanism, Volta default)
  -> __syncwarp / _sync family (Volta surface)
  -> mbarrier (Ampere)
  -> transaction barrier / cluster barrier (Hopper)
```

## "How did data supply evolve?"

Open `evolution-data-supply.md`. The line is:

```
collector unit (Tesla)
  -> RF/crossbar/collector loop (Kepler)
  -> HMMA datapath (Volta)
  -> cp.async (Ampere)
  -> TMA (Hopper)
  -> TMEM (Blackwell)
```

## "How did the memory side evolve?"

Open `evolution-memory-side.md`. The line is:

```
background routing (Tesla)
  -> tunable hardware domain with cache modifiers (Fermi)
  -> dirty-line backing-store control (Fermi continuation)
  -> distributed shared memory + transaction barrier + sync domains (Hopper)
  -> Tensor Memory (Blackwell)
```

## "How did execution model evolve?"

Open `evolution-execution-model.md`. The line is:

```
warp lockstep, one PC per warp (Tesla)
  -> per-thread PC + ITS (Volta)
  -> Thread Block Cluster (Hopper)
  -> single-thread issue for matrix initiation (Blackwell tcgen05.mma)
```

## "How did precision evolve?"

Open `evolution-precision.md`. The line is:

```
FP16 in FPU (Pascal)
  -> FP16 + FP32 accumulate Tensor Core (Volta)
  -> INT8 / INT4 / Binary (Turing)
  -> BF16 / TF32 / FP64 Tensor Core / 2:4 sparsity (Ampere)
  -> FP8 E4M3 / E5M2 + Transformer Engine (Hopper)
  -> FP4 / FP6 / NVFP4 / MX block-scaled MMA (Blackwell)
```

## "How did interconnect evolve?"

Open `evolution-interconnect.md`. The line is:

```
on-chip crossbar with virtual channels (Tesla)
  -> NVLink 1.0 (Pascal)
  -> NVLink 2.0 (Volta)
  -> NVLink 3.0 + NVSwitch 1 (Ampere)
  -> NVLink 4.0 + NVSwitch 2 + SHARP (Hopper)
  -> NVLink 5.0 + NV-HBI (in-package die-to-die) + NVL72 (Blackwell)
```

## "Did Microsoft / Intel / Meta / Micron / AMD do something similar?"

Open `competitor-tensor-storage.md`. The corpus tracks tensor-near
storage parallels in 5 non-NVIDIA patents.

## "Which generation does opcode X belong to?"

Open `sass-evolution-timeline.md`. It has per-generation SASS deltas
and a quick-reference table mapping opcodes to minimum compute
capability.

## "What is the corpus's overarching thesis?"

Open `three-rewrites.md`. One paragraph: nearly two decades of
NVIDIA evolution are **three rewrites** at three different levels of
the machine, separated by long consolidation phases. Fermi rewrites
the memory side; Volta rewrites the execution path and execution
model; Hopper rewrites the cooperation boundary.

## "I'm reading a CUDA-version-N feature; where does it fit?"

| CUDA version | First-supported architecture | Key card |
|---|---|---|
| CUDA 1.x–3.x | Tesla / Fermi | `arch-tesla.md` / `arch-fermi.md` |
| CUDA 5.x | Kepler | `arch-kepler.md` |
| CUDA 6.x | Maxwell | `arch-maxwell.md` |
| CUDA 8.x | Pascal | `arch-pascal.md` |
| CUDA 9.x | Volta | `arch-volta.md` |
| CUDA 10.x | Turing | `arch-turing.md` |
| CUDA 11.x | Ampere | `arch-ampere.md` |
| CUDA 11.8 / 12.x | Ada | `arch-ada.md` |
| CUDA 12.x | Hopper | `arch-hopper.md` |
| CUDA 12.8 / 13.x | Blackwell | `arch-blackwell.md` |

## "I'm reading historical simulator context; what objects mattered?"

Per generation, the historically load-bearing simulator objects:

| Generation | Must model |
|---|---|
| Tesla | Warp PC, busy-bit scoreboard, single SM cooperation scope, banked RF + collector |
| Fermi | + L2 slice / partition / FB-DRAM, cache modifiers, write-back L2 |
| Kepler | + RF/crossbar/collector loop, FP64 unit, Hyper-Q |
| Maxwell | + Software scoreboard, convergence-barrier objects, write-through L2 |
| Pascal | + FP16 in FPU, NVLink 1.0, Unified Memory |
| Volta | + HMMA datapath, per-thread PC, _sync mask, Tensor Core (1st gen) |
| Turing | + INT8 / INT4 / Binary MMA, uniform datapath, RT Core (out of corpus scope) |
| Ampere | + cp.async (`LDGSTS`), BF16 / TF32, structured 2:4 sparsity, MIG partitions |
| Ada | + Consumer FP8 (no cluster / DSMEM / TMA / WGMMA) |
| Hopper | + Thread Block Cluster, DSMEM address mapping, TMA descriptor, transaction barrier, WGMMA, memory sync domains |
| Blackwell | + TMEM as ISA-visible storage, tcgen05.mma single-thread issue, FP4 / NVFP4 / MX block-scaled MMA, NV-HBI die-to-die |

## "I want to teach someone the corpus; what's the reading order?"

1. `INDEX.md` — corpus map.
2. `three-rewrites.md` — the thesis.
3. `patent-family-map.md` — the cast.
4. `evolution-execution-model.md` and `evolution-data-supply.md` —
   the two technical axes.
5. Per-generation cards in chronological order.
6. The remaining evolution-* cards as needed.
7. `competitor-tensor-storage.md` for the IP landscape.
8. Return to `INDEX.md` as a lookup table.

## See also

- `INDEX.md` — topic-oriented router (the canonical entry point).
- `three-rewrites.md` — corpus thesis.
- `patent-family-map.md` — patent index.

## Escalate to raw when

- A question doesn't fit any of these problem types — read
  NVIDIA_patentd.md directly. The translated source is always
  authoritative.
