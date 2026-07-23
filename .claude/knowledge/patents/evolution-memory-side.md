---
Use when: explaining how the memory-side hardware domain develops; placing L2 / partition / cache modifiers / atomic ordering / completion visibility on one timeline; reasoning about why memory synchronization domains exist.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.2 (Fermi rewrite) and the cross-references through §2.4 (Hopper extensions).
Supporting sources: per-arch cards (Fermi, Volta, Hopper, Blackwell); references/NVIDIA_Manuals/cuda_guide/04-13_l2-cache-control.md (plus the 04-13-01 through 04-13-08 sub-files for set-aside, access policies, persistence example, query/control APIs); references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md.
Patents anchored here: US9639479B2 (parallel cache hierarchy), US9639466B2 (cache-to-backing-store sync), US20230289242A1 (transaction barrier).
Out of scope: data supply (`evolution-data-supply.md`); execution model (`evolution-execution-model.md`).
Version family: Fermi through Blackwell. Mirror snapshot: 2026-05.
---

# Memory Side — passive cache → tunable domain → DSM → TMEM

The memory-side hardware domain (L2 / partition / DRAM) doesn't just
get bigger across generations. It **gains semantics**. This card
walks the timeline from background routing to first-class object.

## Tesla — memory side as background

Pre-Fermi, the memory side has no software-visible semantics:

- No L1 / L2 data cache (only texture caches).
- No cache-policy modifiers.
- Atomic ordering / write-back paths fuzzy at higher layers.
- Per-address ownership undefined: an SM that writes a global address
  has no specific commitment about where that write lives in the
  hierarchy.

The memory side is just "where addresses end up after the LSU."

## Fermi — memory side becomes a hardware domain (US9639479B2)

The first rewrite of the corpus. **Priority 2009-09-23, filed
2010-09-22.** The patent's importance is **fixing four things into
one set of hardware boundaries**:

1. `partition unit` — a per-address-range processing domain.
2. `L2 slice` — the cache slice inside a partition.
3. `FB DRAM interface` — the boundary out to external DRAM.
4. `.cg / .cs / .cv / .wt` cache-policy modifiers — software-side
   intent expression.

What this enables:

- A stable per-address landing point. Same global address always
  routes to the same L2 slice / partition.
- Cache modifiers let software express **intent** (cache aggressively,
  stream, weaken local copies, write-through) without changing
  consistency behavior.
- Atomic ordering can be handled at the L2 slice for the relevant
  address.
- Write-back and completion paths have a common location.

**Critical caveat (still true in modern PTX)**: cache modifiers are
**performance hints only**. They affect cache path, eviction tendency,
fetch location — they do **not** alter memory consistency. True
synchronization is via the CUDA/PTX memory model, fences, atomics,
barriers.

For details: `arch-fermi.md`, `three-rewrites.md`.

## Fermi continuation — fine-tuned dirty-line control (US9639466B2)

**Filed 2012-10-30.** Adds machinery for fine-grained control over
when L2 dirty data becomes visible to the backing store / DRAM —
cleanup and backpressure trigger only when dirty-line counts cross a
configurable threshold.

This is the early form of what later generations call "memory
synchronization domains" — a sense that you can target specific
classes of memory traffic instead of forcing the whole memory world
to sync.

## Kepler / Maxwell / Pascal — consolidation, no rewrite

Memory side stays Fermi-style. L2 grows in capacity (Fermi 768 KB →
Kepler 1.5 MB → Maxwell 2 MB → Pascal 4 MB GP100). Memory technology
moves (GDDR3 → GDDR5 → GDDR5X → HBM2). But the **semantic boundary**
established by Fermi doesn't move.

L2 changes from write-back (Fermi/Kepler) to write-through (Maxwell/
Pascal/Volta/Turing) — but that's an implementation choice, not a
boundary change.

## Volta / Turing / Ampere / Ada — large-scale L2 + ECC

L2 grows substantially:

- Volta: 6 MB
- Turing TU102: 5.5 MB
- A100: 40 MB (huge jump for datacenter)
- GA102: 6 MB (consumer)
- AD102: 96 MB (16× consumer Ampere)

All Fermi-style semantics. Cache modifiers carry forward. PTX 8.1+ adds
`.lu` (last-use hint) and L2 set-aside / persistence APIs that descend
from the Fermi boundaries — they extend the modifier set, not change
the architecture.

## Hopper — transaction barrier extends memory-side semantics

`US20230289242A1` (filed 2022-03-10) introduces the **transaction-aware
barrier**: extends shared-memory-backed barriers with a transaction
count separate from thread-arrival count.

Why: in the TMA + WGMMA producer-consumer chain, "have all threads
arrived?" and "has all data they were supposed to produce arrived?"
are different questions. The transaction barrier answers both.

PTX: `mbarrier.arrive.expect_tx`, `mbarrier.complete_tx::bytes`.

This is the memory-side semantics line continuing — Fermi gave the
memory side a stable per-address domain; Hopper gives it explicit
**event arrival** semantics that match the new async-transport
primitives.

For details: `arch-hopper.md`, `evolution-synchronization.md`.

## Hopper — Distributed Shared Memory (US20230289189A1)

DSM is also a memory-side semantic extension. Pre-Hopper, shared
memory has no cross-block visibility. Hopper extends shared memory's
effective scope: blocks within a cluster can address each other's SMEM.

This is the cooperation-boundary expansion that makes Hopper rewrite
#3, but it's read here as a **memory-side** change too: shared memory
gains a new visibility class (intra-cluster).

For details: `arch-hopper.md`.

## Hopper — memory synchronization domains

Hopper formalizes the idea that fences/barriers should target
**specific** classes of memory effects and observers — not the whole
memory world. This idea was already present in Fermi's L2 dirty-line
patent (US9639466B2) but is now first-class in PTX.

Specific scopes (PTX 8.0+ on sm_90+):

- `cta` scope — within a thread block.
- `cluster` scope — within a thread block cluster.
- `gpu` scope — within the GPU.
- `sys` scope — system-wide.

Plus **memory_sync_domain** annotations that reduce fence cost when
only a subset of traffic actually needs to drain.

## Blackwell — TMEM as a memory-side object

The newest extension: Tensor Memory (TMEM) is **on-SM dedicated tensor
storage** that's first-class in the ISA.

While TMEM is data-supply (`evolution-data-supply.md`), it also pushes
the memory hierarchy itself: now there are **four** levels visible
to ISA on Blackwell —

```text
RF (per-thread)
  -> SMEM (per-block, 228 KB on B200)
  -> TMEM (per-CTA / dedicated tensor storage, 256 KB on B200)
  -> DSMEM (cluster scope, distributed across SMs)
  -> L2 (chip scope)
  -> DRAM (HBM3e, board scope)
```

TMEM-aware instructions (`LDT/LDTM`, `STT/STTM`, `UTCCP`, `UTCSHIFT`,
`tcgen05.mma`) treat it as a named hardware domain — exactly the
treatment Fermi gave to L2 / partition.

## What stays constant from Fermi onward

- **Per-address stability**: same global address always lands in the
  same processing domain (the L2 slice it maps to). Software can rely
  on this.
- **Cache modifiers as hints**: never alter consistency.
- **Memory-model carries the consistency contract**: fences, atomics,
  scopes are the load-bearing primitives.

## Why a future agent cares

- "Use `.cg` to synchronize" is wrong. Always wrong. Cache modifiers
  are hints — synchronization is via the memory model.
- A workload depending on stable L2 partitioning of an address
  requires **Fermi+** semantics. This is essentially every modern GPU.
- Memory synchronization domains are **sm_90+**. Pre-Hopper code that
  wants per-domain fences will need a different approach.
- DSMEM access of another block's SMEM requires **Hopper+** and a
  preceding `cluster.sync()`. Lifetime: peers must not access exited
  blocks' SMEM.
- TMEM is **Blackwell-only**. It's a real on-SM hardware location,
  not a virtual construct.

## Reading the line as one story

```
Tesla:     memory side as background routing
Fermi:     memory side as a hardware domain (cache modifiers, partition+L2 slice+FB)
Volta+:    L2 grows; semantics unchanged
Hopper:    add transaction-aware barriers + memory sync domains + DSMEM scope
Blackwell: add TMEM as an ISA-visible storage object
```

Three steps of semantic enrichment, separated by long capacity-growth
phases.

## See also

- `three-rewrites.md` — Fermi as rewrite #1.
- `arch-fermi.md` — the rewrite in detail.
- `arch-hopper.md` — DSM + transaction barrier + sync domains.
- `arch-blackwell.md` — TMEM as ISA object.
- `evolution-data-supply.md` — TMEM also lives in this story.
- `evolution-synchronization.md` — transaction barrier in the
  sync-primitive timeline.
- `.claude/knowledge/ptx/ptx-isa-cache-hints.md` — current
  PTX cache modifiers.
- `.claude/knowledge/cuda/cuda-sync-memory-ordering.md` —
  CUDA / PTX memory model.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need the formal PTX memory consistency axioms — see
  `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md`.
- You need L2 cache-control CUDA APIs (set-aside, access policy
  windows, persistence) — see CUDA Programming Guide §4.13.
- You need claim-level reading of the cache-hierarchy or transaction-
  barrier patents — Google Patents US9639479B2 / US20230289242A1.
- You need memory-synchronization-domain examples — PTX
  09_instruction-set.md (search for `memory_sync_domain`).
