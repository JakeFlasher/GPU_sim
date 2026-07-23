---
Use when: framing TMEM as part of an industry-wide trend; comparing how Microsoft / Intel / Meta / Micron / AMD address tensor-near-storage pressure; explaining why "TMEM" is NVIDIA-specific naming for an idea that has many parallel implementations; reasoning about IP landscape for tensor-class accelerators.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.6.
Supporting sources: each patent's Google Patents listing.
Patents anchored here: US10372456B2 (Microsoft), EP4020173B1 (Intel), US11922306B2 (Meta), US10956315B2 / US12229044B2 (Micron family), US11609785B2 (AMD).
Out of scope: NVIDIA's TMEM (covered in `arch-blackwell.md`); patents that aren't about tensor-near storage.
Version family: 2017–2020 priorities for the cited patents. Mirror snapshot: 2026-05.
---

# Competitor Tensor-Near Storage — same direction, different names

NVIDIA's `TMEM` (`Tensor Memory`) on Blackwell is a specific naming.
But the underlying pressure — **fix tensor tiles / intermediate
results / high-reuse operands in a dedicated storage layer near the
operator** — is industry-wide. Every major accelerator vendor has
patented their answer to this question, with different names.

This card maps the parallels.

## The unifying pressure

Once matrix / tensor operators become the main path, the traditional
hierarchy of:

```
RF (per-thread)
  -> SMEM/L1 (per-block)
  -> L2 (chip)
```

is often **insufficient** to simultaneously carry:

- **Throughput**: matrix tile reuse needs many reads from a small
  staging area.
- **Reuse**: same tile consumed across many sub-operations.
- **Orchestration**: scale factors, metadata, partial sums, weights,
  activations all need explicit life-cycle handling.

So each vendor elevates a **tensor-local storage** abstraction into an
explicit object. The names differ but the architectural function is
similar.

## The five competitor patents

### `US10372456B2` — Microsoft, "Tensor processor instruction set architecture"

- **Earliest priority**: 2017-05-24.
- **Assignee**: Microsoft Technology Licensing LLC.
- **What's named explicitly**:
  - `tensor memory manager`
  - input queue
  - **tensor register file** (divided by operator type)
- **Key idea**: different tensor calculators have **their own
  dedicated tensor register files** alongside them. Tensors are
  routed by operation type into near-end register storage instead
  of all repeatedly forwarded through one unified large store.
- **Most analogous to TMEM**: yes — explicit per-operator-type
  tensor-near register files. The closest naming match.
- **Differences from NVIDIA's TMEM**:
  - Per-operator-type segmentation; TMEM is unified per-CTA.
  - "Tensor memory manager" makes the routing explicit; TMEM uses
    address descriptors.

### `EP4020173B1` — Intel, "Loading a tile of a matrix operations accelerator"

- **Earliest priority**: 2020-12-24.
- **Assignee**: Intel Corp.
- **What's named explicitly**:
  - `data buffers` / `tile registers` of the matrix-operations
    accelerator
  - `load-to-tile` / `store-from-tile` direct paths between tile
    storage and vector registers
- **Key idea**: tile load/store is an **ISA-level object**, not just
  a microarchitectural detail.
- **Closest to Blackwell**: yes — Blackwell's `LDT/LDTM` /
  `STT/STTM` are the GPU-side equivalents of Intel's CPU-side
  load-to-tile / store-from-tile.
- **Note**: Intel's matrix engine is CPU-side (AMX in their
  data-center CPUs), not GPU-side. The architectural pattern carries
  across the CPU/GPU divide.

### `US11922306B2` — Meta, "Tensor controller architecture"

- **Earliest priority / filed**: 2020-12-28.
- **Assignee**: Meta Platforms Inc.
- **Central novelty**: a **counter-stack controller**. Each controller
  traverses feature maps across n dimensions via
  dimension-specific counters. A dependency controller manages
  counter incrementation based on token availability, enabling
  efficient hardware-level memory address generation for ML
  operations.
- **Embodiment level** (mentioned but not the inventive heart):
  multi-bank activation memory + weight buffer (which can be
  implemented as a register file) + output buffer + tensor controller
  chained into a local data-orchestration pipeline.
- **Reading**: Meta's framing is **how to schedule access** to local
  tensor storage, not just declare its existence. The activation /
  weight / output buffer split is the embodiment that maps closest
  to NVIDIA's per-tensor-class storage idea, but the patent's
  inventive heart is on the address-generator side.
- **Closest to**: TMA + TMEM in spirit (descriptor-driven access
  generation + dedicated near-compute storage), though Meta's
  framing is on the controller, not the storage.

### `US10956315B2` / `US12229044B2` — Micron, "Memory devices … tensor memory access"

- **Earliest priority**: 2018-07-24.
- **Assignee**: Micron Technology Inc.
- **What's named explicitly**: a memory controller with a
  **data address generator** that produces sequences of memory
  addresses based on starting address and tensor/matrix dimensions
  (row, column, diagonal, determinant, or N-th dimension). A
  **buffer** for reading/writing data per the generated sequences.
- **Key approach**: cuts in from the **memory side** — the memory
  controller itself is tensor-aware. Frequently reused matrix data
  can be re-read from the buffer instead of repeatedly going back to
  the memory array.
- **Reading**: shows that "tensor-aware memory" can grow first from
  the **memory subsystem side**, not the compute side. NVIDIA's TMA
  + TMEM is compute-near; Micron's approach is memory-near; they
  meet in the middle.
- **Family**: US10956315B2 and US12229044B2 are continuations of the
  same family; same priority.

### `US11609785B2` — AMD, "Matrix data broadcast architecture"

- **Earliest priority**: 2019-12-23.
- **Assignee**: Advanced Micro Devices Inc.
- **What's named explicitly**: a parallel processor core supporting
  **broadcast of shared data** to multiple compute units rather than
  separate memory requests per CU. A compiler identifies thread
  groups sharing data and emits a **single access request** listing
  the target compute units.
- **Reading**: AMD doesn't introduce an explicit "tensor register
  file." Instead it makes broadcast/reuse a **first-class memory-
  access primitive**. Same matrix data is reused multiple times near
  the operator, but the storage stays in the existing
  local-data-store hierarchy.
- **Closest to**: NVIDIA's DSMEM + TMA in concept (one access
  delivers data to many consumers near the operator). The
  architectural lever is different from a dedicated TMEM-style
  storage.

## The pattern when read together

| Vendor | Storage object name | Primary innovation lever |
|---|---|---|
| NVIDIA (Blackwell) | Tensor Memory (TMEM) | Dedicated per-CTA tensor storage; descriptor-driven MMA |
| Microsoft | Tensor register file (per operator type) | Operator-typed near-end register storage |
| Intel | Tile registers + tile load/store | ISA-level CPU-side tile load/store |
| Meta | Multi-bank activation + weight + output + counter-stack controller | Address-generation controller for tensor traversal |
| Micron | Tensor-friendly memory-controller buffer | Memory-side data-buffer for tensor patterns |
| AMD | Broadcast data to multiple CUs | Reuse via broadcast, not dedicated storage |

**Industry consensus is not on one name. It's on one direction**:
once matrix / tensor operators are the main path, the traditional
`RF + SMEM/L1 + L2` is often insufficient. Each vendor elevates
"tensor-local storage" into a more explicit object — under different
names — or addresses the same throughput / reuse / orchestration
problem from a different angle (broadcast, address generation,
memory-controller buffers).

## Why a future agent cares

- "TMEM is unique to NVIDIA" is shallow. The naming is unique; the
  architectural pressure is industry-wide. When framing IP
  landscape, mention the parallels.
- Reading non-NVIDIA accelerator patents: words like "tensor register
  file," "tile registers," "activation/weight/output buffer,"
  "tensor-aware memory controller," and "GEMM broadcast" are pointing
  at the same pressure. Don't read them as unrelated.
- An accelerator design-decision question — "do we need a dedicated
  near-operator tensor storage?" — has at least 6 prior-art answers
  with substantially different IP positions.

## See also

- `arch-blackwell.md` — NVIDIA's TMEM as an ISA object.
- `evolution-data-supply.md` — TMEM in the data-supply timeline.
- `patent-family-map.md` — full patent index with all 5 competitor
  patents.

## Escalate to raw when

- You need claim-level reading of any competitor patent — Google
  Patents is the canonical source.
- You need to compare TMEM to a specific competitor product — NVIDIA
  has not published direct comparisons. Architecture deep-dives by
  third parties (e.g., SemiAnalysis, ChipsAndCheese) sometimes do.
- You need IP-licensing implications — out of scope; consult counsel.
