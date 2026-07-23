---
Use when: looking up a patent ID by name; resolving "which generation does patent X anchor"; verifying priority/filing dates; mapping NVIDIA patents vs the parallel competitor patents (Microsoft / Intel / Meta / Micron / AMD); answering "is patent X already in our corpus?"
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2 (per-section patent citations) and §2.6 (competitor patents).
Supporting sources: per-arch cards (`arch-tesla.md` through `arch-blackwell.md`); each card lists its anchored patents in the frontmatter.
Patents anchored here: every patent in the corpus.
Out of scope: full patent claim analysis (use Google Patents); patents outside this corpus.
Version family: corpus snapshot 2026-05-02. Verified against Google Patents.
---

# Patent Family Map — corpus master index

Master list of every patent cited in the corpus. Every entry has been
verified against Google Patents (priority date and title).

## NVIDIA patents (in priority order)

### `US7339592B2` — multi-port RF emulation
- **Title**: "Simulating multiported memories using lower port count
  memories."
- **Filed / priority**: 2004-07-13.
- **Inventors**: Lindholm, Liu, Moy, Nickolls, Siu.
- **What it claims**: simulate a multi-ported memory using lower-port
  banks; a **collector unit** per bank gathers source operands across
  cycles and forwards to an execution unit when complete.
- **Anchored generation**: Tesla.
- **Card**: `arch-tesla.md`.

### `US7490208B1` — compact multi-ported register file
- **Title**: "Architecture for compact multi-ported register file."
- **Filed / priority**: 2004-10-05.
- **What it claims**: register file from single-port RAM with
  phase-multiplexed access; N read + M write serialized over (N+M)
  clock phases; ≥50% area reduction vs true multi-ported memory.
- **Anchored generation**: Tesla.
- **Card**: `arch-tesla.md`.

### `US8539130B2` — virtual channels for crossbar
- **Title**: "Virtual channels for effective packet transfer."
- **Priority**: 2009-09-24, **filed**: 2010-08-31.
- **What it claims**: crossbar with multiple virtual channels;
  arbitration logic selects a VC per request based on source-client
  type and request type; high-priority VCs avoid deadlock.
- **Anchored generation**: Tesla (formalization), described as
  evidence that on-chip traffic organization was a foundational
  problem from very early.
- **Card**: `arch-tesla.md`.

### `US9639479B2` — parallel cache hierarchy (THE FERMI REWRITE)
- **Title**: "Instructions for managing a parallel cache hierarchy."
- **Priority**: 2009-09-23, **filed**: 2010-09-22.
- **What it claims**: bundles `partition unit` + `L2 slice` + `FB DRAM
  interface` + `.cg/.cs/.cv/.wt` cache-policy modifiers into one
  hardware boundary; software can express access intent on global
  load/store.
- **Anchored generation**: **Fermi (rewrite #1)**.
- **Card**: `arch-fermi.md`, `three-rewrites.md`.

### `US9639466B2` — fine-tuned cache-to-backing-store sync
- **Title**: "Control mechanism for fine-tuned cache to backing-store
  synchronization."
- **Filed / priority**: 2012-10-30.
- **What it claims**: when a write marks a line dirty, evaluate
  whether dirty-line count crosses a configurable threshold; suppress
  or transmit a dirty-data notification to FB logic accordingly.
- **Anchored generation**: Fermi continuation (memory-side control
  refinement).
- **Card**: `arch-fermi.md`.

### `US9612836B2` — software-based scoreboarding
- **Title**: "System, method, and computer program product for
  implementing software-based scoreboarding."
- **Filed / priority**: 2014-02-03.
- **Inventors**: Ohannessian Jr., Fetterman, Giroux, Choquette, Qiu,
  Gadre, Vishnu.
- **What it claims**: dependency-barrier instruction with immediate
  + register-id; comparison gates dispatch; 6-bit scoreboard
  registers; moves scoreboarding from hardware-implicit to
  software/compiler-explicit.
- **Anchored generation**: Maxwell.
- **Card**: `arch-maxwell.md`.

### `US10067768B2` — convergence barrier for divergent threads
- **Title**: "Execution of divergent threads using a convergence
  barrier."
- **Earliest priority**: 2014-07-18, **filed**: 2015-07-13.
- **Inventors**: Choquette, Tirumala.
- **What it claims**: divergence handled via barrier objects (not
  stack/priority); compiler inserts BMOV/BSSY/BSYNC/BREAK; YIELD
  mechanism guarantees forward progress; divergence management
  decoupled from scheduling.
- **Anchored generation**: Maxwell (mechanism) → Volta (default).
- **Cards**: `arch-maxwell.md`, `arch-volta.md`.

### `US10338919B2` — generalized acceleration of MMA (THE VOLTA REWRITE)
- **Title**: "Generalized acceleration of matrix multiply accumulate
  operations."
- **Priority**: 2017-05-08, **filed**: 2017-11-29.
- **Inventors**: Siu, Choquette, Alben.
- **What it claims**: MMA datapath with `RF banks → crossbar → operand
  collectors → HMMA datapath → result queue → RF writeback`; produces
  multiple result-matrix elements per pass; collectors stage operands
  for reuse across dot-product passes.
- **Anchored generation**: **Volta (rewrite #2)**.
- **Card**: `arch-volta.md`, `three-rewrites.md`.

### `US20230289189A1` — Distributed Shared Memory (THE HOPPER REWRITE)
- **Title**: "Distributed Shared Memory."
- **Filed / priority**: 2022-03-10.
- **Inventors** (sample): Bangalore Prabhakar, Hirota, Krashinsky,
  Long, Pharris, Dash, Tuckey, Duluk Jr., Shah, Durant, Choquette,
  Werness, Govil, Patel, Deb, Navada, Edmondson, Palmer, Gandhi,
  Manyam, Parle, Giroux, Gadre, Heinrich.
- **What it claims**: blocks of memory distributed across processing
  cores; threads on one core can access the memory block local to
  another core; new programming and hardware execution model for
  shared-memory cooperation at scopes larger than a single SM.
- **Anchored generation**: **Hopper (rewrite #3)**.
- **Card**: `arch-hopper.md`, `three-rewrites.md`.

### `US12141082B2` — TMA (Tensor Memory Access Unit)
- **Title**: "Method and apparatus for efficient access to
  multidimensional data structures and/or other large data blocks."
- **Filed / priority**: 2022-03-10.
- **What it claims**: a Tensor Memory Access Unit (TMAU) closely
  coupled to streaming multiprocessors; offloads data movement from
  cores; accepts kilobyte/megabyte data-block requests via a single
  instruction; converts multidimensional coordinates to memory
  addresses; handles transfers between global and shared memory
  asynchronously.
- **External naming**: TMA. Internal: TMAU.
- **Anchored generation**: Hopper.
- **Card**: `arch-hopper.md`.

### `US20230289242A1` — transaction barrier (mbarrier)
- **Title**: "Hardware accelerated synchronization with asynchronous
  transaction support."
- **Filed / priority**: 2022-03-10.
- **What it claims**: a transaction barrier extends or replaces the
  shared-memory-backed arrive-wait barrier; receives arrival updates
  from CTAs in the same CGA running on different SMs; tracks
  transaction count separately from thread-arrival count.
- **PTX surface**: `mbarrier.complete_tx`, `mbarrier.arrive.expect_tx`.
- **Anchored generation**: Hopper.
- **Card**: `arch-hopper.md`.

## Competitor patents (NVIDIA_patentd.md §2.6)

These are **non-NVIDIA** patents that show how other companies
elevate "tensor-near storage" / "tensor-aware buffer" / "tile
register" as an explicit object — paralleling NVIDIA's TMEM but under
different names.

### `US10372456B2` — Microsoft, tensor processor ISA
- **Title**: "Tensor processor instruction set architecture."
- **Earliest priority**: 2017-05-24.
- **Assignee**: Microsoft Technology Licensing LLC.
- **What it claims**: tensor memory manager + input queue + tensor
  register file divided by operator type; different tensor calculators
  have their own dedicated tensor register files; tensors routed by
  operation type into near-end register storage.
- **Card**: `competitor-tensor-storage.md`.

### `EP4020173B1` — Intel, matrix-operations accelerator tile load/store
- **Title**: "Apparatuses, methods, and systems for instructions for
  loading a tile of a matrix operations accelerator."
- **Earliest priority**: 2020-12-24.
- **Assignee**: Intel Corp.
- **What it claims**: ISA-level objects for `data buffers` /
  `tile registers` of a matrix operations accelerator with
  `load-to-tile` / `store-from-tile` direct paths to/from vector
  registers.
- **CPU-side parallel** to Blackwell's `LDT/LDTM`/`STT/STTM`.
- **Card**: `competitor-tensor-storage.md`.

### `US11922306B2` — Meta, tensor controller architecture
- **Title**: "Tensor controller architecture."
- **Filed / priority**: 2020-12-28.
- **Assignee**: Meta Platforms Inc.
- **What it claims** (central novelty): a counter-stack
  controller; each controller traverses feature maps across n
  dimensions via dimension-specific counters; dependency controller
  manages counter incrementation based on token availability.
  Embodiment-level, the patent additionally chains multi-bank
  activation memory + weight buffer + output buffer + tensor
  controller into a local data-orchestration pipeline.
- **Card**: `competitor-tensor-storage.md`.

### `US10956315B2` / `US12229044B2` — Micron, tensor-friendly memory devices
- **Title**: "Memory devices and methods which may facilitate tensor
  memory access."
- **Earliest priority**: 2018-07-24.
- **Assignee**: Micron Technology Inc.
- **What it claims**: memory controller with a data-address generator
  that produces sequences of memory addresses based on starting
  address and tensor/matrix dimensions; buffer for reading and
  writing data per generated sequences.
- **Approach**: cuts in from the **memory side** rather than from
  near the compute datapath.
- **Card**: `competitor-tensor-storage.md`.

### `US11609785B2` — AMD, matrix data broadcast architecture
- **Title**: "Matrix data broadcast architecture."
- **Earliest priority**: 2019-12-23.
- **Assignee**: Advanced Micro Devices Inc.
- **What it claims**: parallel processor core supports broadcast of
  shared data to multiple compute units rather than separate memory
  requests per CU; compiler identifies thread groups sharing data and
  emits a single access request listing target CUs.
- **Approach**: tensor-near reuse without an explicit "tensor
  register file."
- **Card**: `competitor-tensor-storage.md`.

## Lookup tables

### By NVIDIA generation

| Generation | Anchoring patents |
|---|---|
| Tesla | US7339592B2, US7490208B1, US8539130B2 |
| Fermi | US9639479B2, US9639466B2 |
| Kepler | (no new anchoring patent — builds on Tesla collector) |
| Maxwell | US9612836B2, US10067768B2 |
| Pascal | (no new anchoring patent — adds FP16 + NVLink + Unified Memory) |
| Volta | US10338919B2 (HMMA datapath); US10067768B2 reapplied as default model |
| Turing | (no new anchoring patent — extends Volta HMMA precisions) |
| Ampere | (no NVIDIA patent in corpus; "shard scheduling" cited as patent-side inference) |
| Ada | (no anchoring patent — consumer Hopper-era) |
| Hopper | US20230289189A1 (DSM), US12141082B2 (TMA), US20230289242A1 (transaction barrier) |
| Blackwell | (extends Hopper patents; TMEM is SASS-visible but not anchored to a single corpus patent) |

### By patent ID (alphabetical)

| Patent | Card |
|---|---|
| EP4020173B1 (Intel) | `competitor-tensor-storage.md` |
| US7339592B2 | `arch-tesla.md` |
| US7490208B1 | `arch-tesla.md` |
| US8539130B2 | `arch-tesla.md` |
| US9612836B2 | `arch-maxwell.md` |
| US9639466B2 | `arch-fermi.md` |
| US9639479B2 | `arch-fermi.md`, `three-rewrites.md` |
| US10067768B2 | `arch-maxwell.md`, `arch-volta.md` |
| US10338919B2 | `arch-volta.md`, `three-rewrites.md` |
| US10372456B2 (Microsoft) | `competitor-tensor-storage.md` |
| US10956315B2 (Micron) | `competitor-tensor-storage.md` |
| US11609785B2 (AMD) | `competitor-tensor-storage.md` |
| US11922306B2 (Meta) | `competitor-tensor-storage.md` |
| US12141082B2 | `arch-hopper.md` |
| US12229044B2 (Micron) | `competitor-tensor-storage.md` |
| US20230289189A1 | `arch-hopper.md`, `three-rewrites.md` |
| US20230289242A1 | `arch-hopper.md` |

### By assignee

| Assignee | Patents |
|---|---|
| NVIDIA | US7339592B2, US7490208B1, US8539130B2, US9612836B2, US9639466B2, US9639479B2, US10067768B2, US10338919B2, US12141082B2, US20230289189A1, US20230289242A1 |
| Microsoft | US10372456B2 |
| Intel | EP4020173B1 |
| Meta | US11922306B2 |
| Micron | US10956315B2, US12229044B2 |
| AMD | US11609785B2 |

## Verification status

All 17 patents above were verified against Google Patents on
2026-05-02 (titles, priority dates, filing dates, assignees). The
verification is captured in the previous translation pass; no
changes are expected unless a future agent finds a discrepancy.

## What this corpus does NOT include

- NVIDIA's RT Core patents (parallel graphics line; not in
  NVIDIA_patentd.md).
- DLSS / OFA patents.
- NVLink PHY-level patents.
- Cache-modifier extensions patents post-Fermi (e.g., L2 persistence).
- Independent Thread Scheduling patents beyond US10067768B2 (the
  corpus treats it as the load-bearing one).
- Cluster scheduler patents — only DSM is in the corpus; NVIDIA likely
  has cluster-scheduler-specific patents that are not cited.

If a future agent encounters a relevant patent not on this list,
add it to:

1. The corresponding `arch-*.md` card's frontmatter.
2. This card's lookup tables.

## See also

- `three-rewrites.md` — the three patents that anchor the rewrite
  thesis.
- All `arch-*.md` cards — each lists anchored patents in frontmatter.
- `competitor-tensor-storage.md` — non-NVIDIA patents.
- `sass-evolution-timeline.md` — opcodes that anchored patents
  surface as SASS instructions.
- `glossary.md` — one-line term definitions.
- `INDEX.md` — corpus router.
- `problem-playbook.md` — question-driven router.

## Escalate to raw when

- You need claim-level analysis of any patent — open Google Patents
  with the patent ID. The "What it claims" lines here are condensed.
- You need the full inventor list — the corpus only records inventors
  for the most-cited patents.
- You need related-family patents (continuations, divisionals) —
  Google Patents shows the family tree.
