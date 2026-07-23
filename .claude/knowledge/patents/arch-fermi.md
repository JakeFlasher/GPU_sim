---
Use when: explaining when L2 / partition / cache modifiers became software-visible; reasoning about why CUDA has `.cg/.cs/.cv/.wt` modifiers; placing the first "memory side as a hardware domain" architectural rewrite; understanding Fermi-era GPC / SM organization.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.2.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/1.fermi.md (SM config table); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: US9639479B2 (parallel cache hierarchy / partition+L2+FB+modifiers, priority 2009-09-23, filed 2010-09-22), US9639466B2 (fine-tuned cache-to-backing-store sync, filed 2012-10-30).
Out of scope: Kepler's data-supply loop completion (`arch-kepler.md`); Maxwell's barrier rewrite (`arch-maxwell.md`); the Volta/Hopper rewrites that depend on this foundation.
Version family: Fermi architecture, 2010. Representative product: GF100 (GTX 480, 2010-03). Compute capability 2.x.
---

# Fermi (2010, GF100) — memory-side becomes a real hardware domain

Fermi is **rewrite #1** of the three-rewrites thesis (`three-rewrites.md`).
The breakpoint is not execution, not data supply, not synchronization,
not interconnect — it's the **memory side**. Before Fermi the L2 /
partition / DRAM path is background routing; after Fermi it is a
software-aware tunable hardware domain.

## SM organization (GF100)

| Component | Count |
|---|---|
| Warp Scheduler | 2 |
| Dispatch Unit | 2 |
| INT32+FP32 Core | 32 (in 2 lanes of 16) |
| LD/ST Unit | 16 |
| SFU | 4 |
| Register file / SM | 32 KB |
| Shared memory + L1 (configurable) | 64 KB total |
| Configurations | 48 KB SMEM + 16 KB L1, or 16 KB SMEM + 48 KB L1 |
| L2 cache (whole chip) | 768 KB |
| Memory | GDDR5 |

**Notes for downstream readers:**

- Fermi renames Tesla's TPC layer to **GPC** (Graphics Processing
  Cluster). 4 GPCs in GF100, each with multiple SMs.
- First architecture with **L1 + L2 data caches**.
- L1 and shared memory **share a 64 KB pool**, runtime-configurable.
- L2 is **write-back**.
- LD/ST units are now an independent functional class.
- Address space widens from 32-bit (Tesla) to **64-bit**.
- Register file holds 32,768 32-bit registers per SM.
- Each CUDA core implements **FMA** (fused multiply-add) directly. Tesla
  did not fully follow IEEE-754; Fermi adds full IEEE-754 with all
  rounding modes and subnormals.
- Per-cycle 16 double-precision FMA throughput.

## The architectural rewrite — memory side becomes a domain

Tesla's limit was that the memory side was just "where address requests
land." Fermi promotes it to a software-observable, software-tunable
hardware domain. Three direct pressures drove this:

1. **`global` access semantics fork**: software needs to express which
   accesses cache aggressively, which are streaming, which want to
   weaken local copies and refetch, which want write-through.
2. **L1 cannot be the common owner**: L1 is per-SM-local, so it cannot
   serve as the common processing point when different SMs touch the
   same global address. A common owner closer to L2/partition/DRAM is
   needed — the L2 slice / partition processing domain that the address
   maps to.
3. **Atomics, write-back, residency, sync need a stable per-address
   landing point** so ordering, backpressure, cleanup, and completion
   paths don't sit at a higher level doing fuzzy coordination.

`US9639479B2` is the patent that fixes all of this into one set of
hardware boundaries.

## US9639479B2 — the foundational Fermi patent

- **Priority** 2009-09-23, **filed** 2010-09-22, NVIDIA.
- **Title**: "Instructions for managing a parallel cache hierarchy."
- **What it nails down as hardware boundaries**:
  - `partition unit`
  - `L2 slice` (the per-address-range cache slice inside a partition)
  - `FB DRAM interface` (the boundary out to external DRAM)
  - `.cg / .cs / .cv / .wt` load/store policy modifiers
- **The crucial reading**: it's not just "introduce L2." It's
  consolidating those four things into the same hardware boundary so
  software can describe access intent and the per-address processing
  domain can act on it.

### Cache modifier semantics (PTX-level)

| Modifier | Semantics | Suitable accesses |
|---|---|---|
| `.cg` | Cache at global / L2 level; do not preferentially occupy each SM's L1 | Multiple blocks reusing one parameter table — share via L2, avoid duplicating in per-SM L1 |
| `.cs` | Streaming — even if cached, prefer eviction | Single-use, sequential-scan input (image stream, activations) |
| `.cv` | For system-memory-related accesses, prefer refetch over reusing possibly-stale lines | Reading data the host or external agent may update; pair with proper sync |
| `.wt` | Write-through-style store; push to memory side / system memory | Host-mapped buffers, mailboxes, queue metadata; pair with correct ordering |

**Critical caveat**: cache modifiers are **performance hints only**.
They affect cache path, eviction tendency, fetch location — they do
**not** alter memory consistency. True synchronization and visibility
are still carried by the CUDA/PTX memory model, fences, atomics,
barriers.

## US9639466B2 — finer L2-to-backing-store control

- **Filed** 2012-10-30, NVIDIA.
- **Title**: "Control mechanism for fine-tuned cache to backing-store
  synchronization."
- **Claim**: when a write marks a line dirty, evaluate whether the
  total dirty-line count in a set crosses a configurable threshold;
  suppress or transmit a dirty-data notification to the frame-buffer
  logic accordingly.
- **Why it matters**: this is Fermi's L2 design continuing to mature.
  It answers **when** L2 dirty data becomes visible to the backing
  store and when cleanup/backpressure is triggered, without firing on
  every single dirty line. This is one of the early forms of what
  later becomes "memory synchronization domains" in Hopper.

## What Fermi enables for later generations

The "stable processing domain" that Fermi establishes is the
prerequisite for:

- **Ordered atomics**: the same address stably lands in some L2 slice;
  that slice handles ordering, waiting, execution. No fuzzy global
  coordination at higher levels.
- **Residency control**: which data should preferentially stay in L2
  vs be evicted faster under streaming policy.
- **Memory synchronization domains** (later Hopper feature): a
  fence/barrier targets some class of memory effects and observers,
  not the whole memory world.
- **Coherence / completion visibility**: when a remote write actually
  becomes visible to another observer, even within the coherent
  domain.

All of those questions need a stable per-address landing point first.
That's Fermi's contribution.

## "Stable" — a careful definition

"Stable" here means **stable processing domain**: regardless of which
SM originates the request, the same global address lands in a
consistent memory-side processing domain.

It does **not** guarantee an L2 hit. It does **not** imply data
permanently sits in L2. It does **not** endow cache operators with
synchronization semantics.

Example: SM A writes a completion flag, SM B polls it. The program
must still use the right fence / atomic / barrier / memory-model
mechanism to express visibility. The hardware just needs a common
location to handle "has the write completed, do stale local copies
need weakening, who wins on same-address atomics, when is it safe to
deliver to other observers."

## Why a future agent cares

- Any analysis claiming "use `.cg` to synchronize" is wrong — they're
  performance hints, not consistency primitives. Reroute to the PTX
  memory consistency model card (`.claude/knowledge/cuda/cuda-sync-memory-ordering.md`)
  or PTX 08_memory-consistency-model.md.
- Fermi-era trace replay needs the per-SM L1 + write-back L2 model
  with a per-address-range partitioning. If the simulator
  L2 design doesn't have partition slices, it's pre-Fermi semantics
  and won't match real hardware.
- The CUDA Programming Guide's L2 cache control APIs all descend from
  the boundaries this patent established.

## Kepler / Maxwell as continuation, not breakpoints

Fermi is the rewrite. Kepler and Maxwell are the cleanup phase:

- **Kepler** completes the **`RF -> crossbar -> collector`** data-supply
  loop on the SM side. See `arch-kepler.md`.
- **Maxwell** pushes synchronization from busy-bit toward barrier-style
  (software scoreboard + convergence barrier) — preparing the protocol
  Volta will need. See `arch-maxwell.md`.

Together they close out the Fermi-era machine and set up the
preconditions for Volta's executive-side rewrite.

## See also

- `three-rewrites.md` — Fermi as rewrite #1; Volta and Hopper as #2 and #3.
- `evolution-memory-side.md` — Fermi → Hopper transaction barrier →
  Blackwell TMEM as one continuous line.
- `evolution-data-supply.md` — Fermi's L2 partitioning intersects with
  the data-supply chain (Kepler completes the SM-side loop after
  Fermi).
- `arch-kepler.md`, `arch-maxwell.md` — the post-Fermi consolidation.
- `patent-family-map.md` — full patent IDs and dates.

## Escalate to raw when

- You need exact GF100 vs GF110 deltas — gpu-arch-wiki has only the
  GF100 SM card; check NVIDIA's Fermi whitepaper.
- You need claim-level reading of the cache-modifier patent — open
  US9639479B2 on Google Patents.
- You need PTX-level cache-modifier syntax for current ISA — open
  `.claude/knowledge/ptx/ptx-isa-cache-hints.md` (the modifiers
  Fermi introduced are unchanged in modern PTX).
- You need the L2 cache control CUDA API surface — open the CUDA
  Programming Guide §4.13 "L2 Cache Control."
