---
Use when: explaining why the GPU has multiple barrier types; placing software scoreboard / convergence barrier / mbarrier / transaction barrier on one timeline; reasoning about why `bar.sync` vs `mbarrier` vs `mbarrier.complete_tx` exist as separate primitives.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §3.4.
Supporting sources: per-arch cards (Maxwell, Volta, Ampere, Hopper); references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (mbarrier, barrier.cluster, transaction-aware barrier).
Patents anchored here: US9612836B2 (software scoreboard, Maxwell), US10067768B2 (convergence barrier, Maxwell→Volta), US20230289242A1 (transaction barrier, Hopper).
Out of scope: memory ordering / fences (`evolution-memory-side.md`); execution model (`evolution-execution-model.md`).
Version family: Tesla through Blackwell. Mirror snapshot: 2026-05.
---

# Synchronization Primitives — register-deps → scoreboard → mbarrier → transaction barrier

The synchronization vocabulary of a CUDA / PTX programmer expands
across generations. This card walks the timeline of **what** is
being waited on, **at what scope**, and **with what mechanism**.

## The original problem (Tesla)

Tesla / early-GPU synchronization is mostly:

- **Register dependencies**: a write to register X must complete
  before X is read.
- **Pending-write tracking**: a busy-bit per register.
- **Scoreboard**: hardware-internal, software cannot read or write it.
- **Local progress conditions**: can this instruction issue right now?

It answers "can this instruction issue?" — not "is this batch of
async work actually consumable?"

## Maxwell — software scoreboarding (US9612836B2)

**Filed 2014-02-03.** Inventors include Choquette and Giroux.

The innovation: move scoreboarding from **hardware-implicit** to
**software/compiler-explicit**. The compiler emits a dependency-barrier
instruction with an immediate value and a scoreboard register
identifier. Comparing the immediate to the register's value gates
dispatch.

Mechanics:

- 6-bit scoreboard counters (0–63).
- Compiler-emitted `DEPBAR`-class instructions check whether the
  counter is at-or-below the immediate.
- Cheaper hardware (less circuit complexity, less power) and more
  flexible scheduling.

PTX/SASS surface: `DEPBAR`, later `LDGDEPBAR` for async-load
dependencies.

## Maxwell — convergence barrier (US10067768B2)

**Earliest priority 2014-07-18, filed 2015-07-13.** Inventors:
Choquette, Tirumala.

Replaces stack-based / priority-based reconvergence with **explicit
barrier objects**. Threads entering a divergence region register on a
barrier; threads at the convergence point wait; threads still
executing carry on; once all participants arrive, the barrier
releases all of them.

Critical innovation: the divergence-management mechanism is
**decoupled from thread scheduling**. Either can change without the
other.

YIELD mechanism guarantees forward progress — no thread can
indefinitely block any other. **This is what makes Volta's ITS
correct.**

Compiler-inserted SASS: `BMOV`, `BSSY`, `BSYNC`, `BREAK`. Not visible
in CUDA C++ or PTX.

For details: `arch-maxwell.md`.

## Volta — sync primitives reach the user surface

Volta exposes new explicit-mask synchronization to the programmer:

- `__syncwarp(mask)` — wait until the masked threads of the warp are
  at this PC.
- `__shfl_sync`, `__ballot_sync`, `__any_sync`, `__all_sync` and the
  `_sync` family — explicit-mask warp intrinsics.
- `__activemask()` — query which threads are active.

These exist because Volta's per-thread PC means the hardware **no
longer assumes** all 32 threads are at the same PC. The programmer
must say which threads to require.

## Ampere — execution-barrier metadata stabilizes

Ampere matures the barrier object machinery:

- `mbarrier` — software-managed barrier object in shared memory.
  PTX: `mbarrier.init`, `mbarrier.arrive`, `mbarrier.test_wait`,
  `mbarrier.try_wait`, `mbarrier.invalidate`.
- `Join / Wait / Cancel` semantics stabilize.

`mbarrier` lives in shared memory (per-block by default), is software-
managed, and supports an arrival count — useful for cooperative-group-
style sync where the participating thread set is determined at runtime
rather than baked into the warp/CTA shape.

CUDA C++ surface: `cuda::barrier`.

## Hopper — transaction barrier (US20230289242A1)

**Filed 2022-03-10.**

Extends or replaces shared-memory-backed arrive-wait barriers with
**transaction-aware** barriers that count both thread arrivals **and**
async-transaction arrivals.

### Why ordinary barriers aren't enough

A traditional barrier answers "have all threads arrived?" It cannot
answer "has the data they were supposed to produce **actually
arrived**?"

When the producer is a TMA copy, those are two different events:

- The thread that issued `cp.async.bulk.tensor` reaches its `arrive`
  point.
- The TMA hardware then completes the actual data movement, possibly
  many cycles later.

The transaction barrier waits on **both**.

### Mechanics

- The barrier object accepts an **expected transaction count** in
  bytes.
- Producer (TMA copy) signals `mbarrier.complete_tx::bytes` for each
  completed transaction.
- Consumer waits on the barrier; release happens when **arrival count
  AND transaction-count expectation are both met**.

### PTX surface

- `mbarrier.arrive.expect_tx`
- `mbarrier.complete_tx::bytes`
- `mbarrier.try_wait.parity`

### CUDA C++ surface

- `cuda::barrier` with `barrier_arrive_tx` / `barrier_expect_tx`.

For details: `arch-hopper.md`.

## Hopper — cluster barrier

Cluster sync is its own primitive, layered on top of CTA-level
barriers:

- PTX: `barrier.cluster.{arrive|wait|sync}`.
- CUDA C++: `cluster.sync()` from cooperative-groups.

A `cluster.sync()` is required:

- Before any DSMEM access of another block's shared memory.
- Before a block exits while peers may still read its shared memory.

## Hopper — memory synchronization domains

Beyond the barrier types, Hopper adds **scope reduction** for fences
and barriers:

- `cta` / `cluster` / `gpu` / `sys` scopes.
- `memory_sync_domain` annotations let a fence target a subset of
  memory traffic.

This descends from the Fermi-era "fine-tuned cache-to-backing-store
sync" idea (US9639466B2) — the principle that a sync should target
specific memory effects, not the whole world.

See `evolution-memory-side.md` for that thread.

## Blackwell — sync primitives unchanged

Blackwell keeps Hopper's sync primitive set. New additions are mostly
in the matrix-instruction layer (`tcgen05` family) which has its own
completion / commit / wait machinery analogous to `wgmma.commit_group`
/ `wgmma.wait_group` on Hopper.

## Sync timeline summary table

| What is being waited on | Mechanism | Generation introduced |
|---|---|---|
| Register write completion | hardware busy-bit / scoreboard | Tesla |
| Per-warp dependency | software scoreboard (`DEPBAR`) | Maxwell |
| Warp-divergence reconvergence | convergence barrier (`BSSY/BSYNC`) | Maxwell (mechanism) → Volta (default) |
| Specific warp-mask convergence | `__syncwarp(mask)` / `_sync` family | Volta |
| Block-level threads | `__syncthreads()` (`bar.sync`) | Tesla (always) |
| Cooperative thread set | `mbarrier` software barrier | Ampere |
| Async transaction (TMA copy) | `mbarrier.complete_tx::bytes` (transaction barrier) | Hopper |
| Cluster-level threads | `cluster.sync()` (`barrier.cluster`) | Hopper |
| Memory-effect class / observer class | `memory_sync_domain` | Hopper |

## Why a future agent cares

- A program that uses `__syncwarp` without a mask depended on pre-
  Volta default-everyone semantics. Modern PTX requires the mask.
- Mistaking `bar.sync` for `mbarrier` loses async-transaction
  capability.
- TMA copies that aren't paired with a transaction barrier will
  consume garbage on Hopper. The thread arriving doesn't mean the
  data arrived.
- Cluster barriers (`cluster.sync()`) are required around any DSMEM
  read; you cannot omit them assuming "blocks just see each other's
  SMEM."
- Pre-Maxwell, software cannot manipulate scoreboards. Trace replay
  has to model implicit-hardware scoreboard semantics in pre-Maxwell
  cases.
- Pre-Volta, CPU-style mutex loops within a warp deadlock. Forward
  progress isn't guaranteed without YIELD.

## See also

- `arch-maxwell.md` — software scoreboard + convergence barrier.
- `arch-volta.md` — `_sync` family becomes important.
- `arch-ampere.md` — `mbarrier` machinery stabilizes.
- `arch-hopper.md` — transaction barrier + cluster barrier + sync
  domains.
- `evolution-memory-side.md` — synchronization scope ↔ memory-effect
  domain.
- `evolution-execution-model.md` — convergence barrier underpins ITS.
- `.claude/knowledge/ptx/ptx-isa-barriers.md` — current PTX
  syntax for all barrier instructions.
- `.claude/knowledge/cuda/cuda-sync-memory-ordering.md` —
  CUDA-level sync semantics.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need precise `mbarrier` state-machine semantics — see
  `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md` (search
  for "mbarrier").
- You need transaction-barrier patent claims — Google Patents
  US20230289242A1.
- You need YIELD's effect on scheduler fairness — academic papers
  on Volta ITS or NVIDIA's Volta whitepaper.
