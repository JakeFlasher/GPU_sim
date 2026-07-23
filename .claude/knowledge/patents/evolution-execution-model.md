---
Use when: explaining how the GPU execution-progress unit changes generation by generation; reasoning about why `__syncwarp()` exists; placing per-thread PCs / convergence barriers / cluster launch on one timeline; deciding what "warp" actually means at a given compute capability.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §3.1.
Supporting sources: per-arch cards (Tesla, Maxwell, Volta, Hopper, Blackwell); references/NVIDIA_Manuals/cuda_guide/01-02-02-01-01_thread-block-clusters.md.
Patents anchored here: US10067768B2 (convergence barrier, Maxwell-era → Volta default); US20230289189A1 (Distributed Shared Memory, Hopper).
Out of scope: data supply (`evolution-data-supply.md`); synchronization primitives (`evolution-synchronization.md`); precision (`evolution-precision.md`).
Version family: Tesla through Blackwell. Mirror snapshot: 2026-05.
---

# Execution Model — from `warp lockstep` to cluster-scale launch

The GPU execution-progress unit isn't fixed. It changes across
generations as hardware peels back assumptions. This card walks the
timeline.

## The core problem (Tesla baseline)

Early GPUs treat **warp as a natural synchronous-advance unit**. This
simplifies control logic — only one PC per warp, one active mask,
one reconvergence stack — but compresses divergence, long waits, and
async events into "the whole warp pauses together."

That default propagates through every generation until Volta breaks it.

## Timeline

| Generation | Default execution unit | Key mechanism |
|---|---|---|
| Tesla | Warp (32 threads, one PC) | `active mask` + reconvergence stack |
| Fermi | Same as Tesla | (rewrite is on memory side, not execution) |
| Kepler | Same | (rewrite is on data supply) |
| Maxwell | Same | (mechanism added: software scoreboard + convergence barrier; not yet default) |
| Pascal | Same | (no execution-model change) |
| **Volta** | **Per-thread PC (ITS)** | Convergence barrier becomes default; `__syncwarp()` family important |
| Turing | Same as Volta | (extends precision; uniform datapath emerges in ISA) |
| Ampere | Same | (`cp.async` adds async object; "shard scheduling" cited as patent-side inference) |
| Ada | Same | (consumer Hopper-era; no exec-model change) |
| **Hopper** | **Cluster (multiple co-scheduled CTAs)** | Thread Block Cluster + DSM as new cooperation tier |
| Blackwell | Cluster + per-CTA TMEM as compute target | Single-thread issue (`tcgen05.mma`) for matrix initiation |

## Volta's rewrite — implicit warp lockstep gone

Two things change at once when ITS becomes the default:

### 1. Per-thread PC

Before Volta: one PC per warp.
After Volta: each thread has independent PC and call stack. Threads
on different paths can be **interleaved** by the scheduler at any
cycle.

The `_sync` family (`__shfl_sync`, `__ballot_sync`, etc.) takes a
mask explicitly because the hardware no longer assumes the full
32-thread warp is at the same PC.

### 2. Convergence barriers replace the reconvergence stack

`US10067768B2` (Maxwell-era) makes the divergence-management
mechanism a barrier object that's **decoupled from scheduling**.
Either can change without the other. The compiler inserts
`BSSY/BSYNC/BMOV/BREAK` to mark divergence regions and
reconvergence points.

The **YIELD** mechanism in the same patent gives forward-progress
guarantees. Without YIELD, ITS would be incorrect (a thread holding
a lock could starve another thread waiting for it).

### Why this is a real breakpoint

CPU-style mutexes become safe on Volta+. They are **not** safe pre-Volta,
because the warp-lockstep model can't tolerate "thread A holds lock,
thread B waits for it" within one warp. ITS + YIELD makes that
work.

## Hopper's rewrite — cluster as a new cooperation tier

The grid hierarchy adds a level:

```text
Tesla–Ampere:  Grid -> Thread Block       -> Thread
Hopper+:       Grid -> Thread Block Cluster -> Thread Block -> Thread
```

A cluster is **1-D, 2-D, or 3-D** sets of thread blocks, all
co-scheduled in a single GPC. Blocks within a cluster:

- Share a hardware-guaranteed resident scope.
- Can read/write/atomic each other's shared memory via DSMEM.
- Synchronize at cluster scope via `cluster.sync()` (PTX:
  `barrier.cluster.{arrive|wait|sync}`).
- Cannot move between GPCs mid-execution.

This means execution is no longer just "how does a warp advance" but
also "which **larger** objects can be launched cooperatively." See
`arch-hopper.md` for the full mechanism.

### Cluster-size rules

- Portable cluster size: **8** across all sm_90+ devices.
- Nonportable opt-in (B200): **16** via
  `cudaFuncAttributeNonPortableClusterSizeAllowed`.
- Other arches' nonportable ceilings must be probed via launch test;
  `cudaOccupancyMaxPotentialClusterSize` can over-report.
- Grid dim must be a multiple of cluster dim in every axis (runtime
  rejects other launches with `cudaErrorInvalidValue`).

## Blackwell — single-thread issue

`tcgen05.mma` introduces a different execution-model wrinkle:
**single-thread issue semantics**. One thread can launch the entire
matrix operation; operands and accumulators live in TMEM.

This is **not** a return to per-warp lockstep. Threads still have
independent PCs (ITS still default). What changes is the matrix
**initiation surface**: it goes from warp-group collective issue
(Hopper `wgmma.mma_async`) to single-thread issue (Blackwell
`tcgen05.mma`).

Read together with the cluster tier:

| Tier | Granularity | Generation introduced |
|---|---|---|
| Thread | individual PC (ITS) | Volta |
| Warp | scheduling unit (when same PC) | Tesla |
| Warp group | 4 warps × 32 = 128 threads | Hopper (WGMMA) |
| CTA / thread block | shared memory + barriers | Tesla |
| Cluster | 1–16 CTAs co-scheduled | Hopper |
| CTA-pair | tcgen05 dual-CTA | Blackwell |
| Grid | full launch | Tesla |

## What stays default

- **SIMT execution** — divergent paths still serialize within a warp;
  ITS just lets them be interleaved at finer granularity.
- **One-PC-per-warp scheduling** — the scheduler still picks one PC
  per warp per cycle. ITS doesn't multi-issue PCs simultaneously
  within a warp.
- **CTA → SM mapping** — a thread block is still scheduled to one SM.
  Cluster lifts the cooperation scope, not the per-CTA placement.

## Why a future agent cares

- "32-thread warp moving as one" is **only correct pre-Volta**. Volta
  + later: explicit `__syncwarp()` to recover that property.
- "Threads in a warp can run CPU-style mutexes" is **only correct
  post-Volta**. Pre-Volta: deadlock by construction.
- "Block-to-block via global memory only" is **only correct
  pre-Hopper**. Hopper+: DSMEM within a cluster.
- "All warps in a warp group must execute the same MMA instruction"
  is **only Hopper**. Blackwell `tcgen05.mma` uses single-thread issue.
- A trace replay assuming pre-Volta execution model on Volta+ HW will
  miss interleaved divergence patterns.

## See also

- `arch-tesla.md` — baseline machine where the warp-lockstep default
  comes from.
- `arch-maxwell.md` — convergence-barrier mechanism that prepared
  Volta.
- `arch-volta.md` — when ITS becomes the default execution model.
- `arch-hopper.md` — cluster + DSMEM as the new cooperation tier.
- `arch-blackwell.md` — single-thread issue for matrix.
- `evolution-synchronization.md` — synchronization primitives evolve
  alongside the execution model.
- `.claude/knowledge/cuda/cuda-warp-divergence.md` — current-
  ISA reconvergence semantics.
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md` — current-
  CUDA cluster API.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need pre-Volta vs Volta divergence behavior compared in code —
  see Collabora's "Re-converging control flow on NVIDIA GPUs" blog
  (2024-04-25).
- You need the formal PTX memory consistency / scheduling model —
  `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md`.
- You need exact `barrier.cluster.{arrive|wait|sync}` PTX semantics —
  `.claude/knowledge/ptx/ptx-isa-barriers.md`.
- You need `tcgen05.mma` issue semantics in detail — PTX
  09_instruction-set.md §9.7.16, "Tensor Core 5th Generation
  Instructions."
