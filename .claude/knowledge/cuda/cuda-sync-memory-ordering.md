---
Use when: debugging data races, ordering bugs, or "stale-value" reads in CUDA kernels; reasoning about memory consistency, atomics, fences (`__threadfence`, `fence.acq_rel`, `fence.sc`), and acquire/release semantics across threads, warps, blocks, or clusters; verifying correctness of `atomicCAS` / `atomicAdd` patterns; mapping CUDA C++ `memory_order_*` to PTX scopes (`.cta`/`.cluster`/`.gpu`/`.sys`); reasoning about race conditions in cluster microbenchmarks; understanding when a read can observe a write.
Primary raw source: references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md
Supporting sources:
- references/NVIDIA_Manuals/cuda_guide/ (grep for __syncthreads, __threadfence, atomicAdd)
- .claude/knowledge/ptx/ptx-isa-barriers.md (barrier instruction syntax)
- .claude/knowledge/ptx/ptx-isa-memory-spaces.md (fence/membar syntax)
- .claude/knowledge/ptx/ptx-isa-load-store.md (ld/st/atom qualifier tables)
Repo touchpoints:
- workloads/simbench_bw/ (cluster and memory workloads rely on correct sync)
- tools/nvbit_tracer/ (optional dynamic evidence for synchronization sites)
Out of scope: active simulator memory-ordering implementation, barrier instruction syntax details (see ptx-isa-barriers.md), fence/membar PTX syntax (see ptx-isa-memory-spaces.md), optimization advice like "avoid atomics" (see perf-instruction-controlflow.md).
Version family: PTX ISA on sm_70+. Mirror snapshot 2026-04-16.
---

# CUDA Synchronization & Memory Ordering

This card consolidates the memory consistency model for kernel correctness and
non-root measurement design. The existing PTX cards document instruction syntax
in detail; this card focuses on the ordering rules, scope hierarchy, and CUDA
C++ to PTX mapping that determine when writes become visible.

## Operation Type Taxonomy

Every PTX memory instruction falls into one of these categories:

| Type | Examples | Ordering guarantee |
|------|----------|-------------------|
| **weak** | `ld` / `st` with no qualifier | None. May be reordered freely. |
| **relaxed** | `ld.relaxed.gpu` / `atom.relaxed.cta` | Atomic (if atom), scoped, but no ordering beyond data dependency. |
| **acquire** | `ld.acquire.gpu` | Subsequent ops cannot move before this load. |
| **release** | `st.release.gpu` | Prior ops cannot move after this store. |
| **acq_rel** | `atom.acq_rel.gpu` | Combined acquire + release on a single atomic. |
| **volatile** | `ld.volatile` | Equivalent to `relaxed.sys` with instruction-count preservation. Deprecated for sync. |
| **fence** | `fence.sc.gpu` / `fence.acq_rel.cta` | Ordering only (no data movement). |

**Strong** = any of the above (not weak). **Synchronizing** = barrier, fence, acquire, or release.

## Scope Hierarchy

Every strong operation declares its visibility scope:

```
.cta  <  .cluster  <  .gpu  <  .sys
```

**Key rule**: two `atom`/`red` operations are mutually atomic only if each
operation's scope includes the other thread. Mismatched scopes break atomicity.

**State space constrains effective scope**: `ld.relaxed.shared.sys` has the
same synchronizing effect as `.cluster` because no thread outside the cluster
can access shared memory.

The warp is explicitly NOT a scope in the PTX model.

## Morally Strong Predicate

Two operations are "morally strong" if they are both strong with mutually-inclusive
scopes, use the same proxy, and (if both are memory ops) overlap completely.
Most consistency axioms apply only to morally-strong pairs.

## Causality Order

The memory model defines which writes a read may observe via causality order:

1. **Program order**: within a single thread, earlier ops precede later ops.
2. **Synchronizes-with**: release-acquire pairs across threads establish cross-thread ordering.
3. **Transitive closure**: causality = transitive closure of (program order + synchronizes-with).

**Causality axiom**: a read R cannot return a write W if there exists a write W' such that W causally precedes W' causally precedes R.

## CUDA C++ to PTX Mapping

| CUDA C++ | PTX equivalent | Scope | Ordering |
|----------|---------------|-------|----------|
| `__syncthreads()` | `bar.cta.sync.aligned` | CTA | Full barrier (sync + memory ordering) |
| `__threadfence_block()` | `fence.sc.cta` | CTA | Caller's ops only (no cross-thread sync) |
| `__threadfence()` | `fence.sc.gpu` | GPU | Device-scope SC fence |
| `__threadfence_system()` | `fence.sc.sys` | System | Crosses CPU-GPU boundary |
| `atomicAdd(addr, val)` | `atom.global.add` | GPU | `.relaxed` (no ordering!) |
| `atomicAdd_block(...)` | `atom.shared.add` / `atom.global.add.cta` | CTA | `.relaxed` |
| `atomicAdd_system(...)` | `atom.global.add.sys` | System | `.relaxed` |
| `cuda::atomic_thread_fence(order, scope)` | `fence.{sc,acq_rel}.{cta,gpu,sys}` | explicit | Modern preferred API |

**Critical**: all legacy `atomicX()` functions use `memory_order_relaxed`. They
provide atomicity but NO ordering. For ordering, use `cuda::atomic<T, scope>`
with explicit memory order, or insert `__threadfence*()` around the atomic.

## Measurement-Critical Gotchas

1. **`red` vs `atom` for acquire patterns**: A `red` (reduction) instruction
   does NOT form an acquire pattern with a subsequent `fence.acquire` (PTX ISA
   section 8.11). Only `atom` does. Confusing the two can make a workload look
   nondeterministic.

2. **Async operations are NOT in program order**: `cp.async`, `cp.async.bulk`,
   and `wgmma.mma_async` have separate ordering via `cp.async.commit_group`
   and mbarrier. The issuing thread cannot assume the async op completed
   at the next instruction.

3. **Vector ops are unordered scalar ops**: A vector load/store is modeled as
   independent scalar operations on each element in unspecified order.
   Atomicity is per-element only, not per-vector.

4. **Texture/surface accesses are excluded**: The memory consistency model does
   not apply to texture accesses (including `ld.global.nc`).

## Fence Types Compared

| Fence | Effect | Legacy alias |
|-------|--------|-------------|
| `fence.sc.cta` | SC fence at CTA scope | `membar.cta` |
| `fence.sc.gpu` | SC fence at device scope | `membar.gl` |
| `fence.sc.sys` | SC fence at system scope | `membar.sys` |
| `fence.acq_rel.{scope}` | Acquire-release only. Does NOT prevent store-buffering. Weaker than SC. | — |
| `fence.proxy.async` | Orders prior async copies relative to subsequent sync accesses | — |

**SC vs acq_rel**: `fence.sc` establishes a total order among all morally-strong
`fence.sc` operations, preventing the store-buffering anomaly (thread A writes X,
fences, reads Y; thread B writes Y, fences, reads X — at least one reader sees
the other's write). `fence.acq_rel` does NOT prevent this outcome.

**When to use proxy fences**: after `cp.async` + `cp.async.commit_group` +
mbarrier-wait, a `fence.proxy.async` is required before the issuing thread can
read the copied data. Without the proxy fence, the read may observe stale data
because async copies use a different proxy than synchronous loads.

## Measurement Implications

Synchronization choices are part of the workload contract. Record barrier and
fence structure when comparing elapsed time across variants, and avoid
attributing timing changes to memory hierarchy alone when the synchronization
scope or ordering strength changed.

## Modeling context (papers)

- [Exploring Modern GPU Memory System Design Challenges through Accurate Modeling](references/papers/knowledge_base/active/2018__exploring_modern_gpu_memory_system_design_challenges_through_accurate_modeling.pdf) — explains how memory-system modeling mistakes distort cycle counts; retained as prior art for interpreting synchronization and memory effects.

## See also

- `.claude/knowledge/ptx/ptx-isa-barriers.md` — full barrier instruction syntax and mbarrier lifecycle.
- `.claude/knowledge/ptx/ptx-isa-memory-spaces.md` — full fence/membar PTX syntax and scope tables.
- `.claude/knowledge/ptx/ptx-isa-load-store.md` — ld/st/atom qualifier tables and atomicity rules.
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md` — cluster sync discipline for DSMEM.

## Escalate to raw when

- You need the formal axioms (coherence, causality, SC fence ordering) — see `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md` sections 8.9-8.10.
- You need litmus tests for specific ordering patterns — see section 8.11 of the same file.
- You need `cuda::atomic<T, scope>` API details — grep `references/NVIDIA_Manuals/cuda_guide/` for `cuda::atomic`.
