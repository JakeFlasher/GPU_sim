---
Use when: writing, modifying, or reviewing cluster-using CUDA workloads under the current non-root measurement project; deciding how a multi-block cluster launch should be configured; reasoning about why `cluster.sync()` is needed; diagnosing a cluster-launch failure.
Primary raw source: references/NVIDIA_Manuals/cuda_guide/02-02-03-08_distributed-shared-memory.md.
Supporting sources: references/NVIDIA_Manuals/cuda_guide/01-02-02-01-01_thread-block-clusters.md; references/NVIDIA_Manuals/cuda_guide/03-01-02-01_launching-with-clusters-using-cudalaunchkernelex.md; references/NVIDIA_Manuals/cuda_guide/05-06-03-01-02_class-cluster-group.md (cluster_group API).
Repo touchpoints: workloads/simbench_bw/l2/cluster_remote_reduction/, tools/cloud/preflight.sh, workloads/simbench_bw/.
Out of scope: arch-specific cluster ceilings and tuning numbers (in Blackwell / Ampere cards); PTX-level cluster barriers / async copy; generic cooperative groups.
Version family: CUDA 13.x, compute capability 9.0+. Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# Thread Block Clusters & Distributed Shared Memory (DSMEM)

Clusters are an optional **third** level of grouping introduced at compute
capability 9.0 (Hopper). A cluster groups **adjacent** thread blocks
scheduled **simultaneously on SMs within a single GPC**. Blocks in a
cluster can read/write/atomic each other's shared memory — that joint
address space is **Distributed Shared Memory (DSMEM)**.

This is a workload surface for non-root wall-clock and cache-behavior
prediction. Historical cluster-modeling plans are no longer active.

## Execution model (the parts that affect modeling)

- Clusters are **1-D, 2-D, or 3-D** sets of thread blocks, laid out like blocks/grid.
- Cluster indexing does **not** change grid dims or a block's grid index; it adds a cluster index.
- All blocks in a cluster **execute simultaneously in a single GPC**. A block cannot move between GPCs mid-execution.
- **Grid dim must be a multiple of cluster dim** in every axis. The runtime rejects launches that violate this (`cudaErrorInvalidValue`).
- `cluster.sync()` guarantees all blocks in the cluster are resident. Any DSMEM access that reads another block's shared memory **requires** a preceding `cluster.sync()`. Any block that is about to exit while peers may still read its shared memory **requires** a trailing `cluster.sync()`.

## Cluster-size ceiling

**The manuals support three factual statements**; do not over-generalize:

| Scope | What the manual says |
|-------|----------------------|
| Portable | Max cluster size **8** is portable across all cluster-supporting GPUs (compute capability 9.0+). |
| Arch-specific | The **actual** per-device maximum is queried via `cudaOccupancyMaxPotentialClusterSize`; it varies per GPU and per kernel. |
| Nonportable opt-in | Blackwell **B200** allows cluster size **16** after setting `cudaFuncAttributeNonPortableClusterSizeAllowed`. Other arches' nonportable ceilings must be confirmed by launch probe, not assumed. |

**Repo observation (SM120, cloud RTX PRO 6000 Blackwell / GB202):** the runtime can
report `cudaOccupancyMaxPotentialClusterSize` above 8 even when launching
a cluster of that size fails without the opt-in. Treat the true ceiling
as **"largest size that `cudaLaunchKernelEx` actually accepted"**, not
the advertised potential. `cluster_support_probe.cu` encodes this
discipline: it enumerates `{1, 2, 4, 8, max_potential}`, launch-tests
each, and reports the largest that launched and round-tripped
`cluster.num_blocks()` correctly.

## Two launch paths

### A. Compile-time via `__cluster_dims__` / triple-chevron

```cpp
// Fix cluster dim at compile time
__global__ void __cluster_dims__(2, 1, 1) my_kernel(...) { ... }

// Launch with triple chevron; cluster is implicit from the attribute
my_kernel<<<gridDim, blockDim>>>(...);
```

Use this only when the cluster dim is **truly fixed**. Changing it at launch time is an error.

### B. Runtime via `cudaLaunchKernelEx`

```cpp
cudaLaunchConfig_t config = {0};
config.gridDim  = dim3(N / threads_per_block, 1, 1);
config.blockDim = dim3(threads_per_block, 1, 1);
config.dynamicSmemBytes = /* per-block dyn shmem */;
config.stream   = stream;

cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim.x = cluster_size;
attr[0].val.clusterDim.y = 1;
attr[0].val.clusterDim.z = 1;

config.attrs    = attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel_fn, arg0, arg1, ...);
```

This is what `cluster_support_probe.cu` uses (see `cluster_support_probe.cu:143`).

### Launch attributes of interest

| Attribute | Purpose |
|-----------|---------|
| `cudaLaunchAttributeClusterDimension` | Required cluster dim (3-D). Grid dim must be divisible by this. |
| `cudaLaunchAttributePreferredClusterDimension` | (compute capability 10.0+) Preferred dim, integer multiple of the required dim. Kernel must handle **either** dim at runtime; not all clusters are guaranteed the preferred size. |
| `cudaFuncAttributeNonPortableClusterSizeAllowed` | Opt-in required to launch cluster sizes > 8. Set via `cudaFuncSetAttribute`. |

## Cluster-aware Cooperative Groups APIs

From `<cooperative_groups.h>` (namespace `cg`), `cluster_group`:

| Member | Returns |
|--------|---------|
| `cg::this_cluster()` | `cluster_group` for the current block's cluster |
| `.num_blocks()` | Total blocks in the cluster (product of cluster dim) |
| `.block_rank()` | This block's rank within the cluster |
| `.dim_blocks()` | `dim3` cluster dim (x, y, z) |
| `.block_index()` | `dim3` this block's (x, y, z) coord in the cluster |
| `.sync()` | All-blocks-in-cluster barrier |
| `map_shared_rank(ptr, r)` (static) | Pointer into block rank `r`'s shared memory |
| `query_shared_rank(const void* addr)` (static) | Rank that owns the given shared-memory address |

Key contracts:

- `cluster.num_blocks()` at runtime **must** match the launched cluster dim.
- Atomics on DSMEM pointers work exactly like local `__shared__` atomics: `atomicAdd(remote_smem + offset, 1)`.

## DSMEM — what it is and is not

- **Is**: a unified address space over the shared memory of all blocks in a cluster. Total DSMEM bytes = `blocks_per_cluster * per_block_shared_bytes`.
- **Is**: readable, writable, atomic-able from any block in the cluster.
- **Is NOT**: persistent across clusters. DSMEM does not span the grid.
- **Is NOT**: shared with blocks in other clusters, even on the same SM.
- **Is NOT**: an L1/L2 tier — it is still shared memory.

### Required synchronization discipline (most common bug)

```cpp
__global__ void k() {
    extern __shared__ int smem[];
    smem[threadIdx.x] = 0;

    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();                                 // REQUIRED #1: all blocks
                                                    //   started AND local smem
                                                    //   initialized.
    int* remote = cluster.map_shared_rank(smem, peer_rank);
    atomicAdd(remote + offset, 1);                  // DSMEM write
    cluster.sync();                                 // REQUIRED #2: all DSMEM
                                                    //   ops complete; safe
                                                    //   for blocks to exit.
}
```

Skip either `cluster.sync()`:

- Missing #1: you may read/write another block's shared memory before it has initialized or even started executing.
- Missing #2: a block may exit while peer blocks still hold pointers into its shared memory — **undefined behavior**, not a deterministic hang.

**Repo policy:** every DSMEM workload should preserve both syncs when remote
shared memory may be accessed. The reason is correctness first and comparable
timing second.

## Occupancy for clusters

| API | Returns |
|-----|---------|
| `cudaOccupancyMaxPotentialClusterSize(&size, kernel, &cfg)` | Max cluster size the runtime thinks is usable for this kernel/config |
| `cudaOccupancyMaxActiveClusters(&n, kernel, &cfg)` | How many clusters of the specified dim can run concurrently |

Both take a `cudaLaunchConfig_t` with the kernel's block / shared / cluster attributes — pass the same config you'll pass to `cudaLaunchKernelEx`.

**Do NOT** use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for cluster-using kernels; it models single-block occupancy, not cluster occupancy.

## Repo-level Measurement Contract

**Repo policy**:

1. Cluster blocks are co-resident in a single GPC and execute simultaneously. Model as a scheduled unit, not as independent blocks.
2. Cluster dim divides grid dim exactly; scheduler never produces partial clusters.
3. `cluster.sync()` is a hard barrier across the cluster — model as such.
4. DSMEM latency sits closer to `__shared__` than to L2.
5. Per-device ceiling is probe-verified, not runtime-advertised.

## Failure modes you will actually hit

| Symptom | Likely cause | Fix / where to look |
|---------|--------------|---------------------|
| `cudaErrorInvalidValue` on `cudaLaunchKernelEx` | Grid dim not a multiple of cluster dim | Check `config.gridDim % attr.val.clusterDim == {0,0,0}` |
| Size-N launch fails where N should work | `cudaFuncAttributeNonPortableClusterSizeAllowed` not set | `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` before launch |
| `cluster.num_blocks()` ≠ launched dim | Compile-time `__cluster_dims__` conflicts with runtime attr | Use either compile-time OR runtime cluster dim, not both |
| DSMEM reads stale / wrong data | Missing `cluster.sync()` before or after | See "Required synchronization discipline" |
| Sticky `cudaErrorLaunchFailure` after a failed probe | Device-side error is sticky per context | `cudaGetLastError()` to clear (see `cluster_support_probe.cu:184`) |

## Modeling context (papers)
- [ClusterSim: Modeling Thread Block Clusters in Hopper GPUs](references/papers/knowledge_base/active/2025__clustersim_modeling_thread_block_clusters_in_hopper_gpus.pdf) — first dedicated cluster execution model; useful prior art for deciding which non-root observables to collect.

## See also

- `.claude/knowledge/ptx/ptx-isa-barriers.md` — PTX-level `barrier.cluster.*` / `fence.cluster.sync_restrict` / `mbarrier.*` semantics.
- `.claude/knowledge/ptx/ptx-isa-async-copy.md` — `cp.async.bulk.shared::cluster.*` and related DSMEM load/store PTX.
- `.claude/knowledge/blackwell/blackwell-tuning.md` — SM120 / B200 cluster ceilings and tuning guidance, including the portable-vs-nonportable ceiling story.
- `.claude/knowledge/blackwell/blackwell-compatibility.md` — `-arch=sm_120a` cluster-feature gating and JIT/cubin rules.
- `workloads/simbench_bw/l2/cluster_remote_reduction/` — current cluster workload touchpoint.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the full list of `cudaLaunchAttribute*` constants (many exist beyond cluster attrs — programmatic dependent launches, green contexts, launch stream ordering) — see `references/NVIDIA_Manuals/cuda_guide/03-01-01_cudalaunchkernelex.md`.
- You need "work stealing with cluster launch control", fire-and-forget launches, device-side launches — see `03-05-01-04_work-stealing-with-cluster-launch-control.md` and `04-12_cluster-launch-control-work-stealing-with-cluster-launch-control.md`.
- You need DSMEM behavior under programmatic dependent launch (PDL) — see `03-01-04_programmatic-dependent-kernel-launch.md`.
- You need the full `cluster_group` member listing, including methods not used in this repo — `references/NVIDIA_Manuals/cuda_guide/05-06-03-01-02_class-cluster-group.md`.
