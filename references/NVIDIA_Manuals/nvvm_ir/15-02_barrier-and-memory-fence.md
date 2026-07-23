---
title: "15.2. Barrier and Memory Fence"
section: "15.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#barrier-and-memory-fence"
---

## [15.2. Barrier and Memory Fence](https://docs.nvidia.com/cuda/nvvm-ir-spec#barrier-and-memory-fence)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#barrier-and-memory-fence "Permalink to this headline")

```text
declare void @llvm.nvvm.barrier0()
```

waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `llvm.nvvm.barrier0()` are visible to all threads in the block.

```text
declare i32 @llvm.nvvm.barrier0.popc(i32)
```

is identical to `llvm.nvvm.barrier0()` with the additional feature that it evaluates predicate for all threads of the block and returns the number of threads for which predicate evaluates to non-zero.

```text
declare i32 @llvm.nvvm.barrier0.and(i32)
```

is identical to `llvm.nvvm.barrier0()` with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for all of them.

```text
declare i32 @llvm.nvvm.barrier0.or(i32)
```

is identical to `llvm.nvvm.barrier0()` with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for any of them.

```text
declare void @llvm.nvvm.cluster.barrier(i32 %flags)
```

Synchronize and communicate among threads in the same cluster. This intrinsic is only supported for Hopper+. The %flags is encoded according to the following table:

| %flags bits | Meaning |
| --- | --- |
| 31-8 | Reserved |
| 7-4 | Memory ordering (See Cluster Barrier Memory Ordering Encoding below) |
| 3-0 | Operation mode (See Cluster Barrier Operation Mode Encoding below) |

Cluster Barrier Operation Mode Encoding

| Encoding | Mode | Description |
| --- | --- | --- |
| 0 | Arrive | Arrive at cluster barrier |
| 1 | Wait | Wait at cluster barrier |
| 2-15 | RESERVED | RESERVED |

Cluster Barrier Memory Ordering Encoding

| Encoding | Mode | Description |
| --- | --- | --- |
| 0 | Default | All synchronous memory accesses requested by the executing entry prior to arrive are performed and are visible to all the entrys in the cluster after wait. |
| 1 | Relaxed | All previously fenced memory accesses requested by the executing entry prior to arrive are performed and are visible to all the entrys in the cluster after wait. This ordering is only supported when the operation mode is Arrive. |
| 2-15 | RESERVED | RESERVED |

```text
declare void @llvm.nvvm.membar.cta()
```

is a memory fence at the thread block level. This intrinsic is deprecated. Please use nvvm.membar with flags as argument instead.

```text
declare void @llvm.nvvm.membar.gl()
```

is a memory fence at the device level. This intrinsic is deprecated. Please use nvvm.membar with flags as argument instead.

```text
declare void @llvm.nvvm.membar.sys()
```

is a memory fence at the system level. This intrinsic is deprecated. Please use nvvm.membar with flags as argument instead.

```text
declare void @llvm.nvvm.membar(i32 %flags)
```

Wait for all prior memory accesses requested by this thread to be performed at a membar level defined by the membar mode below. The memory barrier enforces vertical ordering only. It makes no guarantees as to execution synchronization with other threads. For horizontal synchronization, a barrier should be used instead, or in addition to membar.

The %flags is encoded according to the following table:

| %flags bits | Meaning |
| --- | --- |
| 31-4 | Reserved |
| 3-0 | Membar modes (See Membar Mode Encoding.) |

Membar Mode Encoding

| Encoding | Mode | Description |
| --- | --- | --- |
| 0 | GLOBAL | Membar at the global level |
| 1 | CTA | Membar at the CTA level |
| 2 | SYSTEM | Membar at the system level |
| 3 | RESERVED | RESERVED |
| 4 | CLUSTER | Membar at the cluster level, only on Hopper+ |
| 5-15 | RESERVED | RESERVED |
