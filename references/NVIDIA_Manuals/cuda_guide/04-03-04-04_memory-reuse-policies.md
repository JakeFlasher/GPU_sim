---
title: "4.3.4.4. Memory Reuse Policies"
section: "4.3.4.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#memory-reuse-policies"
---

### [4.3.4.4. Memory Reuse Policies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-reuse-policies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-reuse-policies "Permalink to this headline")

In order to service an allocation request, the driver attempts to reuse memory
that was previously freed via `cudaFreeAsync()` before attempting to
allocate more memory from the OS. For example, memory freed in a stream can
be reused immediately in a subsequent allocation request on the same stream.
When a stream is synchronized with the CPU, the memory that was
previously freed in that stream becomes available for reuse for an allocation
in any stream.
Reuse policies can be applied to both default and explicit memory pools.

The stream-ordered allocator has a few controllable allocation policies. The
pool attributes `cudaMemPoolReuseFollowEventDependencies`,
`cudaMemPoolReuseAllowOpportunistic`, and
`cudaMemPoolReuseAllowInternalDependencies` control these policies and are detailed below.
These policies can be enabled or disabled  through a call to `cudaMemPoolSetAttribute`.
Upgrading to a newer CUDA driver may change, enhance, augment and/or reorder
the enumeration of the reuse policies.

#### [4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreusefolloweventdependencies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreusefolloweventdependencies "Permalink to this headline")

Before allocating more physical GPU memory, the allocator examines dependency
information established by CUDA events and tries to allocate from memory freed
in another stream.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event,originalStream);

// waiting on the event that captures the free in another stream
// allows the allocator to reuse the memory to satisfy
// a new allocation request in the other stream when
// cudaMemPoolReuseFollowEventDependencies is enabled.
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);
```

#### [4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreuseallowopportunistic)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreuseallowopportunistic "Permalink to this headline")

When the `cudaMemPoolReuseAllowOpportunistic` policy is enabled, the allocator
examines freed allocations to see if the free operations stream order semantic has been
met, for example the stream has passed the point of execution indicated by the
free operation. When this policy is disabled, the allocator will still reuse memory made
available when a stream is synchronized with the CPU. Disabling this policy
does not stop the `cudaMemPoolReuseFollowEventDependencies` from applying.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// after some time, the kernel finishes running
wait(10);

// When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
// can be fulfilled with the prior allocation based on the progress of originalStream.
cudaMallocAsync(&ptr2, size, otherStream);
```

#### [4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudamempoolreuseallowinternaldependencies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudamempoolreuseallowinternaldependencies "Permalink to this headline")

Failing to allocate and map more physical memory from the OS, the driver will
look for memory whose availability depends on another stream’s pending
progress. If such memory is found, the driver will insert the required
dependency into the allocating stream and reuse the memory.

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// When cudaMemPoolReuseAllowInternalDependencies is enabled
// and the driver fails to allocate more physical memory, the driver may
// effectively perform a cudaStreamWaitEvent in the allocating stream
// to make sure that future work in ‘otherStream’ happens after the work
// in the original stream that would be allowed to access the original allocation.
cudaMallocAsync(&ptr2, size, otherStream);
```

#### [4.3.4.4.4. Disabling Reuse Policies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#disabling-reuse-policies)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#disabling-reuse-policies "Permalink to this headline")

While the controllable reuse policies improve memory reuse, users may want to
disable them. Allowing opportunistic reuse (such as
`cudaMemPoolReuseAllowOpportunistic`) introduces run to run variance in
allocation patterns based on the interleaving of CPU and GPU execution.
Internal dependency insertion (such as
`cudaMemPoolReuseAllowInternalDependencies`) can serialize work in
unexpected and potentially non-deterministic ways when the user would rather
explicitly synchronize an event or stream on allocation failure.
