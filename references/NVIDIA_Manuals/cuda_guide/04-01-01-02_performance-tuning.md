---
title: "4.1.1.2. Performance Tuning"
section: "4.1.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#performance-tuning"
---

### [4.1.1.2. Performance Tuning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#performance-tuning)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#performance-tuning "Permalink to this headline")

In order to achieve good performance with unified memory, it is important to:

- understand how paging works on your system, and how to avoid unnecessary page faults
- understand the various mechanisms allowing you to keep data local to the accessing processor
- consider tuning your application for the granularity of memory transfers of your system.

As general advice, performance hints (see [Performance Hints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-perf-hints))
might provide improved performance, but using them incorrectly might degrade performance
compared to the default behavior.
Also note that any hint has a performance cost associated with it on the host,
thus useful hints must at the very least improve performance enough to overcome this cost.

#### [4.1.1.2.1. Memory Paging and Page Sizes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-paging-and-page-sizes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-paging-and-page-sizes "Permalink to this headline")

To better understand the performance implication of unified memory, it is important to understand virtual addressing,
memory pages and page sizes.
This sub-section attempts to define all necessary terms and explain why paging matters for performance.

All currently supported systems for unified memory use a virtual address space:
this means that memory addresses used by an application represent a _virtual_ location
which might be _mapped_ to a physical location where the memory actually resides.

All currently supported processors, including both CPUs and GPUs, additionally use
memory _paging_. Because all systems use a virtual address space, there are two types
of memory pages:

- Virtual pages: This represents a fixed-size contiguous chunk of virtual memory
per process tracked by the operating system, which can be _mapped_ into physical memory.
Note that the virtual page is linked to the _mapping_: for example, a single
virtual address might be mapped into physical memory using different page sizes.
- Physical pages: This represents a fixed-size contiguous chunk of memory
the processor’s main Memory Management Unit (MMU) supports and into which
a virtual page can be mapped.

Currently, all x86_64 CPUs use a default physical page size of 4KiB.
Arm CPUs support multiple physical page sizes - 4KiB, 16KiB, 32KiB and 64KiB - depending on the exact CPU.
Finally, NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger.
Note that these sizes are subject to change in future hardware.

The default page size of virtual pages usually corresponds to the physical page size,
but an application may use different page sizes as long as they are supported by the
operating system and the hardware. Typically, supported virtual page sizes must be
powers of 2 and multiples of the physical page size.

The logical entity tracking the mapping of virtual pages into physical pages will be referred to as a _page table_,
and each mapping of a given virtual page with a given virtual size to physical pages is called a _Page Table Entry (PTE)_.
All supported processors provide specific caches for the page table to speed up the translation of
virtual addresses to physical addresses. These caches are called _Translation Lookaside Buffers (TLBs)_.

There are two important aspects for performance tuning of applications:

- the choice of virtual page size,
- whether the system offers a combined page table used by both CPUs and GPUs,
or separate page tables for each CPU and GPU individually.

##### [4.1.1.2.1.1. Choosing the Right Page Size](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#choosing-the-right-page-size)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#choosing-the-right-page-size "Permalink to this headline")

In general, small page sizes lead to less (virtual) memory fragmentation but more TLB misses,
whereas larger page sizes lead to more memory fragmentation but less TLB misses.
Additionally, memory migration is generally more expensive with larger page sizes compared to
smaller page sizes, because we typically migrate full memory pages. This can cause
larger latency spikes in an application using large page sizes. See also the next section
for more details on page faults.

One important aspect for performance tuning is that TLB misses are generally
significantly more expensive on the GPU compared to the CPU. This means that
if a GPU thread frequently accesses random locations of unified memory mapped
using a small enough page size, it might be significantly slower compared to
the same accesses to unified memory mapped using a large enough page size.
While a similar effect might occur for a CPU thread randomly accessing a large
area of memory mapped using a small page size, the slowdown is less pronounced,
meaning that the application might want to trade-off this slowdown with
having less memory fragmentation.

Note that in general, applications should not tune their performance to the
physical page size of a given processor, since physical page sizes are subject
to change depending on the hardware. The advice above only applies to virtual
page sizes.

##### [4.1.1.2.1.2. CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency "Permalink to this headline")

Hardware-coherent systems such as NVIDIA Grace Hopper offer a logically combined
page table for both CPUs and GPUs.
This is important because in order to access system-allocated memory from the GPU,
the GPU uses whichever page table entry was created by the CPU for the requested memory.
If that page table entry uses the default CPU page size of 4KiB or 64KiB,
accesses to large virtual memory areas will cause significant TLB misses,
thus significant slowdowns.

On the other hand, on software-coherent systems where the CPUs and GPUs each have their own logical
page table, different performance tuning aspects should be considered:
in order to guarantee coherency, these systems
usually use _page faults_ in case a processor accesses a memory address mapped
into the physical memory of a different processor. Such a page fault means that:

- It needs to be ensured that the currently owning processor (where the physical page currently resides)
cannot access this page anymore, either by deleting the page table entry or updating it.
- It needs to be ensured that the processor requesting access can access this page,
either by creating a new page table entry or updating and existing entry, such that
it becomes valid/active.
- The physical page backing this virtual page must be moved/migrated to the processor
requesting access: this can be an expensive operation, and the amount of work
is proportional to the page size.

Overall, hardware-coherent systems provide significant performance benefits
compared to software-coherent systems in cases where frequent concurrent accesses
to the same memory page are made by both CPU and GPU threads:

- less page-faults: these systems do not need to use page-faults for emulating coherency or migrating memory,
- less contention: these systems are coherent at cache-line granularity instead of page-size granularity, that is,
when there is contention from multiple processors within a cache line, only the cache line is exchanged which is much smaller than the smallest page-size,
and when the different processors access different cache-lines within a page, then there is no contention.

This impacts the performance of the following scenarios:

- atomic updates to the same address concurrently from both CPUs and GPUs
- signaling a GPU thread from a CPU thread or vice-versa.

#### [4.1.1.2.2. Direct Unified Memory Access from the Host](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#direct-unified-memory-access-from-the-host)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#direct-unified-memory-access-from-the-host "Permalink to this headline")

Some devices have hardware support for coherent reads, stores and atomic accesses
from the host on GPU-resident unified memory.
These devices have the attribute `cudaDevAttrDirectManagedMemAccessFromHost` set to 1.
Note that all hardware-coherent systems have this attribute set for NVLink-connected devices.
On these systems, the host has direct access to GPU-resident memory without page faults and
data migration. Note that with CUDA managed memory, the `cudaMemAdviseSetAccessedBy` hint with location type `cudaMemLocationTypeHost` is necessary
to enable this direct access without page faults, see example below.

**System Allocator**

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // for shared page table systems, the following hint is not necesary
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  free(ret);
}
```

**Managed**

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // set direct access hint

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  cudaFree(ret);
```

After `write` kernel is completed, `ret` will be created and initialized in GPU memory.
Next, the CPU will access `ret` followed by `append` kernel using the same `ret` memory again.
This code will show different behavior depending on the system architecture and support of hardware coherency:

- on systems with `directManagedMemAccessFromHost=1`:
CPU accesses to the managed buffer will not trigger any migrations;
the data will remain resident in GPU memory and any subsequent GPU kernels
can continue to access it directly without inflicting faults or migrations
- on systems with `directManagedMemAccessFromHost=0`:
CPU accesses to the managed buffer will page fault and initiate data migration;
any GPU kernel trying to access the same data first time will page fault and
migrate pages back to GPU memory.

#### [4.1.1.2.3. Host Native Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#host-native-atomics)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#host-native-atomics "Permalink to this headline")

Some devices, including NVLink-connected devices of hardware-coherent systems, support hardware-accelerated atomic accesses to CPU-resident memory. This implies that atomic accesses to host memory do not have to be emulated with a page fault. For these devices, the attribute `cudaDevAttrHostNativeAtomicSupported` is set to 1.

#### [4.1.1.2.4. Atomic Accesses and Synchronization Primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#atomic-accesses-and-synchronization-primitives)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#atomic-accesses-and-synchronization-primitives "Permalink to this headline")

CUDA unified memory supports all atomic operations available to host and device threads, enabling all threads to cooperate by concurrently accessing the same shared memory location. The [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) library provides many heterogeneous synchronization primitives tuned for concurrent use between host and device threads, including `cuda::atomic`, `cuda::atomic_ref`, `cuda::barrier`, `cuda::semaphore`, among many others.

On software-coherent systems, atomic accesses from the device to file-backed host memory are not supported. The following example code is valid on hardware-coherent systems but exhibits undefined behavior on other systems:

```cuda
#include <cuda/atomic>

#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  // this will be closed/deleted by default on exit
  FILE* tmp_file = tmpfile64();
  // need to allocate space in the file, we do this with posix_fallocate here
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // initialize the value in our file-backed memory
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device and host thread access ptr concurrently, using cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // this will always be 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}
```

On software-coherent systems, atomic accesses to unified memory may incur page faults which can lead to significant latencies. Note that this is not the case for all GPU atomics to CPU memory on these systems: operations listed by `nvidia-smi -q | grep "Atomic Caps Outbound"` may avoid page faults.

On hardware-coherent systems, atomics between host and device do not require page faults, but may still fault for other reasons that can cause any memory access to fault.

#### [4.1.1.2.5. Memcpy()/Memset() Behavior With Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memcpy-memset-behavior-with-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memcpy-memset-behavior-with-unified-memory "Permalink to this headline")

`cudaMemcpy*()` and `cudaMemset*()` accept any unified memory pointer as arguments.

For `cudaMemcpy*()`, the direction specified as `cudaMemcpyKind` is a performance hint, which can have a higher performance impact if any of the arguments is a unified memory pointer.

Thus, it is recommended to follow the following performance advice:

- When the physical location of unified memory is known, use an accurate `cudaMemcpyKind` hint.
- Prefer `cudaMemcpyDefault` over an inaccurate `cudaMemcpyKind` hint.
- Always use populated (initialized) buffers: avoid using these APIs to initialize memory.
- Avoid using `cudaMemcpy*()` if both pointers point to system-allocated memory:
launch a kernel or use a CPU memory copy algorithm such as `std::memcpy` instead.

#### [4.1.1.2.6. Overview of Memory Allocators for Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#overview-of-memory-allocators-for-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#overview-of-memory-allocators-for-unified-memory "Permalink to this headline")

For systems with full CUDA unified memory support various different allocators may be used to allocate unified memory.
The following table shows an overview of a selection of allocators with their respective features. Note that all information in this section is subject to change in future CUDA versions.

| API | Placement Policy | Accessible From | Migrate Based On Access [^[2]] | Page Sizes [^[4]] [^[5]] |
| --- | --- | --- | --- | --- |
| `malloc`, `new`,  `mmap` | First touch/hint [^[1]] | CPU, GPU | Yes [^[3]] | System or huge page size [^[6]] |
| `cudaMallocManaged` | First touch/hint | CPU, GPU | Yes | CPU resident: system page size GPU resident: 2MB |
| `cudaMalloc` | GPU | GPU | No | GPU page size: 2MB |
| `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister` | CPU | CPU, GPU | No | Mapped by CPU: system page size   Mapped by GPU: 2MB |
| Memory pools, location type host: `cuMemCreate`, `cudaMemPoolCreate` | CPU | CPU, GPU | No | Mapped by CPU: system page size   Mapped by GPU: 2MB |
| Memory pools, location type device: `cuMemCreate`, `cudaMemPoolCreate`, `cudaMallocAsync` | GPU | GPU | No | 2MB |

[^[1]]: For `mmap`, file-backed memory is placed on the CPU by default, unless specified otherwise through `cudaMemAdviseSetPreferredLocation` (or `mbind`, see bullet points below).

[^[2]]: This feature can be overridden with `cudaMemAdvise`. Even if access-based migrations are disabled, if the backing memory space is full, memory might migrate.

[^[3]]: File-backed memory will not migrate based on access.

[^[4]]: The default system page size is 4KiB or 64KiB on most systems, unless huge page size was explicitly specified (for example, with `mmap` `MAP_HUGETLB` / `MAP_HUGE_SHIFT`). In this case, any huge page size configured on the system is supported.

[^[5]]: Page-sizes for GPU-resident memory may evolve in future CUDA versions.

[^[6]]: Currently huge page sizes may not be kept when migrating memory to the GPU or placing it through first-touch on the GPU.

The table [Overview of unified memory support of different allocators](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-um-allocators) shows the difference in semantics of several allocators that may be considered to allocate data accessible from multiple processors at a time, including host and device.
For additional details about `cudaMemPoolCreate`, see the [Memory Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-pools) section, for additional details about `cuMemCreate`, see the [Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management) section.

On hardware-coherent systems where device memory is exposed as a NUMA domain to the system, special allocators such as `numa_alloc_on_node` may be used to pin memory to the given NUMA node, either host or device. This memory is accessible from both host and device and does not migrate. Similarly,
`mbind` can be used to pin memory to the given NUMA node(s), and can cause file-backed memory to be placed on the given NUMA node(s) before it is first accessed.

The following applies to allocators of memory that is shared:

- System allocators such as `mmap` allow sharing the memory between processes using the `MAP_SHARED` flag. This is supported in CUDA and can be used to share memory between different devices connected to the same host. However, this is currently not supported for sharing memory between multiple hosts as well as multiple devices. See [Inter-Process Communication (IPC) with Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-fork-managed-memory) for details.
- For access to unified memory or other CUDA memory through a network on multiple hosts, consult the documentation of the communication library used, for example [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), [NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html), [OpenMPI](https://www.open-mpi.org/faq/?category=runcuda), [UCX](https://docs.mellanox.com/category/hpcx), etc.

#### [4.1.1.2.7. Access Counter Migration](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#access-counter-migration)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#access-counter-migration "Permalink to this headline")

On hardware-coherent systems, the access counters feature keeps track of the frequency of access that a GPU makes to memory located on other processors. This is needed to ensure memory pages are moved to the physical memory of the processor that is accessing the pages most frequently. It can guide migrations between CPU and GPU, as well as between peer GPUs, a process called access counter migration.

Starting with CUDA 12.4, access counters are supported system-allocated memory. Note that file-backed memory does not migrate based on access. For system-allocated memory, access counters migration can be switched on by using the `cudaMemAdviseSetAccessedBy` hint to a device with the corresponding device id. If access counters are on, one can use  `cudaMemAdviseSetPreferredLocation` set to host to prevent migrations.  Per default `cudaMallocManaged` migrates based on a fault-and-migrate mechanism. [^[7]]

The driver may also use access counters for more efficient thrashing mitigation or memory oversubscription scenarios.

[^[7]]: Current systems allow the use of access-counter migration with managed memory when the accessed-by device hint is set. This is an implementation detail and should not be relied on for future compatibility.

#### [4.1.1.2.8. Avoid Frequent Writes to GPU-Resident Memory from the CPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#avoid-frequent-writes-to-gpu-resident-memory-from-the-cpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#avoid-frequent-writes-to-gpu-resident-memory-from-the-cpu "Permalink to this headline")

If the host accesses unified memory, cache misses may introduce more traffic than expected between host and device. Many CPU architectures require all memory operations to go through the cache hierarchy, including writes. If system memory is resident on the GPU, this means that frequent writes by the CPU to this memory can cause cache misses, thus transferring the data first from the GPU to CPU before writing the actual value into the requested memory range. On software-coherent systems, this may introduce additional page faults, while on hardware-coherent systems, it may cause higher latencies between CPU operations. Thus, in order to share data produced by the host with the device, consider writing to CPU-resident memory and reading the values directly from the device. The code below shows how to achieve this with unified memory.

**System Allocator**

```cuda
  size_t data_size = sizeof(int);
  int* data = (int*)malloc(data_size);
  // ensure that data stays local to the host and avoid faults
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  free(data);
```

**Managed**

```cuda
  int* data;
  size_t data_size = sizeof(int);
  cudaMallocManaged(&data, data_size);
  // ensure that data stays local to the host and avoid faults
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  cudaFree(data);
```

#### [4.1.1.2.9. Exploiting Asynchronous Access to System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#exploiting-asynchronous-access-to-system-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#exploiting-asynchronous-access-to-system-memory "Permalink to this headline")

If an application needs to share results from work on the device with the host, there are several possible options:

1. The device writes its result to GPU-resident memory, the result is transferred using `cudaMemcpy*`, and the host reads the transferred data.
2. The device directly writes its result to CPU-resident memory, and the host reads that data.
3. The device writes to GPU-resident memory, and the host directly accesses that data.

If independent work can be scheduled on the device while the result is transferred/accessed by the host, options 1 or 3 are preferred. If the device is starved until the host has accessed the result, option 2 might be preferred. This is because the device can generally write at a higher bandwidth than the host can read, unless many host threads are used to read the data.

**1. Explicit Copy**

```cuda
void exchange_explicit_copy(cudaStream_t stream) {
  int* data, *host_data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  host_data = (int*)malloc(n_bytes);
  // allocate, since we touch on the device first, will be GPU-resident
  cudaMallocManaged(&data, n_bytes);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the device
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // transfer to host
  cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
  cudaFree(data);
  free(host_data);
}
```

**2. Device Direct Write**

```cuda
void exchange_device_direct_write(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the host
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
  kernel<<<1, 16, 0, stream>>>(data);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

**3. Host Direct Read**

```cuda
void exchange_host_direct_read(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the device
  cudaMemLocation device_loc = {};
  cudaGetDevice(&device_loc.id);
  device_loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the GPU
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // sync stream to ensure data may be accessed (has been written by device)
  cudaStreamSynchronize(stream);
  // read data directly from host
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
```

Finally, in the Explicit Copy example above, instead of using `cudaMemcpy*` to transfer data, one could use a host or device kernel to perform this transfer explicitly. For contiguous data, using the CUDA copy-engines is preferred because operations performed by copy-engines can be overlapped with work on both the host and device. Copy-engines might be used in `cudaMemcpy*` and `cudaMemPrefetchAsync` APIs, but there is no guarantee. that copy-engines are used with `cudaMemcpy*` API calls. For the same reason, explicitly copy is preferred over direct host read for large enough data: if both host and device perform work that does not saturate their respective memory systems, the transfer can be performed by the copy-engines concurrently with the work performed by both host and device.

Copy-engines are generally used for both transfers between host and device as well as between peer devices within an NVLink-connected system.  Due to the limited total number of copy-engines, some systems may have a lower bandwidth of `cudaMemcpy*` compared to using the device to explicitly perform the transfer.  In such a case, if the transfer is in the critical path of the application, it may be preferred to use an explicit device-based transfer.
