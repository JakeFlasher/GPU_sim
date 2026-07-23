---
title: "4.17.2.2. Single-Node, Multi-GPU"
section: "4.17.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#single-node-multi-gpu"
---

### [4.17.2.2. Single-Node, Multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#single-node-multi-gpu)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#single-node-multi-gpu "Permalink to this headline")

In a multi-GPU system, the user has to provide host information for
the placement. As we mentioned, a natural way to express that
information would be by using NUMA node IDs and EGM follows this
approach. Therefore, using the `cuDeviceGetAttribute` function the
user should be able to learn the closest NUMA node id. (See [Socket Identifiers: What are they? How to access them?](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#socket-identifiers-what-are-they-how-to-access-them)).
Then the user can allocate and manage EGM memory using VMM (Virtual
Memory Management) API or CUDA Memory Pool.

#### [4.17.2.2.1. Using VMM APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-vmm-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-vmm-apis "Permalink to this headline")

The first step in memory allocation using Virtual Memory Management APIs
is to create a physical memory chunk that will provide a backing for the
allocation. See CUDA Programming Guide’s [Virtual Memory Management section](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#virtual-memory-management)
for more details. In EGM allocations the user has to explicitly provide
`CU_MEM_LOCATION_TYPE_HOST_NUMA`  as the location type and
*numaID* as the location identifier. Also in EGM, allocations must
be aligned to appropriate granularity of the platform. The following
code snippet shows allocating physical memory with `cuMemCreate`:

```c++
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

After physical memory allocation, we have to reserve an address space
and map it to a pointer. These procedures do not have EGM-specific
changes:

```c++
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

Finally, the user has to explicitly protect mapped virtual address
ranges. Otherwise access to the mapped space would result in a crash.
Similar to the memory allocation, the user has to provide
`CU_MEM_LOCATION_TYPE_HOST_NUMA` as the location type and
*numaId* as the location identifier. Following code snippet create
an access descriptors for the host node and the GPU to give read and
write access for the mapped memory to both of them:

```c++
CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);
```

#### [4.17.2.2.2. Using CUDA Memory Pool](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-cuda-memory-pool)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-cuda-memory-pool "Permalink to this headline")

To define EGM, the user can create a memory pool on a node and give
access to peers. In this case, the user has to explicitly define
`cudaMemLocationTypeHostNuma` as the location type and *numaId*
as the location identifier. The following code snippet shows creating a
memory pool `cudaMemPoolCreate`:

```c++
cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);
```

Additionally, for direct connect peer access, it is also possible to use
the existing peer access API, `cudaMemPoolSetAccess`. An example
for an *accessingDevice* is shown in the following code snippet:

```c++
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);
```

When the memory pool is created, and accesses are given, the user can
set created memory pool to the *residentDevice* and start allocating
memory using `cudaMallocAsync`:

```c++
cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);
```

> **Note**
>
> EGM is mapped with 2MB pages. Therefore, users may encounter more TLB
> misses when accessing very large allocations.
