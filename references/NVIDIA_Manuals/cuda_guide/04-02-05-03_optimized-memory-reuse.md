---
title: "4.2.5.3. Optimized Memory Reuse"
section: "4.2.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#optimized-memory-reuse"
---

### [4.2.5.3. Optimized Memory Reuse](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#optimized-memory-reuse)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#optimized-memory-reuse "Permalink to this headline")

CUDA reuses memory in two ways:

- Virtual and physical memory reuse within a graph is based on virtual address assignment, like in the stream ordered allocator.
- Physical memory reuse between graphs is done with virtual aliasing: different graphs can map the same physical memory to their unique virtual addresses.

#### [4.2.5.3.1. Address Reuse within a Graph](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#address-reuse-within-a-graph)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#address-reuse-within-a-graph "Permalink to this headline")

CUDA may reuse memory within a graph by assigning the same virtual address ranges to different allocations whose lifetimes do not overlap. Since virtual addresses may be reused, pointers to different allocations with disjoint lifetimes are not guaranteed to be unique.

The following figure shows adding a new allocation node (2) that can reuse the address freed by a dependent node (1).

![Adding New Alloc Node 2](images/_________-______-______1.png)

Figure 28 Adding New Alloc Node 2[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id9 "Link to this image")

The following figure shows adding a new alloc node (4). The new alloc node is not dependent on the free node (2) so cannot reuse the address from the associated alloc node (2). If the alloc node (2) used the address freed by free node (1), the new alloc node 3 would need a new address.

![Adding New Alloc Node 3](images/_________-______-______2.png)

Figure 29 Adding New Alloc Node 3[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id10 "Link to this image")

#### [4.2.5.3.2. Physical Memory Management and Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#physical-memory-management-and-sharing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#physical-memory-management-and-sharing "Permalink to this headline")

CUDA is responsible for mapping physical memory to the virtual address before the allocating node is reached in GPU order. As an optimization for memory footprint and mapping overhead, multiple graphs may use the same physical memory for distinct allocations if they will not run simultaneously; however, physical pages cannot be reused if they are bound to more than one executing graph at the same time, or to a graph allocation which remains unfreed.

CUDA may update physical memory mappings at any time during graph instantiation, launch, or execution. CUDA may also introduce synchronization between future graph launches in order to prevent live graph allocations from referring to the same physical memory. As for any allocate-free-allocate pattern, if a program accesses a pointer outside of an allocation’s lifetime, the erroneous access may silently read or write live data owned by another allocation (even if the virtual address of the allocation is unique). Use of compute sanitizer tools can catch this error.

The following figure shows graphs sequentially launched in the same stream. In this example, each graph frees all the memory it allocates. Since the graphs in the same stream never run concurrently, CUDA can and should use the same physical memory to satisfy all the allocations.

![Sequentially Launched Graphs](images/_________-______-______3.png)

Figure 30 Sequentially Launched Graphs[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id11 "Link to this image")
