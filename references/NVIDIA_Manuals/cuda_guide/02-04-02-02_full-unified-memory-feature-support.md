---
title: "2.4.2.2. Full Unified Memory Feature Support"
section: "2.4.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#full-unified-memory-feature-support"
---

### [2.4.2.2. Full Unified Memory Feature Support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#full-unified-memory-feature-support)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#full-unified-memory-feature-support "Permalink to this headline")

Most Linux systems have full unified memory support.  If device attribute `cudaDevAttrPageableMemoryAccess` is 1, then all system memory, whether allocated by CUDA APIs or system APIs, operates as unified memory with full feature support. This includes file-backed memory allocations created with `mmap`.

If `cudaDevAttrPageableMemoryAccess` is 0, then only memory allocated as managed memory by CUDA behaves as unified memory. Memory allocated with system APIs is not managed and is not necessarily accessible from GPU kernels.

In general, for unified allocations with full support:

- Managed memory is usually allocated in the memory space of the processor where it is first touched
- Managed memory is usually migrated when it is used by a processor other than the processor where it currently resides
- Managed memory is migrated or accessed at the granularity of memory pages (software coherence) or cache lines (hardware coherence)
- Oversubscription is allowed: an application may allocate more managed memory than is physically available on the GPU

Allocation and migration behavior can deviate from the above. This can by influenced the programmer using [hints and prefetches](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-mem-advise-prefetch).  Full coverage of full unified memory support can be found in [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-pageable-systems).

#### [2.4.2.2.1. Full Unified Memory with Hardware Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#full-unified-memory-with-hardware-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#full-unified-memory-with-hardware-coherency "Permalink to this headline")

On hardware such as Grace Hopper and Grace Blackwell, where an NVIDIA CPU is used and the interconnect between the CPU and GPU is NVLink Chip-to-Chip (C2C), address translation services (ATS) are available. `cudaDevAttrPageableMemoryAccessUsesHostPageTables` is 1 when ATS is available.

With ATS, in addition to full unified memory support for all host allocations:

- GPU allocations (e.g. `cudaMalloc`) can be accessed from the CPU (`cudaDevAttrDirectManagedMemAccessFromHost` will be 1)
- The link between CPU and GPU supports native atomics (`cudaDevAttrHostNativeAtomicSupported` will be 1)
- Hardware support for coherence can improve performance compared to software coherence

ATS provides all capabilities of [HMM](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management). When ATS is available, HMM is automatically disabled. Further discussion of hardware vs. software coherency is found in [CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-hw-coherency).

#### [2.4.2.2.2. HMM - Full Unified Memory with Software Coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#hmm-full-unified-memory-with-software-coherency)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#hmm-full-unified-memory-with-software-coherency "Permalink to this headline")

_Heterogeneous Memory Management_ (HMM) is a feature available on Linux operating systems (with appropriate kernel versions) which enables software-coherent [full unified memory support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory-full). Heterogeneous memory management brings some of the capabilities and convenience provided by ATS to PCIe-connected GPUs.

On Linux with at least Linux Kernel 6.1.24, 6.2.11, or 6.3 or later, heterogeneous memory management (HMM) may be available. The following command can be used to find if the addressing mode is `HMM`.

```c++
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```

When HMM is available, [full unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory-full) is supported and all system allocations are implicitly unified memory. If a system also has [ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services), HMM is disabled and ATS is used, since ATS provides all the capabilities of HMM and more.
