---
title: "4.2.5.4. Performance Considerations"
section: "4.2.5.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs--performance-considerations"
---

### [4.2.5.4. Performance Considerations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#performance-considerations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#performance-considerations "Permalink to this headline")

When multiple graphs are launched into the same stream, CUDA attempts to allocate the same physical memory to them because the execution of these graphs cannot overlap. Physical mappings for a graph are retained between launches as an optimization to avoid the cost of remapping. If, at a later time, one of the graphs is launched such that its execution may overlap with the others (for example if it is launched into a different stream) then CUDA must perform some remapping because concurrent graphs require distinct memory to avoid data corruption.

In general, remapping of graph memory in CUDA is likely caused by these operations:

- Changing the stream into which a graph is launched
- A trim operation on the graph memory pool, which explicitly frees unused memory (discussed in [graph-memory-nodes-physical-memory-footprint](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-graph-memory-nodes-physical-memory-footprint))
- Relaunching a graph while an unfreed allocation from another graph is mapped to the same memory will cause a remap of memory before relaunch

Remapping must happen in execution order, but after any previous execution of that graph is complete (otherwise memory that is still in use could be unmapped). Due to this ordering dependency, as well as because mapping operations are OS calls, mapping operations can be relatively expensive. Applications can avoid this cost by launching graphs containing allocation memory nodes consistently into the same stream.

#### [4.2.5.4.1. First Launch / cudaGraphUpload](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#first-launch-cudagraphupload)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#first-launch-cudagraphupload "Permalink to this headline")

Physical memory cannot be allocated or mapped during graph instantiation because the stream in which the graph will execute is unknown. Mapping is done instead during graph launch. Calling `cudaGraphUpload` can separate out the cost of allocation from the launch by performing all mappings for that graph immediately and associating the graph with the upload stream. If the graph is then launched into the same stream, it will launch without any additional remapping.

Using different streams for graph upload and graph launch behaves similarly to switching streams, likely resulting in remap operations. In addition, unrelated memory pool management is permitted to pull memory from an idle stream, which could negate the impact of the uploads.
