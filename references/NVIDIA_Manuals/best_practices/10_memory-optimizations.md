---
title: "10. Memory Optimizations"
section: "10"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations"
---

# [10. Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#memory-optimizations)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations "Permalink to this headline")

Memory optimizations are the most important area for performance. The goal is to maximize the use of the hardware by maximizing bandwidth. Bandwidth is best served by using as much fast memory and as little slow-access memory as possible. This chapter discusses the various kinds of memory on the host and device and how best to set up data items to use the memory effectively.
