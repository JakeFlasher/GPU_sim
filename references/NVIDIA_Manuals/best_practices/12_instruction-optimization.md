---
title: "12. Instruction Optimization"
section: "12"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#instruction-optimization"
---

# [12. Instruction Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#instruction-optimization)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#instruction-optimization "Permalink to this headline")

Awareness of how instructions are executed often permits low-level optimizations that can be useful, especially in code that is run frequently (the so-called hot spot in a program). Best practices suggest that this optimization be performed after all higher-level optimizations have been completed.
