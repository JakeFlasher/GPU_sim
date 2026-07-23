---
title: "8.3. State spaces"
section: "8.3"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-state-spaces"
---

## [8.3. State spaces](https://docs.nvidia.com/cuda/parallel-thread-execution#memory-consistency-state-spaces)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-state-spaces "Permalink to this headline")

The relations defined in the memory consistency model are independent of state spaces. In
particular, causality order closes over all memory operations across all the state spaces. But the
side-effect of a memory operation in one state space can be observed directly only by operations
that also have access to the same state space. This further constrains the synchronizing effect of a
memory operation in addition to scope. For example, the synchronizing effect of the PTX instruction
`ld.relaxed.shared.sys` is identical to that of `ld.relaxed.shared.cluster`, since no thread
outside the same cluster can execute an operation that accesses the same memory location.
