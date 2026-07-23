---
title: "8.1. Scope and applicability of the model"
section: "8.1"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#scope-and-applicability"
---

## [8.1. Scope and applicability of the model](https://docs.nvidia.com/cuda/parallel-thread-execution#scope-and-applicability)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#scope-and-applicability "Permalink to this headline")

The constraints specified under this model apply to PTX programs with any PTX ISA version number,
running on `sm_70` or later architectures.

The memory consistency model does not apply to texture (including `ld.global.nc`) and surface
accesses.

### [8.1.1. Limitations on atomicity at system scope](https://docs.nvidia.com/cuda/parallel-thread-execution#limitations-system-scope-atomicity)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#limitations-system-scope-atomicity "Permalink to this headline")

When communicating with the host CPU, certain strong operations with system scope may not be
performed atomically on some systems. For more details on atomicity guarantees to host memory, see
the _CUDA Atomicity Requirements_.
