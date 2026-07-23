---
title: "10.31. Special Registers: %aggr_smem_size"
section: "10.31"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-aggr-smem-size"
---

## [10.31. Special Registers: %aggr_smem_size](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-aggr-smem-size)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-aggr-smem-size "Permalink to this headline")

**`%aggr_smem_size`**

Total size of shared memory used by a CTA of a kernel.

**Syntax (predefined)**

```text
.sreg .u32 %aggr_smem_size;
```

**Description**

A predefined, read-only special register initialized with total aggregated size of shared memory
consisting of the size of user shared memory allocated (statically and dynamically) at launch time
and the size of shared memory region which is reserved for the NVIDIA system software use.

**PTX ISA Notes**

Introduced in PTX ISA version 8.1.

**Target ISA Notes**

Requires `sm_90` or higher.

**Examples**

```text
mov.u32  %r, %aggr_smem_size;
```
