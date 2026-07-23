---
title: "10.17. Special Registers: %cluster_nctarank"
section: "10.17"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-nctarank"
---

## [10.17. Special Registers: %cluster_nctarank](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-cluster-nctarank)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-nctarank "Permalink to this headline")

**`%cluster_nctarank`**

Number of CTA identifiers in a cluster across all dimensions.

**Syntax (predefined)**

```text
.sreg .u32 %cluster_nctarank;
```

**Description**

A predefined, read-only special register initialized with the nunber of CTAs within a cluster across
all dimensions.

**PTX ISA Notes**

Introduced in PTX ISA version 7.8.

**Target ISA Notes**

Requires `sm_90` or higher.

**Examples**

```text
.reg .b32 %r;

mov.u32  %r, %cluster_nctarank;
```
