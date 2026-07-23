---
title: "10.16. Special Registers: %cluster_ctarank"
section: "10.16"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctarank"
---

## [10.16. Special Registers: %cluster_ctarank](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-cluster-ctarank)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctarank "Permalink to this headline")

**`%cluster_ctarank`**

CTA identifier in a cluster across all dimensions.

**Syntax (predefined)**

```text
.sreg .u32 %cluster_ctarank;
```

**Description**

A predefined, read-only special register initialized with the CTA rank within a cluster across all
dimensions.

It is guaranteed that:

```text
0  <=  %cluster_ctarank <  %cluster_nctarank
```

**PTX ISA Notes**

Introduced in PTX ISA version 7.8.

**Target ISA Notes**

Requires `sm_90` or higher.

**Examples**

```text
.reg .b32 %r;

mov.u32  %r, %cluster_ctarank;
```
