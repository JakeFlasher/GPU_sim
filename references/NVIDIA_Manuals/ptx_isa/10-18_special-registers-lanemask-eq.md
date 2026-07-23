---
title: "10.18. Special Registers: %lanemask_eq"
section: "10.18"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-eq"
---

## [10.18. Special Registers: %lanemask_eq](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-lanemask-eq)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-eq "Permalink to this headline")

**`%lanemask_eq`**

32-bit mask with bit set in position equal to the thread’s lane number in the warp.

**Syntax (predefined)**

```text
.sreg .u32 %lanemask_eq;
```

**Description**

A predefined, read-only special register initialized with a 32-bit mask with a bit set in the
position equal to the thread’s lane number in the warp.

**PTX ISA Notes**

Introduced in PTX ISA version 2.0.

**Target ISA Notes**

`%lanemask_eq` requires `sm_20` or higher.

**Examples**

```text
mov.u32     %r, %lanemask_eq;
```
