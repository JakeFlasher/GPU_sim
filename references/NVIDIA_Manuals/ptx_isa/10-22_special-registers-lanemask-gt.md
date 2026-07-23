---
title: "10.22. Special Registers: %lanemask_gt"
section: "10.22"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-gt"
---

## [10.22. Special Registers: %lanemask_gt](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-lanemask-gt)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-gt "Permalink to this headline")

**`%lanemask_gt`**

32-bit mask with bits set in positions greater than the thread’s lane number in the warp.

**Syntax (predefined)**

```text
.sreg .u32 %lanemask_gt;
```

**Description**

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
greater than the thread’s lane number in the warp.

**PTX ISA Notes**

Introduced in PTX ISA version 2.0.

**Target ISA Notes**

`%lanemask_gt` requires `sm_20` or higher.

**Examples**

```text
mov.u32     %r, %lanemask_gt;
```
