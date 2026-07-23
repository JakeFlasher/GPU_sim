---
title: "10.3. Special Registers: %laneid"
section: "10.3"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-laneid"
---

## [10.3. Special Registers: %laneid](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-laneid)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-laneid "Permalink to this headline")

**`%laneid`**

Lane Identifier.

**Syntax (predefined)**

```text
.sreg .u32 %laneid;
```

**Description**

A predefined, read-only special register that returns the thread’s lane within the warp. The lane
identifier ranges from zero to `WARP_SZ-1`.

**PTX ISA Notes**

Introduced in PTX ISA version 1.3.

**Target ISA Notes**

Supported on all target architectures.

**Examples**

```text
mov.u32  %r, %laneid;
```
