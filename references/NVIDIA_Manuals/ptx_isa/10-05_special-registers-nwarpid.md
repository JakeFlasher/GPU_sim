---
title: "10.5. Special Registers: %nwarpid"
section: "10.5"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nwarpid"
---

## [10.5. Special Registers: %nwarpid](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-nwarpid)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nwarpid "Permalink to this headline")

**`%nwarpid`**

Number of warp identifiers.

**Syntax (predefined)**

```text
.sreg .u32 %nwarpid;
```

**Description**

A predefined, read-only special register that returns the maximum number of warp identifiers.

**PTX ISA Notes**

Introduced in PTX ISA version 2.0.

**Target ISA Notes**

`%nwarpid` requires `sm_20` or higher.

**Examples**

```text
mov.u32  %r, %nwarpid;
```
