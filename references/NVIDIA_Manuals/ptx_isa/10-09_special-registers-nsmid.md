---
title: "10.9. Special Registers: %nsmid"
section: "10.9"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid"
---

## [10.9. Special Registers: %nsmid](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-nsmid)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid "Permalink to this headline")

**`%nsmid`**

Number of SM identifiers.

**Syntax (predefined)**

```text
.sreg .u32 %nsmid;
```

**Description**

A predefined, read-only special register that returns the maximum number of SM identifiers. The SM
identifier numbering is not guaranteed to be contiguous, so `%nsmid` may be larger than the
physical number of SMs in the device.

**PTX ISA Notes**

Introduced in PTX ISA version 2.0.

**Target ISA Notes**

`%nsmid` requires `sm_20` or higher.

**Examples**

```text
mov.u32  %r, %nsmid;
```
