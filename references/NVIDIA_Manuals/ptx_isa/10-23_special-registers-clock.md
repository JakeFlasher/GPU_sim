---
title: "10.23. Special Registers: %clock, %clock_hi"
section: "10.23"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock"
---

## [10.23. Special Registers: %clock, %clock_hi](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-clock)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock "Permalink to this headline")

**`%clock`, `%clock_hi`**

**`%clock`**

  A predefined, read-only 32-bit unsigned cycle counter.

**`%clock_hi`**

  The upper 32-bits of `%clock64` special register.

**Syntax (predefined)**

```text
.sreg .u32 %clock;
.sreg .u32 %clock_hi;
```

**Description**

Special register `%clock` and `%clock_hi` are unsigned 32-bit read-only cycle counters that wrap
silently.

**PTX ISA Notes**

`%clock` introduced in PTX ISA version 1.0.

`%clock_hi` introduced in PTX ISA version 5.0.

**Target ISA Notes**

`%clock` supported on all target architectures.

`%clock_hi` requires `sm_20` or higher.

**Examples**

```text
mov.u32 r1,%clock;
mov.u32 r2, %clock_hi;
```
