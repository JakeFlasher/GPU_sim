---
title: "13.2. Changes in PTX ISA Version 9.1"
section: "13.2"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#changes-in-ptx-isa-version-9-1"
---

## [13.2. Changes in PTX ISA Version 9.1](https://docs.nvidia.com/cuda/parallel-thread-execution#changes-in-ptx-isa-version-9-1)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#changes-in-ptx-isa-version-9-1 "Permalink to this headline")

**New Features**

PTX ISA version 9.1 introduces the following new features:

- Adds support for `.volatile` qualifier with `.local` state space for `ld` and
`st` instructions.
- Adds support for `.f16x2` and `.bf16x2` source types for `cvt` instruction
with destination types `.e2m1x2`, `.e2m3x2`, `.e3m2x2`, `.e4m3x2`, `.e5m2x2`.
- Adds support for `.scale_vec::4X` with `.ue8m0` as `.stype` with `.kind::mxf4nvf4` for
`mma`/`mma.sp` instructions.
- Adds support for `.s2f6x2` instruction type for `cvt` instruction.
- Adds support for `multimem.cp.async.bulk` and `multimem.cp.reduce.async.bulk` instructions.

**Semantic Changes and Clarifications**

None.
