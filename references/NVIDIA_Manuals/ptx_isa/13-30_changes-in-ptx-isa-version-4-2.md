---
title: "13.30. Changes in PTX ISA Version 4.2"
section: "13.30"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#changes-in-ptx-isa-version-4-2"
---

## [13.30. Changes in PTX ISA Version 4.2](https://docs.nvidia.com/cuda/parallel-thread-execution#changes-in-ptx-isa-version-4-2)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#changes-in-ptx-isa-version-4-2 "Permalink to this headline")

**New Features**

PTX ISA version 4.2 introduces the following new features:

- Support for `sm_53` target architecture.
- Support for arithmetic, comparsion and texture instructions for `.f16` and `.f16x2` types.
- Support for `memory_layout` field for surfaces and `suq` instruction support for querying this
field.

**Semantic Changes and Clarifications**

Semantics for parameter passing under ABI were updated to indicate `ld.param` and `st.param`
instructions used for argument passing cannot be predicated.

Semantics of `{atom/red}.add.f32` were updated to indicate subnormal inputs and results are
flushed to sign-preserving zero for atomic operations on global memory; whereas atomic operations on
shared memory preserve subnormal inputs and results and don’t flush them to zero.
