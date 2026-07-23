---
title: "1.3. PTX ISA Version 9.2"
section: "1.3"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-isa-version-9-2"
---

## [1.3. PTX ISA Version 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution#ptx-isa-version-9-2)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-isa-version-9-2 "Permalink to this headline")

PTX ISA version 9.2 introduces the following new features:

- Adds support for `.u8x4` and `.s8x4` instruction types for `add`, `sub`, `min`, `max`,
`neg` instructions.
- Adds support for `add.sat.{u16x2/s16x2/u32}` instruction.
- Adds support for `.b128` type for `st.async` instruction.
- Adds support for `.ignore_oob` qualifier for `cp.async.bulk` instruction.
- Adds support for `.bf16x2` destination type for `cvt` instruction with `.e4m3x2`,
`.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` source types.
