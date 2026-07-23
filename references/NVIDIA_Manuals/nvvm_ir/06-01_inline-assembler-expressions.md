---
title: "6.1. Inline Assembler Expressions"
section: "6.1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#inline-assembler-expressions"
---

## [6.1. Inline Assembler Expressions](https://docs.nvidia.com/cuda/nvvm-ir-spec#inline-assembler-expressions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#inline-assembler-expressions "Permalink to this headline")

Inline assembler of PTX instructions is supported, with the following supported constraints:

| Constraint | Type |
| --- | --- |
| c | i8 |
| h | i16 |
| r | i32 |
| l | i64 |
| f | f32 |
| d | f64 |

The inline asm metadata `!srcloc` is accepted and ignored.

The inline asm dialect `inteldialect` is not supported.
