---
title: "8.4.9. cuda_tile.trunci"
section: "8.4.9"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-trunci"
---

### [8.4.9. cuda_tile.trunci](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-trunci)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-trunci "Permalink to this headline")

_Truncates the width of an integer tile_

```default
cuda_tile.trunci %from %overflow
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id106)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id106 "Permalink to this headline")

- **from** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input integer tile to truncate. 13.1
- **overflow** ([IntegerOverflow](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-trunci-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id107)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id107 "Permalink to this headline")

- **to** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The truncated integer tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id108)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id108 "Permalink to this headline")

The `trunci` operation converts a tile of integers of a given element type to
one with a strictly smaller width.

The optional *overflow* attribute specifies whether an overflow can occur
when interpreting the operand as a signed and/or unsigned integer. In case
of “no signed wrap”, all truncated bits must have the same value as the
most significant bit of the truncated result. In case of “no unsigned
wrap”, the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation
respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id109)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id109 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
