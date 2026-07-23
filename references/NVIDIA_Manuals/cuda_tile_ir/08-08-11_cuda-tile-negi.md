---
title: "8.8.11. cuda_tile.negi"
section: "8.8.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-negi"
---

### [8.8.11. cuda_tile.negi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-negi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-negi "Permalink to this headline")

_Element-wise integer negation_

```default
cuda_tile.negi %source %overflow
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id336)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id336 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input integer tile. 13.1
- **overflow** ([IntegerOverflow](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-negi-integeroverflow-attr)) - The overflow behavior of the operation. 13.2

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id337)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id337 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The negated integer tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id338)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id338 "Permalink to this headline")

The `negi` operation computes the element-wise negation of the input integer tile.
The input and output tiles are always interpreted as signed integers.

$$
\[\text{negi}(x_i) = -x_i\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation
respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id339)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id339 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id340)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id340 "Permalink to this headline")

```mlir
%source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
%result = negi %source : tile<4xi16>
// %result = [0, -1, -2, -3]
```

See [cuda_tile.negi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-negi-0) for the full example listing.
