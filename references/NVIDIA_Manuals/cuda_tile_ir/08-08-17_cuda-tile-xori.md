---
title: "8.8.17. cuda_tile.xori"
section: "8.8.17"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-xori"
---

### [8.8.17. cuda_tile.xori](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-xori)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-xori "Permalink to this headline")

_Element-wise bitwise XOR_

```default
cuda_tile.xori %lhs %rhs
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id361)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id361 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id362)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id362 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The bitwise XOR of the input tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id363)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id363 "Permalink to this headline")

The `xori` operation computes the element-wise bitwise exclusive or (XOR)
of two tile values with integer element types.

$$
\[\text{xori}(x, y)_i = x_i \oplus y_i\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id364)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id364 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id365)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id365 "Permalink to this headline")

```mlir
%lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
%rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
// This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
// are tiles of shape `4xi32`, and returns the result as `%result`.
%result = xori %lhs, %rhs : tile<4xi32>
```

See [cuda_tile.xori_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-xori-0) for the full example listing.
