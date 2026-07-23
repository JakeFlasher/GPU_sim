---
title: "8.8.15. cuda_tile.shri"
section: "8.8.15"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-shri"
---

### [8.8.15. cuda_tile.shri](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-shri)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-shri "Permalink to this headline")

_Element-wise shift-right_

```default
cuda_tile.shri %lhs %rhs %signedness
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id353)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id353 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand (shift amount). 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-shri-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id354)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id354 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the right shift operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id355)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id355 "Permalink to this headline")

The `shri` operation computes the element-wise right shift of the `lhs` integer operand by
the value of the `rhs` operand for tiles with integer element types.

$$
\[\text{shri}(x, y)_i = x_i \gg y_i\]
$$

When `unsigned`, higher-order bits
are zero-filled; when `signed`, the higher-order bits are filled with
the sign bit.

The `rhs` operand is always interpreted as an unsigned integer.

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id356)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id356 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
