---
title: "8.8.5. cuda_tile.divi"
section: "8.8.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-divi"
---

### [8.8.5. cuda_tile.divi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-divi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-divi "Permalink to this headline")

_Element-wise integer division_

```default
cuda_tile.divi %lhs %rhs %signedness %rounding
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id308)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id308 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-divi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1
- **rounding** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-divi-roundingmode-attr)) - Set the rounding direction (implementing floordiv/ceildiv). 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id309)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id309 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the division. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id310)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id310 "Permalink to this headline")

The `divi` operation computes the element-wise division of two tile values with integer element type.

The default rounding is towards zero. The rounding mode can be set to *positive_inf* (“ceiling division”),
or *negative_inf* (“floor division”), other values are illegal.

The use of the rounding flag *negative_inf* with *unsigned* is not a valid combination.

If the *unsigned* flag is provided, the operands are treated as unsigned integers, otherwise they are
treated as signed integers.

The behavior is undefined if the right hand side is zero. A signed division overflow (minimum value
divided by -1) is undefined behavior.

$$
\[\text{div(lhs, rhs)}_i = \text{lhs}_i / \text{rhs}_i\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id311)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id311 "Permalink to this headline")

- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
