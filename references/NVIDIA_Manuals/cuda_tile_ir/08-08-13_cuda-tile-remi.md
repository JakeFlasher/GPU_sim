---
title: "8.8.13. cuda_tile.remi"
section: "8.8.13"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-remi"
---

### [8.8.13. cuda_tile.remi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-remi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-remi "Permalink to this headline")

_Element-wise integer remainder_

```default
cuda_tile.remi %lhs %rhs %signedness
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id345)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id345 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-remi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id346)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id346 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The remainder after division. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id347)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id347 "Permalink to this headline")

The `remi` operation computes the element-wise remainder of the input tiles
with integer element types using truncated division (rounding towards zero).
Division by zero is undefined behavior.

$$
\[\text{remi}(x, y)_i = x_i - \text{trunc}(x_i / y_i) \times y_i\]
$$

If the operation is signed, the sign of the result matches the sign
of the dividend (`lhs`). For example:

- `remi(7, 3) = 1`
- `remi(7, -3) = 1`
- `remi(-7, 3) = -1`
- `remi(-7, -3) = -1`

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id348)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id348 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `result`, `lhs` and `rhs` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
