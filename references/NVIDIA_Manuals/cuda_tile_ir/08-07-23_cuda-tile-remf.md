---
title: "8.7.23. cuda_tile.remf"
section: "8.7.23"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-remf"
---

### [8.7.23. cuda_tile.remf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-remf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-remf "Permalink to this headline")

_Element-wise floating-point remainder_

```default
cuda_tile.remf %lhs %rhs
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id260)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id260 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id261)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id261 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The remainder after division. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id262)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id262 "Permalink to this headline")

The `remf` operation computes the element-wise floating-point remainder using
truncated division (rounding towards zero).

$$
\[\text{remf}(x, y)_i = x_i - \text{trunc}(x_i / y_i) \times y_i\]
$$

The result has the same sign as the dividend (`lhs`) and its magnitude is
less than the magnitude of divisor (`rhs`).

**Special cases:**

- If `y` is zero, returns `NaN`
- If `x` is infinite and `y` is finite, returns `NaN`
- If `x` is finite and `y` is infinite, returns `x`
- If either argument is `NaN`, returns `NaN`

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id263)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id263 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
