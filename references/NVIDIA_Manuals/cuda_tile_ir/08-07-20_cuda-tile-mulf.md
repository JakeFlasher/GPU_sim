---
title: "8.7.20. cuda_tile.mulf"
section: "8.7.20"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-mulf"
---

### [8.7.20. cuda_tile.mulf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mulf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mulf "Permalink to this headline")

_Element-wise floating-point multiplication_

```default
cuda_tile.mulf %lhs %rhs %rounding_mode %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id246)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id246 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-mulf-roundingmode-attr)) - The rounding mode for the operation. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id247)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id247 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The product of the input tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id248)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id248 "Permalink to this headline")

The `mulf` operation computes the element-wise product between the two input tiles with
with floating-point element types.

If the `flush_to_zero` modifier is specified, denormal numbers are flushed to positive zero.

If the `rounding` modifier is specified, the particular rounding mode will be applied to each
element of the result.

$$
\[\text{mulf}(x, y)_i = x_i \times y_i\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

The below table shows the supported modifiers and rounding modes for each data type. Entries with ‘*’ are emulated in f32.

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
| --- | --- | --- | --- | --- |
| `flush_to_zero` | yes | no | no | no |
| `rounding<nearest_even>` | yes | yes | yes | yes |
| `rounding<zero>` | yes | yes | yes* | yes* |
| `rounding<negative_inf>` | yes | yes | yes* | yes* |
| `rounding<positive_inf>` | yes | yes | yes* | yes* |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id249)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id249 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
