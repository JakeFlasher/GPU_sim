---
title: "8.7.14. cuda_tile.fma"
section: "8.7.14"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-fma"
---

### [8.7.14. cuda_tile.fma](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-fma)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-fma "Permalink to this headline")

_Floating point fused multipy-add_

```default
cuda_tile.fma %lhs %rhs %acc %rounding_mode %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id218)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id218 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **acc** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The accumulator operand. 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-fma-roundingmode-attr)) - The rounding mode for the operation. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id219)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id219 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The fused multiply-add of the input tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id220)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id220 "Permalink to this headline")

Takes three operands `lhs`, `rhs` and `acc`, returns `result = lhs * rhs + acc`.

$$
\[\text{fma}(x, y, z)_i = x_i \times y_i + z_i\]
$$

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id221)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id221 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs`, `acc` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
