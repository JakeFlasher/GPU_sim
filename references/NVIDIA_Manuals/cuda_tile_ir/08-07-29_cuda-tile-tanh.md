---
title: "8.7.29. cuda_tile.tanh"
section: "8.7.29"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-tanh"
---

### [8.7.29. cuda_tile.tanh](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-tanh)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-tanh "Permalink to this headline")

_Element-wise hyperbolic tangent_

```default
cuda_tile.tanh %source %rounding_mode
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id286)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id286 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-tanh-roundingmode-attr)) - The rounding mode for the operation. 13.2

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id287)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id287 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The hyperbolic tangent of the input floating-point tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id288)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id288 "Permalink to this headline")

The `tanh` operation computes the element-wise hyperbolic tangent of the
input floating-point tile. Default rounding mode is *full*.

The `approx` rounding mode implements a fast approximation to hyperbolic tangent.
Subnormal results of this fast approximation are not flushed to zero.

The `full` rounding mode implements a relatively fast full-range approximation.
The maximum ulp error is 2 across the full range of inputs in FP32 and 1 in FP64.

$$
\[\text{tanh}(x)_i = \tanh(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

The below table shows the supported modifiers for each data type. Entries with ‘*’ are emulated in f32.

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
| --- | --- | --- | --- | --- |
| `approx` | yes | no | no | no |
| `full` | yes | yes | yes* | yes* |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id289)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id289 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id290)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id290 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res0 = tanh %in : tile<4xf32>

// tanh with approx modifier
%res1 = tanh %in rounding<approx> : tile<4xf32>
```

See [cuda_tile.tanh_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-tanh-0) for the full example listing.
