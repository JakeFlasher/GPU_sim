---
title: "8.7.11. cuda_tile.exp2"
section: "8.7.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-exp2"
---

### [8.7.11. cuda_tile.exp2](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-exp2)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-exp2 "Permalink to this headline")

_Element-wise power of two_

```default
cuda_tile.exp2 %source %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id203)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id203 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id204)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id204 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of raising 2 to the power of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id205)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id205 "Permalink to this headline")

The `exp2` operation computes the element-wise power of two of the input
floating-point tile.

$$
\[\text{exp2}(x)_i = 2^{x_i}\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

The below table shows the supported modifiers for each data type.

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
| --- | --- | --- | --- | --- |
| `flush_to_zero` | yes | no | no | no |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id206)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id206 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id207)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id207 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = exp2 %in : tile<4xf32>
```

See [cuda_tile.exp2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-exp2-0) for the full example listing.
