---
title: "8.7.30. cuda_tile.tan"
section: "8.7.30"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-tan"
---

### [8.7.30. cuda_tile.tan](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-tan)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-tan "Permalink to this headline")

_Element-wise tangent_

```default
cuda_tile.tan %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id291)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id291 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id292)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id292 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The tangent of the input floating-point tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id293)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id293 "Permalink to this headline")

The `tan` operation computes the element-wise tangent of
the input floating-point tile.

$$
\[\text{tan}(x)_i = \tan(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id294)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id294 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
