---
title: "8.7.25. cuda_tile.sinh"
section: "8.7.25"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-sinh"
---

### [8.7.25. cuda_tile.sinh](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-sinh)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-sinh "Permalink to this headline")

_Element-wise hyperbolic sine_

```default
cuda_tile.sinh %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id269)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id269 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input float tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id270)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id270 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The hyperbolic sine of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id271)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id271 "Permalink to this headline")

The `sinh` operation computes the element-wise hyperbolic sine of the input
floating-point tile.

$$
\[\text{sinh}(x)_i = \sinh(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id272)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id272 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
