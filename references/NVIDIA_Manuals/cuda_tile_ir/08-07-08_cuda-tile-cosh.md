---
title: "8.7.8. cuda_tile.cosh"
section: "8.7.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-cosh"
---

### [8.7.8. cuda_tile.cosh](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cosh)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cosh "Permalink to this headline")

_Element-wise hyperbolic cosine_

```default
cuda_tile.cosh %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id190)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id190 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id191)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id191 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The hyperbolic cosine of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id192)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id192 "Permalink to this headline")

The `cosh` operation computes the element-wise hyperbolic cosine of the
input tile with floating-point element type.

$$
\[\text{cosh}(x)_i = {\cosh x}_i\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id193)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id193 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
