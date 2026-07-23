---
title: "8.7.26. cuda_tile.sin"
section: "8.7.26"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-sin"
---

### [8.7.26. cuda_tile.sin](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-sin)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-sin "Permalink to this headline")

_Element-wise sine_

```default
cuda_tile.sin %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id273)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id273 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input float tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id274)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id274 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The sine of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id275)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id275 "Permalink to this headline")

The `sin` operation computes the element-wise sine of the input floating-point tile.

$$
\[\text{sin}(x)_i = \sin(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id276)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id276 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id277)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id277 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = sin %in : tile<4xf32>
```

See [cuda_tile.sin_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-sin-0) for the full example listing.
