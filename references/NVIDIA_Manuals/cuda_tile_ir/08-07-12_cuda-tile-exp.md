---
title: "8.7.12. cuda_tile.exp"
section: "8.7.12"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-exp"
---

### [8.7.12. cuda_tile.exp](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-exp)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-exp "Permalink to this headline")

_Element-wise exponential_

```default
cuda_tile.exp %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id208)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id208 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input float tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id209)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id209 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The exponential of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id210)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id210 "Permalink to this headline")

The `exp` operation computes the element-wise exponential of the input
floating-point tile.

$$
\[\text{exp}(x)_i = e^{x_i}\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id211)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id211 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id212)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id212 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = exp %in : tile<4xf32>
```

See [cuda_tile.exp_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-exp-0) for the full example listing.
