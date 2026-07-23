---
title: "8.7.6. cuda_tile.ceil"
section: "8.7.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-ceil"
---

### [8.7.6. cuda_tile.ceil](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ceil)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ceil "Permalink to this headline")

_Element-wise ceiling_

```default
cuda_tile.ceil %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id180)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id180 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input float tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id181)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id181 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The ceiling of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id182)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id182 "Permalink to this headline")

The `ceil` operation computes the element-wise ceiling on the input
floating-point tile. The ceiling operation rounds each element up to the
largest integer value that is greater than or equal to the input value.

$$
\[\text{ceil}(x)_i = \min\{n \in \mathbb{Z} \mid n \geq x_i\}\]
$$

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id183)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id183 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id184)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id184 "Permalink to this headline")

```mlir
%result = ceil %source : tile<f32>
```

See [cuda_tile.ceil_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-ceil-0) for the full example listing.
