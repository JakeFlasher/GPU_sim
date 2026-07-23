---
title: "8.7.15. cuda_tile.log2"
section: "8.7.15"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-log2"
---

### [8.7.15. cuda_tile.log2](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-log2)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-log2 "Permalink to this headline")

_Element-wise base-2 logarithm_

```default
cuda_tile.log2 %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id222)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id222 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id223)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id223 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the log2 operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id224)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id224 "Permalink to this headline")

The `log2` operation computes the element-wise base-2 logarithm
of a floating-point tile.

$$
\[\text{log2}(x)_i = \log_2(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id225)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id225 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id226)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id226 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = log2 %in : tile<4xf32>
```

See [cuda_tile.log2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-log2-0) for the full example listing.
