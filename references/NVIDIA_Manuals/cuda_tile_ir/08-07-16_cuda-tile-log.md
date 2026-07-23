---
title: "8.7.16. cuda_tile.log"
section: "8.7.16"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-log"
---

### [8.7.16. cuda_tile.log](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-log)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-log "Permalink to this headline")

_Element-wise natural logarithm_

```default
cuda_tile.log %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id227)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id227 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id228)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id228 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the log operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id229)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id229 "Permalink to this headline")

The `log` operation computes the element-wise natural logarithm of a
floating-point tile.

$$
\[\text{log}(x)_i = \ln(x_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id230)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id230 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
