---
title: "8.7.3. cuda_tile.absf"
section: "8.7.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-absf"
---

### [8.7.3. cuda_tile.absf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-absf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-absf "Permalink to this headline")

_Element-wise floating-point absolute value_

```default
cuda_tile.absf %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id167)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id167 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input float tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id168)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id168 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The absolute value of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id169)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id169 "Permalink to this headline")

The `absf` operation computes the element-wise absolute value of the input float tile.

$$
\[\text{absf}(x)_i = |x|_i\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id170)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id170 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
