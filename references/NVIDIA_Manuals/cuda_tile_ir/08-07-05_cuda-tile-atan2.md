---
title: "8.7.5. cuda_tile.atan2"
section: "8.7.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-atan2"
---

### [8.7.5. cuda_tile.atan2](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-atan2)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-atan2 "Permalink to this headline")

_Element-wise atan2_

```default
cuda_tile.atan2 %x %y
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id175)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id175 "Permalink to this headline")

- **x** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input x float tile. 13.2
- **y** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input y float tile. 13.2

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id176)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id176 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The element-wise result tile. 13.2

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id177)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id177 "Permalink to this headline")

The `atan2` operation calculates the principal value
of the arc tangent of the ratio of first and second input
arguments x / y. The quadrant of the result is determined
by the signs of inputs x and y.

$$
\[(\operatorname{atan2}(x, y))_i = \mathrm{atan2}(x_i, y_i)\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id178)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id178 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `x`, `y` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id179)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id179 "Permalink to this headline")

```mlir
%x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
%y = constant <f32: [1.0,  1.0, 1.0, 0.0]> : tile<4xf32>
%res = atan2 %x, %y : tile<4xf32>
```

See [cuda_tile.atan2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-atan2-0) for the full example listing.
