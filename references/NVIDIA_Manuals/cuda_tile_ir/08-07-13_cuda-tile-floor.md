---
title: "8.7.13. cuda_tile.floor"
section: "8.7.13"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-floor"
---

### [8.7.13. cuda_tile.floor](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-floor)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-floor "Permalink to this headline")

_Element-wise floor rounding_

```default
cuda_tile.floor %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id213)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id213 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input tile to the floor operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id214)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id214 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the floor operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id215)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id215 "Permalink to this headline")

The `floor` operation computes the element-wise floor on the input floating-point tile
rounding each element down to the largest integer that is less than or equal to the element.

$$
\[\text{floor}_i(x_i) = \max\{n \in \mathbb{Z} \mid n \leq x_i\}\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id216)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id216 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id217)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id217 "Permalink to this headline")

```mlir
%source = constant <f32: 1.5> : tile<f32>
%result = floor %source : tile<f32>
```

See [cuda_tile.floor_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-floor-0) for the full example listing.
