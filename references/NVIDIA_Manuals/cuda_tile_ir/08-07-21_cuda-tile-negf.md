---
title: "8.7.21. cuda_tile.negf"
section: "8.7.21"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-negf"
---

### [8.7.21. cuda_tile.negf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-negf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-negf "Permalink to this headline")

_Element-wise floating-point negation_

```default
cuda_tile.negf %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id250)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id250 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id251)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id251 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The negated floating-point tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id252)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id252 "Permalink to this headline")

`negf` is an element-wise operation that negates the sign of `source`.

$$
\[\text{negf}(x)_i = -x_i\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id253)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id253 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id254)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id254 "Permalink to this headline")

```mlir
%source = constant <f32: 0.0> : tile<4xf32>
%result = negf %source : tile<4xf32>
```

See [cuda_tile.negf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-negf-0) for the full example listing.
