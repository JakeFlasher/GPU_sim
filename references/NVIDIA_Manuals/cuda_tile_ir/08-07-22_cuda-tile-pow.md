---
title: "8.7.22. cuda_tile.pow"
section: "8.7.22"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-pow"
---

### [8.7.22. cuda_tile.pow](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-pow)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-pow "Permalink to this headline")

_Element-wise floating-point exponentiation_

```default
cuda_tile.pow %source %exponent
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id255)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id255 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The base tile. 13.1
- **exponent** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The exponent tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id256)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id256 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the pow operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id257)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id257 "Permalink to this headline")

The `pow` operation computes the element-wise exponentiation of the source floating-point tile raised to the power
of the exponent floating-point tile.

$$
\[\text{pow}(x, y)_i = x_i^{y_i}\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id258)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id258 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `result`, `source` and `exponent` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- `source`, `exponent` and `result` must have the same rank.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id259)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id259 "Permalink to this headline")

```mlir
%source = constant <f32: 0.0> : tile<4xf32>
%exponent = constant <f32: 2.0> : tile<4xf32>
%result = pow %source, %exponent : tile<4xf32>
```

See [cuda_tile.pow_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-pow-0) for the full example listing.
