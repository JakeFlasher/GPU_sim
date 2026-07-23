---
title: "8.7.24. cuda_tile.rsqrt"
section: "8.7.24"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-rsqrt"
---

### [8.7.24. cuda_tile.rsqrt](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-rsqrt)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-rsqrt "Permalink to this headline")

_Element-wise reciprocal square root_

```default
cuda_tile.rsqrt %source %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id264)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id264 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input tile to compute the reciprocal square root of. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id265)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id265 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The reciprocal square root of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id266)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id266 "Permalink to this headline")

The `rsqrt` operation computes the element-wise reciprocal square root
of the input floating-point tile.

This operation supports: `flush_to_zero`: if set by the user,
will flush subnormal inputs and results to sign-preserving zero.

$$
\[\text{rsqrt}(x)_i = \frac{1}{\sqrt{x_i}}\]
$$

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See [Floating-Point Math](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-math) for more details.

The below table shows the supported modifiers for each data type.

| Modifier | Float32 | Float64 |
| --- | --- | --- |
| `flush_to_zero` | yes | no |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id267)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id267 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id268)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id268 "Permalink to this headline")

```mlir
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = rsqrt %in : tile<4xf32>

// Rsqrt op with flush to zero modifier
%ftz_res = rsqrt %in flush_to_zero : tile<4xf32>
```

See [cuda_tile.rsqrt_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-rsqrt-0) for the full example listing.
