---
title: "8.8.2. cuda_tile.absi"
section: "8.8.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-absi"
---

### [8.8.2. cuda_tile.absi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-absi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-absi "Permalink to this headline")

_Element-wise integer absolute value_

```default
cuda_tile.absi %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id295)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id295 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input integer tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id296)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id296 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The absolute value of the input tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id297)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id297 "Permalink to this headline")

The `absi` operation computes the absolute value of the input integer tile.

The input tile is always interpreted as a signed integer.
The output tile is always interpreted as an unsigned integer.

$$
\[\text{absi}(x) = |x|\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id298)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id298 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same shape.
- `source` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
