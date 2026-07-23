---
title: "8.4.1. cuda_tile.bitcast"
section: "8.4.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-bitcast"
---

### [8.4.1. cuda_tile.bitcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-bitcast)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-bitcast "Permalink to this headline")

_Bitcast a tile from one element type to another_

```default
cuda_tile.bitcast %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id74)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id74 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The source tile to cast. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id75)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id75 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The casted tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id76)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id76 "Permalink to this headline")

The `bitcast` operation casts the input tile from one element type to
another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`).
Pointer types must use [cuda_tile.ptr_to_int](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-ptr-to-int) or [cuda_tile.int_to_ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-int-to-ptr) instead.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id77)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id77 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
