---
title: "8.4.8. cuda_tile.ptr_to_ptr"
section: "8.4.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-ptr-to-ptr"
---

### [8.4.8. cuda_tile.ptr_to_ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ptr-to-ptr)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ptr-to-ptr "Permalink to this headline")

_Reinterpret a tile of one pointer type as another_

```default
cuda_tile.ptr_to_ptr %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id102)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id102 "Permalink to this headline")

- **source** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - Tile with source pointer element type. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id103)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id103 "Permalink to this headline")

- **result** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - Tile with target pointer element type. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id104)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id104 "Permalink to this headline")

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another
element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use [cuda_tile.ptr_to_int](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-ptr-to-int) or [cuda_tile.int_to_ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-int-to-ptr).
These operations are distinct to enable future compiler reasoning about pointer provenance.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id105)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id105 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
