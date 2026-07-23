---
title: "8.4.6. cuda_tile.int_to_ptr"
section: "8.4.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-int-to-ptr"
---

### [8.4.6. cuda_tile.int_to_ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-int-to-ptr)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-int-to-ptr "Permalink to this headline")

_Convert a tile of integers to a tile of pointers_

```default
cuda_tile.int_to_ptr %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id94)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id94 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input tile of integers. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id95)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id95 "Permalink to this headline")

- **result** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The output tile of pointers. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id96)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id96 "Permalink to this headline")

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The `source` operand is interpreted as an unsigned integer.

The inverse of this operation is [cuda_tile.ptr_to_int](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-ptr-to-int).

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id97)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id97 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
