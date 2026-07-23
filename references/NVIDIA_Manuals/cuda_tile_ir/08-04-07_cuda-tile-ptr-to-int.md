---
title: "8.4.7. cuda_tile.ptr_to_int"
section: "8.4.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-ptr-to-int"
---

### [8.4.7. cuda_tile.ptr_to_int](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ptr-to-int)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ptr-to-int "Permalink to this headline")

_Convert a tile of pointers to a tile of integers_

```default
cuda_tile.ptr_to_int %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id98)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id98 "Permalink to this headline")

- **source** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The input tile of pointers. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id99)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id99 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The output tile of integers. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id100)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id100 "Permalink to this headline")

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The result values should be interpreted as unsigned integers.

The inverse of this operation is [cuda_tile.int_to_ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-int-to-ptr).

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id101)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id101 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
