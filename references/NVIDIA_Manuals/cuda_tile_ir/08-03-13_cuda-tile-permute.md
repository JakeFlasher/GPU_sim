---
title: "8.3.13. cuda_tile.permute"
section: "8.3.13"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-permute"
---

### [8.3.13. cuda_tile.permute](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-permute)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-permute "Permalink to this headline")

_Permute tile dimensions_

```default
cuda_tile.permute %source %permutation
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id50)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id50 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The input tile. 13.1
- **permutation** ([Array](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-array)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The permutation of the dimensions. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id51)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id51 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The permuted tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id52)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id52 "Permalink to this headline")

Permute the dimensions of the input tile `source` according to the `permutation` array.
The `permutation` array is a list of integers that specify the new order of the dimensions.

For example, if the input tile has shape `[2, 4, 8]`, and the permutation is `[2, 0, 1]`,
the output tile will have shape `[8, 2, 4]`.

This operation logically is a change in the indexing of the tile.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id53)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id53 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)).
- `source` and `result` must have the same rank.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id54)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id54 "Permalink to this headline")

```mlir
%arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
%0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
```

See [cuda_tile.permute_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-permute-0) for the full example listing.
