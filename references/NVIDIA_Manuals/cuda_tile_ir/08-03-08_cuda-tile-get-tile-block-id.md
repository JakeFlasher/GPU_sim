---
title: "8.3.8. cuda_tile.get_tile_block_id"
section: "8.3.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-get-tile-block-id"
---

### [8.3.8. cuda_tile.get_tile_block_id](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-tile-block-id)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-tile-block-id "Permalink to this headline")

_Get the currently executing tile block coordinates_

```default
cuda_tile.get_tile_block_id
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id29)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id29 "Permalink to this headline")

No parameters.

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id30)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id30 "Permalink to this headline")

- **blockId_x** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The tile block ID for dimension `x`. 13.1
- **blockId_y** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The tile block ID for dimension `y`. 13.1
- **blockId_z** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The tile block ID for dimension `z`. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id31)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id31 "Permalink to this headline")

`get_tile_block_id` returns a 3-d tile block coordinates (or ID) of the currently
executing tile block.

A tile ID has three dimensions: `x`, `y`, and `z`. This operation returns all
three of them simultaneously. The value of each dimension returned by this
operation is between `0` (including) and the value returned by `get_num_tile_blocks`
for the respective axis (excluding), represented by the inclusive interval
`[0, get_num_tile_blocks(dim) - 1]` . Grid dimensions unspecified at kernel
launch (i.e., a 1-d or 2-d grid) will always be `0` for all tile blocks.

> **Note**
>
> **Grid Dimension Limitation**: Grid dimensions are limited to 2^24-1 (16,777,215)
> per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
> kernel launches for larger workloads.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id32)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id32 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.
