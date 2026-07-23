---
title: "8.3.7. cuda_tile.get_num_tile_blocks"
section: "8.3.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-get-num-tile-blocks"
---

### [8.3.7. cuda_tile.get_num_tile_blocks](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-num-tile-blocks)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-num-tile-blocks "Permalink to this headline")

_Get total number of tile blocks_

```default
cuda_tile.get_num_tile_blocks
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id24)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id24 "Permalink to this headline")

No parameters.

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id25)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id25 "Permalink to this headline")

- **gridSize_x** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The number of tile blocks in dimension `x`. 13.1
- **gridSize_y** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The number of tile blocks in dimension `y`. 13.1
- **gridSize_z** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The number of tile blocks in dimension `z`. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id26)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id26 "Permalink to this headline")

The `get_num_tile_blocks` operation queries the total number of tile blocks
in the form of a 3-tuple specifying the extent of each grid dimension.

A tile `id` is a coordinate in 3-space and therefore the must also be a 3-tuple containing
the extent of each dimension: `x`, `y` and `z`.

When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

For example if the grid used to launch the kernel is `(1024, 1024)` then the
result of this operation will be `(1024, 1024, 1)`.

> **Note**
>
> **Grid Dimension Limitation**: Grid dimensions are limited to 2^24-1 (16,777,215)
> per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
> kernel launches for larger workloads.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id27)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id27 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id28)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id28 "Permalink to this headline")

```mlir
entry @example() {
  %x, %y, %z = get_num_tile_blocks : tile<i32>
  // print "x: %, y: %, z: %\n", %x, %y, %z : tile<i32>, tile<i32>, tile<i32>
}
```

See [cuda_tile.get_num_tile_blocks_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-get-num-tile-blocks-0) for the full example listing.
