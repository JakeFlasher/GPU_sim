---
title: "8.3.15. cuda_tile.reshape"
section: "8.3.15"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-reshape"
---

### [8.3.15. cuda_tile.reshape](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reshape)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reshape "Permalink to this headline")

_Reshape tile dimensions_

```default
cuda_tile.reshape %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id60)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id60 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The source tile to reshape. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id61)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id61 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The reshaped tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id62)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id62 "Permalink to this headline")

The `reshape` operation changes the shape of the `source` operand. `reshape` is
only a change in the indexing of the tile. The number of elements and element type
must remain unchanged.

0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception
where a 0-d tile can be reshaped to shape where the `size(shape) == 1`.

Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming
a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id63)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id63 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same element type (tile).

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id64)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id64 "Permalink to this headline")

```mlir
%cst = constant <i8: 0> : tile<i8>
%0 = reshape %cst
    : tile<i8> -> tile<1x1x1xi8>

%t = constant <f32: 0.0> : tile<8x2xf32>
%1 = reshape %t
    : tile<8x2xf32> -> tile<2x2x4x1xf32>
```

See [cuda_tile.reshape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-reshape-0) for the full example listing.

```mlir
  %cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]>
      : tile<2x4xi32>
  %r0 = reshape %cst
: tile<2x4xi32> -> tile<2x2x2xi32>

// Step 1: Turn source into 1D tile. Use row-major by convention.
// %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
%tmp = reshape %cst
    : tile<2x4xi32> -> tile<8xi32>

// Step 2: Turn 1D tile into result tile. Use row-major by convention.
// %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
%r1 =  reshape %tmp
        : tile<8xi32> -> tile<2x2x2xi32>
```

See [cuda_tile.reshape_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-reshape-1) for the full example listing.
