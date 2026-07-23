---
title: "8.3.5. cuda_tile.extract"
section: "8.3.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-extract"
---

### [8.3.5. cuda_tile.extract](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-extract)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-extract "Permalink to this headline")

_Extract a subtile from a tile_

```default
cuda_tile.extract %source %indices
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id14)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id14 "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The source tile to extract from. 13.1
- **indices** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>>) - The indices of the slice to extract. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id15)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id15 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The extracted subtile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id16)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id16 "Permalink to this headline")

The `extract` operation extracts a subtile from the given source tile.

The shape of the result tile must divide the shape of the source tile
evenly e.g., `tile<4xf32>` is a valid extraction from `tile<8xf32>`, but
`tile<3xf32>` is not.

The `$indices` indicate the number of the slice to extract, but _importantly_ not the offsets
used to construct the subtile for extraction. The semantics of extract means that only
full size slices can be extracted.

Slices of a source tile with the same shape are non-overlapping by definition for
unique indices.

The `indices` operands are interpreted as unsigned integers.

> **Warning**
>
> If the `indices` specify a non-existent (i.e., out-of-bounds) slice, the
> behavior of the operation is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id17)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id17 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same rank.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id18)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id18 "Permalink to this headline")

```mlir
// Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
%c1 = constant <i32: 1> : tile<i32>
%c2 = constant <i32: 2> : tile<i32>
%t = constant <f32: 0.0> : tile<32x8xf32>
// Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
%0 = extract %t[%c1, %c2]
    : tile<32x8xf32> -> tile<4x2xf32>
```

See [cuda_tile.extract_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-extract-0) for the full example listing.
