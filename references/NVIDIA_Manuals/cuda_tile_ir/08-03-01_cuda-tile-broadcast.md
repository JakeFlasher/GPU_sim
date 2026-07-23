---
title: "8.3.1. cuda_tile.broadcast"
section: "8.3.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-broadcast"
---

### [8.3.1. cuda_tile.broadcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-broadcast)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-broadcast "Permalink to this headline")

_Broadcast tile to new shape_

```default
cuda_tile.broadcast %source
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#parameters)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#parameters "Permalink to this headline")

- **source** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The tile to broadcast. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#results)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#results "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The broadcasted tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#description)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#description "Permalink to this headline")

The `broadcast` operation expands each unary (`1`) dimension in the input tile
by duplicating the data along that dimension.

Expansion happens only for dimensions of size one that are stretched or “copied” to match
the size of the dimension implied by the result type of the operation. The operation
does not change the rank of the source tile.  Any change to the rank of the source tile
must be made using reshape-like operations before broadcasting.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#constraints)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#constraints "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `source` and `result` must have the same element type (tile).
- `source` and `result` must have the same rank.
