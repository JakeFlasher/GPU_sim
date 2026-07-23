---
title: "8.3.2. cuda_tile.cat"
section: "8.3.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-cat"
---

### [8.3.2. cuda_tile.cat](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cat)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cat "Permalink to this headline")

_Concatenate tiles along specified dimension_

```default
cuda_tile.cat %lhs %rhs %dim
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id1)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id1 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The right hand side operand. 13.1
- **dim** ([i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)) - The dimension along which to concatenate. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id2)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id2 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The concatenated result tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id3)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id3 "Permalink to this headline")

The `cat` operation concatenates the two input tiles. The input tiles must have the same shape
in all but the concatenating dimension. Concatenation happens along the dimension specified by the
the attribute `dim` the resulting dimension is the sum of the the two input tiles concatenating
dimension.

$$
\[\begin{split}\text{cat}(x, y, dim_{cat})[ \vec{i} ] =
  \begin{cases}
    x[..., i_{cat}, ..., i_n] & \text{if } i_{cat} < d_{cat} \\
    y[..., i_{cat} - d_{cat}, ..., i_n] & \text{if } i_{cat} \geq d_{cat}
  \end{cases}\end{split}\]
$$

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id4)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id4 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same rank.
- `lhs`, `rhs` and `result` must have the same element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)).

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#examples)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#examples "Permalink to this headline")

```mlir
// A valid invocation of cat.
%0 = cat %arg0, %arg1 dim = 1
  : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
// >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
//                [ D, E, F, 4, 5, 6 ]])

// A valid invocation of cat.
%1 = cat %arg0, %arg1 dim = 0
  : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
//
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
//
// >>> %1 = tile([[ A, B, C ],
//                [ D, E, F ],
//                [ 1, 2, 3 ],
//                [ 4, 5, 6 ]])
```

See [cuda_tile.cat_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-cat-0) for the full example listing.
