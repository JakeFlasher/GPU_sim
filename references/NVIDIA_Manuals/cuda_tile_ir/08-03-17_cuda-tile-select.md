---
title: "8.3.17. cuda_tile.select"
section: "8.3.17"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-select"
---

### [8.3.17. cuda_tile.select](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-select)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-select "Permalink to this headline")

_Select values based on condition_

```default
cuda_tile.select %cond %val_if_true %val_if_false
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id70)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id70 "Permalink to this headline")

- **cond** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The condition tile. 13.1
- **val_if_true** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The value if true tile. 13.1
- **val_if_false** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The value if false tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id71)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id71 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The tile of selected values. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id72)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id72 "Permalink to this headline")

The `select` op chooses values based on the binary conditions supplied as
the `cond` operand. The `val_if_true` operand contains the value(s) to use
if the condition is 1. The `val_if_false` operand contains the value(s) to
use if the condition is 0. The choice is made element-wise according to the
values in the condition tile.

$$
\[\begin{split}\text{select}(\text{cond}, x, y)_i = \begin{cases}
  x_i & \text{if } \text{cond}_i = 1 \\
  y_i & \text{if } \text{cond}_i = 0
\end{cases}\end{split}\]
$$

All tiles must have the same shape. The tiles `val_if_true`,
`val_if_false`, and the result must have the same element type. The `cond`
tile must be a tile of `i1` values.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id73)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id73 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `val_if_true`, `val_if_false` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)).
- Operation must infer result types from operands and attributes.
