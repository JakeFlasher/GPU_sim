---
title: "8.3.10. cuda_tile.iota"
section: "8.3.10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-iota"
---

### [8.3.10. cuda_tile.iota](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-iota)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-iota "Permalink to this headline")

_Generate a 1-d tile range from 0 to n-1_

```default
cuda_tile.iota
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id38)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id38 "Permalink to this headline")

No parameters.

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id39)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id39 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the iota operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id40)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id40 "Permalink to this headline")

The `iota` operation generates a 1-d tile with a sequence of integer
values. The starting value is `0` and the stride is `1`. If the shape of
the result tile is `(n)`, then the generated values are `[0, n - 1]`.

$$
\[\text{iota}(n)_i = i \quad \text{for } i \in [0, n-1]\]
$$

The result values should be interpreted as unsigned integers.

> **Note**
>
> The number of elements in the result tile must not exceed
> the maximum value that the element type can express.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id41)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id41 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
