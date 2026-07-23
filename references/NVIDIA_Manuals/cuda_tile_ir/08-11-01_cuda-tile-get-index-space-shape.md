---
title: "8.11.1. cuda_tile.get_index_space_shape"
section: "8.11.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-get-index-space-shape"
---

### [8.11.1. cuda_tile.get_index_space_shape](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-index-space-shape)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-index-space-shape "Permalink to this headline")

_Query the index space dimension size_

```default
cuda_tile.get_index_space_shape %src
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id380)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id380 "Permalink to this headline")

- **src** ([view_type](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-views)) - The source view type. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id381)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id381 "Permalink to this headline")

- ****result** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>>) - The shape of the index space, each value representing the size of the**
  corresponding dimension. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id382)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id382 "Permalink to this headline")

The `get_index_space_shape` operation returns the shape of the index
space of `src`.

The result tile has the same rank as the view’s index space with the elements
representing the size of the corresponding dimension.

The result values should be interpreted as unsigned integers.

> **Warning**
>
> If the individual index space dimension do not fit in the result tile’s element type
> the behavior is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id383)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id383 "Permalink to this headline")

- Operation must not perform any memory side effects.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id384)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id384 "Permalink to this headline")

```mlir
%tensor_view = make_tensor_view %base,
    shape = [2, 2, 4], strides = [2, 2, 1]
    : tensor_view<2x2x4xf32, strides=[2,2,1]>
%partition_view = make_partition_view %tensor_view :
  partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>>
%dim0, %dim1, %dim2 = get_index_space_shape %partition_view :
  partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>> -> tile<i64>
```

See [cuda_tile.get_index_space_shape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-get-index-space-shape-0) for the full example listing.
