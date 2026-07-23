---
title: "8.11.2. cuda_tile.get_tensor_shape"
section: "8.11.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-get-tensor-shape"
---

### [8.11.2. cuda_tile.get_tensor_shape](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-tensor-shape)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-tensor-shape "Permalink to this headline")

_Query the shape of a tensor view_

```default
cuda_tile.get_tensor_shape %src
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id385)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id385 "Permalink to this headline")

- **src** ([tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view)) - The source tensor view. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id386)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id386 "Permalink to this headline")

- **result** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>>) - The shape of the tensor, each value representing the size of the corresponding dimension. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id387)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id387 "Permalink to this headline")

The `get_tensor_shape` operation returns the shape of the tensor
backing the provided tensor view.

The result values should be interpreted as unsigned integers.

> **Warning**
>
> If the tensor dimensions do not fit in the result tile’s element type
> the behavior is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id388)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id388 "Permalink to this headline")

- Operation must not perform any memory side effects.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id389)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id389 "Permalink to this headline")

```mlir
%dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
```

See [cuda_tile.get_tensor_shape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-get-tensor-shape-0) for the full example listing.
