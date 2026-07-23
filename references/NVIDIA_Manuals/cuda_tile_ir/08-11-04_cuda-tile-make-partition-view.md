---
title: "8.11.4. cuda_tile.make_partition_view"
section: "8.11.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-make-partition-view"
---

### [8.11.4. cuda_tile.make_partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-make-partition-view)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-make-partition-view "Permalink to this headline")

_Create a partition view from a tensor view_

```default
cuda_tile.make_partition_view %tensor_view
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id395)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id395 "Permalink to this headline")

- **tensor_view** ([tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view)) - The source tensor view to create a partition view from. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id396)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id396 "Permalink to this headline")

- **result** ([partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view)) - The created partition view. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id397)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id397 "Permalink to this headline")

The `make_partition_view` operation creates a [partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) from a
[tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view). For more details about partition views see [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view).

The operation uses the type constraints of the input tensor view and the annotated return type
to perform the partitioning. The tensor view’s type contains its physical layout in the form
of shapes and strides and the partition view contains the logical size of a single tile.

The resulting partition view can be loaded from using [cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-view-tko) and
stored to using [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-view-tko).

The view memory options act on the computed index space of the partition view, see
[Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) and [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) for detailed semantics.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id398)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id398 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id399)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id399 "Permalink to this headline")

```mlir
%tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
  : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view.
make_partition_view %tensor_view0 :
  partition_view<
    tile=(1024x1x32),
    tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
  >

%s0 = constant <i32: 8192> : tile<i32>
%str0 = constant <i32: 524288> : tile<i32>

%tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
  : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view. The provided tensor_view has a
// dynamically-sized dimension.
make_partition_view %tensor_view1 :
  partition_view<tile=(1024x1x32), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
```

See [cuda_tile.make_partition_view_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-make-partition-view-0) for the full example listing.
