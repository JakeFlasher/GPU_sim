---
title: "8.11.3. cuda_tile.load_view_tko"
section: "8.11.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-load-view-tko"
---

### [8.11.3. cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-load-view-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-load-view-tko "Permalink to this headline")

_Load a tile from a tile view_

```default
cuda_tile.load_view_tko %memory_ordering_semantics %memory_scope %view %index %token %optimization_hints
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id390)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id390 "Permalink to this headline")

- **memory_ordering_semantics** ([MemoryOrderingSemantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-view-tko-memoryorderingsemantics-attr)) - The memory ordering semantics for the load operation. 13.1
- **memory_scope** ([MemoryScope](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-view-tko-memoryscope-attr)) - The memory scope for the atomic operation. 13.1
- **view** ([view_type](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-views)) - The view from which the tile will be loaded. 13.1
- **index** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>>) - The n-dimensional index of the desired element to load from the view. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The optional token for the load operation. 13.1
- **optimization_hints** ([OptimizationHints](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-view-tko-optimizationhints-attr)) - Optimization hints for operation 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id391)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id391 "Permalink to this headline")

- **tile** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The loaded tile. 13.1
- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id392)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id392 "Permalink to this headline")

The `load_view_tko` operation loads a tile from a tile view.

A view is mapping from view-space indices to a particular element in the view, each
view type has a defined mapping from view-space indices to tiles produced from elements
of the view.

For example, the [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) partitions a [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) into
a grid of equally sized tiles. The view indexes one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view’s index
space. The space of valid indices depends on which view is passed to the operation.
For example the index space of a [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) is equal to the
rank of the partitioned tiles.

The `index` operands are interpreted as unsigned integers.

Out of bounds accesses are handled according to the semantics of [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view).

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.
For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location.
For more information, refer to the [memory model section](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model) of the specification.

- `weak` - No concurrent accesses to the source/destination location.
- `relaxed` - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- `acquire` -  There may be concurrent accesses to the location. If this acquire observes a release operation, then _happens before_ is established.

Note: The following variants are not supported by this operation: `release`, `acq_rel`.

The `memory_scope` attribute specifies a communication scope for memory operations.
When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other
threads which are participating in the communication, or data races may occur.

- `tl_blk` - There may be concurrent accesses from within the same tile block.
- `device` - There may be concurrent accesses from within the same device (i.e., GPU).
- `sys` - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify
specific hints for each operation.

- `num_cta_in_cga` - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for [cuda_tile.entry](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-entry).
- `allow_tma` - suggest whether to use TMA for [cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-view-tko) and [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-view-tko).
- `latency` - latency hint for [cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-view-tko) and [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-view-tko).

For example they can be annotated as:

```mlir
optimization_hints=<
  sm_100 = {num_cta_in_cga = 8},
  sm_120 = {num_cta_in_cga = 16}
>
```

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id393)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id393 "Permalink to this headline")

- Operation must encode variadic operand segment sizes in attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id394)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id394 "Permalink to this headline")

```mlir
%tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
  : tensor_view<8192x128xf32, strides=[128,1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.

%view = make_partition_view %tensor_view : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i32: 1> : tile<i32>

// Load a tile at index (0, 0) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[0,64), in the coordinates of tiles.
%tile0, %res_token0 = load_view_tko weak %view[%c0, %c0]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Load a tile at index (0, 1) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[64,128), in the coordinates of tiles.
%tile1, %res_token1 = load_view_tko weak %view[%c0, %c1]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Same example as above but with memory token as input.
%token = make_token : token
%tile2, %res_token2 = load_view_tko weak %view[%c0, %c1] token = %token
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Loads a tile at the dynamic index (%index, %index) in the view's index space.
%tile3, %res_token3 = load_view_tko weak %view[%index, %index]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token
```

See [cuda_tile.load_view_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-load-view-tko-0) for the full example listing.
