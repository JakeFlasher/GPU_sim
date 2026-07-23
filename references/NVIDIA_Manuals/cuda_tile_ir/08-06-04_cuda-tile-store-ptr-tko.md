---
title: "8.6.4. cuda_tile.store_ptr_tko"
section: "8.6.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-store-ptr-tko"
---

### [8.6.4. cuda_tile.store_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-store-ptr-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-store-ptr-tko "Permalink to this headline")

_Store and scatter data from pointer of tile to global memory without ordering guarantees_

```default
cuda_tile.store_ptr_tko %memory_ordering_semantics %memory_scope %destination %value %mask %token %optimization_hints
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id163)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id163 "Permalink to this headline")

- **memory_ordering_semantics** ([MemoryOrderingSemantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-store-ptr-tko-memoryorderingsemantics-attr)) - The memory ordering semantics. 13.1
- **memory_scope** ([MemoryScope](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-store-ptr-tko-memoryscope-attr)) - The optional memory scope. 13.1
- **destination** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The destination pointer tile. 13.1
- **value** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The value tile to store. 13.1
- **mask** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The optional mask for selective storage. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The optional token for operation ordering. 13.1
- **optimization_hints** ([OptimizationHints](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-store-ptr-tko-optimizationhints-attr)) - Optimization hints for operation 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id164)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id164 "Permalink to this headline")

- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token for synchronization. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id165)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id165 "Permalink to this headline")

The `store` operation performs a scatter by storing a tile of data from a tile
into global memory.

The `destination` operand is a tile of pointers indicating the global memory
locations where data from the `value` tile will be stored. When storing i1 values,
each value occupies a full byte in memory. Any nonzero byte is canonicalized to 0x01,
and zero bytes become 0x00.

Additionally, the operation supports an optional `mask` operand, which allows
selective scattering of elements. If provided, only the elements specified by
the `mask` are stored. The shape of the `mask` must align with the shape of
the `value` tile.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.
For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location.
For more information, refer to the [memory model section](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model) of the specification.

- `weak` - No concurrent accesses to the source/destination location.
- `relaxed` - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- `release` - There may be concurrent access to the location. If this release is observed with an acquire operation, then _happens before_ is established.

Note: The following variants are not supported by this operation: `acquire`, `acq_rel`.

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id166)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id166 "Permalink to this headline")

- Operation must encode variadic operand segment sizes in attributes.
- *destination* type is expected a pointer type of *value* type
- shape of ‘destination’ must match the shape of ‘mask’
- Operation must infer result types from operands and attributes.
