---
title: "8.6.2. cuda_tile.load_ptr_tko"
section: "8.6.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-load-ptr-tko"
---

### [8.6.2. cuda_tile.load_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-load-ptr-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-load-ptr-tko "Permalink to this headline")

_Load and gather data from global memory using a pointer tile without ordering guarantees_

```default
cuda_tile.load_ptr_tko %memory_ordering_semantics %memory_scope %source %mask %paddingValue %token %optimization_hints
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id154)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id154 "Permalink to this headline")

- **memory_ordering_semantics** ([MemoryOrderingSemantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-ptr-tko-memoryorderingsemantics-attr)) - The memory ordering semantics for the load operation. 13.1
- **memory_scope** ([MemoryScope](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-ptr-tko-memoryscope-attr)) - The memory scope for the atomic operation. 13.1
- **source** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The source tile of pointers. 13.1
- **mask** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The mask for the load operation. 13.1
- **paddingValue** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The padding value for the load operation. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The token for the load operation. 13.1
- **optimization_hints** ([OptimizationHints](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-load-ptr-tko-optimizationhints-attr)) - Optimization hints for operation 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id155)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id155 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The result of the load operation. 13.1
- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token of the load operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id156)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id156 "Permalink to this headline")

This `load` OP performs a gather operation by loading
a tile of data from global memory into a result tile based on a
tile of pointers provided by the `source` operand.

The `source` operand is a tile of pointers, which specifies the memory
locations from which the data is gathered. The operation loads this data
and returns it as the `result` tile. When loading i1 values, each value
is loaded from a full byte in memory. Any nonzero byte is canonicalized to 0x01,
and zero bytes become 0x00.

Optionally, a `mask` operand can be provided to control the gathering of
elements. If present, only the elements specified by the `mask` are loaded.
The shape of the `mask` must match the shape of the `result`.

When `mask` is present one `paddingValue` can be optionally present as well.
The `paddingValue` must have the same shape of the `source` tile. If
it is not present, the value of masked elements are undefined.

Token-ordered operations are not constrained by program order.
The compiler may reorder them (i.e. place them earlier or
later in program order) unless further constrained by tokens.

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id157)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id157 "Permalink to this headline")

- Operation must encode variadic operand segment sizes in attributes.
- *source* type is expected a pointer type of *result* type
- shape of ‘mask’ must match the shape of ‘source’
- type of ‘paddingValue’ must match the type of ‘result’

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id158)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id158 "Permalink to this headline")

```mlir
%mask = constant <i1: 1> : tile<i1>
%padding = constant <f32: 0.0> : tile<f32>

  // Load without token.
  %result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
      : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

  // Load with token.
  %token0 = make_token : token
  %result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
      : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

  return
```

See [cuda_tile.load_ptr_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-load-ptr-tko-0) for the full example listing.
