---
title: "8.10.2. cuda_tile.atomic_rmw_tko"
section: "8.10.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-atomic-rmw-tko"
---

### [8.10.2. cuda_tile.atomic_rmw_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-atomic-rmw-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-atomic-rmw-tko "Permalink to this headline")

_Atomic read-modify-write on global memory_

```default
cuda_tile.atomic_rmw_tko %memory_ordering_semantics %memory_scope %pointers %mode %arg %mask %token
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id375)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id375 "Permalink to this headline")

- **memory_ordering_semantics** ([MemoryOrderingSemantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-atomic-rmw-tko-memoryorderingsemantics-attr)) - The memory ordering semantics for the load operation. 13.1
- **memory_scope** ([MemoryScope](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-atomic-rmw-tko-memoryscope-attr)) - The memory scope for the atomic operation. 13.1
- **pointers** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The pointer tile to perform atomic operation on. 13.1
- **mode** ([AtomicRMWMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-atomic-rmw-tko-atomicrmwmode-attr)) - The atomic operation mode (e.g., add, max, min, etc.). 13.1
- **arg** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The value tile to use in the atomic operation. 13.1
- **mask** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The mask for the load operation. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The token for the atomic operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id376)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id376 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The result of the atomic operation. 13.1
- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token of the load operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id377)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id377 "Permalink to this headline")

The `atomic_rmw_tko` operation performs element-wise, atomic
read-modify-write operations at the global memory locations specified
by `pointers`. The values written to memory are determined by
`mode` and `arg`. The operation returns the original value
stored at each location before the atomic update.

The shapes of `pointers`, `arg`, and `result` must
match. The element type of the pointer type must match the element types
of both `arg` and `result`. Each (`pointer`, `arg`) pair is
processed in a single atomic transaction.

```mlir
atomic {
  x = *pointer
  y = mode(x, arg)
  *pointer = y
  return x
}
```

An optional parameter, `mask`, specifies which elements participate
in the atomic operation. A *False* value at position `i` excludes
the corresponding element in `pointers` from the operation.
The value returned for a masked-out element is implementation-defined.
The shape of `mask` must match the shape of `pointers`.

The `atomic_addf` operation is defined to round to the nearest even value.
.. note::
The current implementation of the compiler flushes denormals to zero. This behavior
will be fixed in a future version of the compiler and users should not rely on it.

Token-ordered atomic read-modify-write operations are not constrained by
program order. The compiler may reorder them (i.e., move them earlier or
later in the program) unless further constrained by tokens.

Supported data types by `mode`:

> - ADD, AND, MAX, MIN, OR, UMAX, UMIN, XOR: i32, i64
> - ADDF: f16, f32, f64
> - XCHF: i32, i64, f32, f64

The `U` prefix in UMAX and UMIN distinguishes these from their
signed counterparts (MAX and MIN) by interpreting the comparison as
unsigned.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.
For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location.
For more information, refer to the [memory model section](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model) of the specification.

- `relaxed` - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- `acquire` -  There may be concurrent accesses to the location. If this acquire observes a release operation, then _happens before_ is established.
- `release` - There may be concurrent access to the location. If this release is observed with an acquire operation, then _happens before_ is established.
- `acq_rel` - There may be concurrent accesses to the location. This has the effect of both a release and acquire operation.

Note: The following variants are not supported by this operation: `weak`.

The `memory_scope` attribute specifies a communication scope for memory operations.
When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other
threads which are participating in the communication, or data races may occur.

- `tl_blk` - There may be concurrent accesses from within the same tile block.
- `device` - There may be concurrent accesses from within the same device (i.e., GPU).
- `sys` - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `mode` attribute specifies the mode of the atomic read-modify-write operation.

- `and` - Perform bitwise AND as the modification operation.
- `or` - Perform bitwise OR as the modification operation.
- `xor` - Perform bitwise XOR as the modification operation.
- `add` - Perform integer addition as the modification operation.
- `addf` - Perform floating-point addition as the modification operation.
- `max` - Perform maximum as the modification operation.
- `min` - Perform minimum as the modification operation.
- `umax` - Perform unsigned maximum as the modification operation.
- `umin` - Perform unsigned minimum as the modification operation.
- `xchg` - Perform exchange as the modification operation.

The `mode` attribute has a default value of `add`.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id378)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id378 "Permalink to this headline")

- `arg` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)).
- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id379)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id379 "Permalink to this headline")

```mlir
// Reshape the input pointer tile to have a 1d shape
%ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
// Broadcast the reshaped tile to a tile with 8 rows, effectively replicating the pointer 8 times
%ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
// Create a tile of offsets [0, 1, 2, ..., 7] to index into memory
%offsets = iota : tile<8xi32>
// Add the offsets to each pointer in the vector to create 8 unique pointers
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
%vals = constant <f32: [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]> : tile<8xf32>

// Perform atomic addf operations on the memory locations pointed by %ptrs
// without requiring an input token. Returns the original values and a result token
%0, %res_token0 = atomic_rmw_tko relaxed device %ptrs, addf, %vals :
    tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token

// Perform atomic add operations again, this time using the explicit input token
%token = make_token : token
%1, %res_token1 = atomic_rmw_tko relaxed device %ptrs, addf, %vals, token = %token :
    tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
```

See [cuda_tile.atomic_rmw_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-atomic-rmw-tko-0) for the full example listing.
