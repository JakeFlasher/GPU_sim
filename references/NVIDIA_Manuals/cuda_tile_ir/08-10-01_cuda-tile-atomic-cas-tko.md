---
title: "8.10.1. cuda_tile.atomic_cas_tko"
section: "8.10.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-atomic-cas-tko"
---

### [8.10.1. cuda_tile.atomic_cas_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-atomic-cas-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-atomic-cas-tko "Permalink to this headline")

_Atomic compare-and-swap on global memory_

```default
cuda_tile.atomic_cas_tko %memory_ordering_semantics %memory_scope %pointers %cmp %val %mask %token
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id370)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id370 "Permalink to this headline")

- **memory_ordering_semantics** ([MemoryOrderingSemantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-atomic-cas-tko-memoryorderingsemantics-attr)) - The memory ordering semantics for the atomic operation. 13.1
- **memory_scope** ([MemoryScope](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-atomic-cas-tko-memoryscope-attr)) - The memory scope for the atomic operation. 13.1
- **pointers** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The pointers to the memory locations to perform the atomic compare-and-swap operation on. 13.1
- **cmp** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The values to compare against. 13.1
- **val** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The values to swap in. 13.1
- **mask** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The mask for the atomic operation. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The token for the atomic operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id371)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id371 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)) - The result of the atomic operation. 13.1
- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token of the atomic operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id372)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id372 "Permalink to this headline")

The `atomic_cas` operation performs element-wise, atomic
compare-and-swaps at the specified global memory `pointers`. The data in
memory is compared to `cmp` and the data written to memory is specified
by `val`. The operation returns the original value that was stored in memory
before the atomic operation was performed.

The shape (and the element type) of `pointers`, `cmp`,
`val` and `result` must match. The `atomic_cas` operation
performs the following steps for every `(pointer, cmp, val)` tuple in one atomic
transaction. (One atomic transaction per tuple.)

```mlir
  atomic() {
    x = *pointer
    if x == cmp {
    *pointer = val
  }
  return x
}
```

An optional parameter, `mask`, allows specifying which elements participate
in the atomic operation. A false value at position i masks out the
corresponding element in `pointers`, excluding it from the operation. The
returned value for a masked element at position i is `cmp[i]`. If no mask is
provided, all elements are included in the computation by default. The shape of
mask must match that of `pointers`, `cmp`, and `val`.

A token-ordered atomic compare-and-swap is not constrained by program order. The compiler
may reorder it (i.e. place them earlier or later in program order) unless
constrained by tokens.

**Supported data types:**

  - i32, i64: integer values
  - f32, f64: floating-point values

For floating-point types, the comparison uses bitwise equality rather than
IEEE-754 semantics. This means different `NaN` bit patterns are treated as
distinct values, and `+0.0` and `-0.0` are considered different if their bit
representations differ.

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id373)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id373 "Permalink to this headline")

- `cmp`, `val` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)).
- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id374)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id374 "Permalink to this headline")

```mlir
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
%cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
%val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>
%mask = constant <i1: [0, 1, 0, 1, 0, 1, 0, 1]> : tile<8xi1>

// Atomic CAS without input token.
%0, %token = atomic_cas_tko relaxed device %ptrs, %cmp, %val :
  tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

// Atomic CAS without input token.
%1, %token1 = atomic_cas_tko relaxed device %ptrs, %cmp, %val, %mask :
  tile<8xptr<i32>>, tile<8xi32>, tile<8xi1> -> tile<8xi32>, token

// Atomic CAS with input token.
%token2 = make_token : token
%2, %token3 = atomic_cas_tko relaxed device %ptrs, %cmp, %val token=%token2 :
  tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

return
```

See [cuda_tile.atomic_cas_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-atomic-cas-tko-0) for the full example listing.
