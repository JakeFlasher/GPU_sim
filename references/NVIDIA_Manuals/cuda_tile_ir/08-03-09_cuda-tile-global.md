---
title: "8.3.9. cuda_tile.global"
section: "8.3.9"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-global"
---

### [8.3.9. cuda_tile.global](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-global)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-global "Permalink to this headline")

_Allocate static global memory_

```default
cuda_tile.global %sym_name %value %alignment
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id33)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id33 "Permalink to this headline")

- **sym_name** ([Symbol](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-symbol)) - The name of the global variable. 13.1
- **value** ([DenseConstant](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-denseconstant)) - The value to initialize the allocation with. 13.1
- **alignment** ([i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)) - The alignment of the buffer. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id34)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id34 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id35)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id35 "Permalink to this headline")

The `global` operation statically allocates a mutable 1-dimensional location in global
memory and initializes it using `value`. The initialization of the allocation is performed
at [CUDA module](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef)
load time. The lifetime of the allocation is the same as the lifetime of the module.

The allocation may be read or written to by first using [cuda_tile.get_global](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-get-global) to obtain a pointer to the
the memory and then read using [cuda_tile.load_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-ptr-tko) or written to using [cuda_tile.store_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-ptr-tko).

The initial values are stored in memory in linear order, so the pointer returned by [cuda_tile.get_global](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-get-global)
points to the first element, and offsetting the pointer by *x* would allow to load element at position *x*.

`global` operations must be directly nested within the **Tile IR** module. They cannot be defined inside functions.
As globals are defined at the module scope their names are globally unique symbols and must not collide with any other
symbol in the module.

For more detailed semantics of global variables see [Global Variable](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#sub-sec-tile-global).

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id36)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id36 "Permalink to this headline")

- Operation must be a symbol in the global symbol table.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id37)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id37 "Permalink to this headline")

```mlir
global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
entry @example() {}
```

See [cuda_tile.global_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-global-0) for the full example listing.
