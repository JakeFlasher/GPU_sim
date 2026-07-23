---
title: "8.3.4. cuda_tile.entry"
section: "8.3.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-entry"
---

### [8.3.4. cuda_tile.entry](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-entry)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-entry "Permalink to this headline")

_Define a tile kernel_

```default
cuda_tile.entry %sym_name %function_type %arg_attrs %res_attrs %optimization_hints
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id10)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id10 "Permalink to this headline")

- **sym_name** ([Symbol](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-symbol)) - The name of the function. 13.1
- **function_type** ([Type](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-type)) - The type of the function. 13.1
- **arg_attrs** ([Attributes](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-attributes)) - The argument attributes of the function: none of these are supported by CUDA Tile IR at the moment. 13.1
- **res_attrs** ([Attributes](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-attributes)) - The result attributes of the function: none of these are supported by CUDA Tile IR at the moment. 13.1
- **optimization_hints** ([OptimizationHints](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-entry-optimizationhints-attr)) - Compiler architecture-specific optimization hints 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id11)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id11 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id12)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id12 "Permalink to this headline")

The `entry` operation defines a tile kernel; a kernel is a function that can
serve as the program entry point. It has a unique name per-module. A kernel can
not return any value. It must be launched from the host side using `cuLaunchKernel`
or similar CUDA runtime API functions.

Tile kernels require that the user specifies the 3-d grid dimensions at launch which
defines the number of tile blocks (or kernel instances) that will execute the kernel
in parallel.

For detailed semantics of tile kernels see [Tile Kernel](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#sub-sec-tile-kernel).

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id13)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id13 "Permalink to this headline")

- Operation must be a symbol in the global symbol table.
- Operation must implement callable target interface.
- Operation must implement function-like behavior interface.
- Region must not capture SSA values defined above the operation.
- Operation must provide custom parsing and printing methods.
- Each region must contain exactly one block.
