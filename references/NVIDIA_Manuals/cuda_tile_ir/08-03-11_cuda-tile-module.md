---
title: "8.3.11. cuda_tile.module"
section: "8.3.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-module"
---

### [8.3.11. cuda_tile.module](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-module)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-module "Permalink to this headline")

_Top-level module containing a series of defined items._

```default
cuda_tile.module %sym_name
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id42)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id42 "Permalink to this headline")

- **sym_name** ([Symbol](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-symbol)) - The name of the module. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id43)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id43 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id44)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id44 "Permalink to this headline")

A `module` operation represents a single compilation unit and contains
zero or more items (global variables, functions, or kernels).

For detailed description of the semantics of modules, and the full definition of each item type see
[Modules](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#sub-sec-modules).

The `module` operation is the top-level operation in a **Tile IR** module and must
contain only **Tile IR** operations and no other dialects.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id45)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id45 "Permalink to this headline")

- Region must not capture SSA values defined above the operation.
- Operation must provide custom parsing and printing methods.
- All regions must have zero arguments.
- Each region must contain exactly one block.
- Operation must define a symbol scope.
- Regions must not require explicit terminator operations.
- Operation must specify whether regions are SSACFG or Graph kind.
- Operation must contain only dataflow Graph regions.
