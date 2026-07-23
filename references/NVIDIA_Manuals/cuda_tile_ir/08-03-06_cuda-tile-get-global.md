---
title: "8.3.6. cuda_tile.get_global"
section: "8.3.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-get-global"
---

### [8.3.6. cuda_tile.get_global](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-global)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-global "Permalink to this headline")

_Get a pointer to a global variable_

```default
cuda_tile.get_global %name
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id19)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id19 "Permalink to this headline")

- **name** ([Symbol](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-symbol)) - The name of the global variable. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id20)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id20 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)>) - The result of the get_global operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id21)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id21 "Permalink to this headline")

The `get_global` operation returns a pointer to the specified `global`
variable. A global variable is a form of static global memory allocation that can
be declared using the [cuda_tile.global](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-global) operation.

The element type of the returned pointer will be of the same type as the
element type of the declared global variable.

For detailed semantics of global variables see [Global Variable](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#sub-sec-tile-global).

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id22)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id22 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id23)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id23 "Permalink to this headline")

```mlir
global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

entry @example() {
  %ptr = get_global @val : tile<ptr<f32>>
  return
}
```

See [cuda_tile.get_global_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-get-global-0) for the full example listing.
