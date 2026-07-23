---
title: "8.5.1. cuda_tile.assert"
section: "8.5.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-assert"
---

### [8.5.1. cuda_tile.assert](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-assert)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-assert "Permalink to this headline")

_Terminate kernel execution with an error message if condition is false-y_

```default
cuda_tile.assert %condition %message
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id110)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id110 "Permalink to this headline")

- **condition** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The condition tile to check. 13.1
- **message** ([String](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-string)) - The error message to display if assertion fails. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id111)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id111 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id112)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id112 "Permalink to this headline")

The `assert` operation takes as `condition` a tile of
`i1` values. For each value that is `0`, it prints the given
error message, along with the index of the value within the tile.

If at least one value is `0`, an error is signalled to the host
side. The kernel, including the tile block that failed the assertion,
may keep running.

Assertions are for debugging purposes. They can affect performance and it
is therefore recommended to remove them in production code.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id113)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id113 "Permalink to this headline")

No constraints.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id114)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id114 "Permalink to this headline")

```mlir
assert %arg0, "assertion failed" : tile<i1>
```

See [cuda_tile.assert_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-assert-0) for the full example listing.
