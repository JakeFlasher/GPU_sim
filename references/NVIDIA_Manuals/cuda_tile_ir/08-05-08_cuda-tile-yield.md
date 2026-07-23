---
title: "8.5.8. cuda_tile.yield"
section: "8.5.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-yield"
---

### [8.5.8. cuda_tile.yield](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-yield)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-yield "Permalink to this headline")

_Yield a value from the block_

```default
cuda_tile.yield %operands
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id145)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id145 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The operands to yield to the parent operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id146)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id146 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id147)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id147 "Permalink to this headline")

The `yield` operation terminates a block that must yield control back to the parent operation
such as `if`, `scan`, `reduce`.

The operation may yield any number of `$operands` to the parent upon termination. The number of values yielded
and the execution semantics of how they are yielded are determined by the parent operation.

> **Note**
>
> Unlike standard MLIR control flow dialects `yield` is not used for loop control flow, see
> [cuda_tile.break](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-break) and [cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-continue) for loop control flow.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id148)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id148 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must terminate its parent basic block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id149)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id149 "Permalink to this headline")

```mlir
%condition = constant <i1: true> : tile<i1>
// Yield from the body of an if conditional.
if %condition  {
    yield
}

// Yield values from within an if conditional.
%x, %y = if %condition -> (tile<f32>, tile<f32>) {
    %x_then = constant <f32: 0.0> : tile<f32>
    %y_then = constant <f32: 1.0> : tile<f32>
    yield %x_then, %y_then : tile<f32>, tile<f32>
} else {
    %x_else = constant <f32: 2.0> : tile<f32>
    %y_else = constant <f32: 3.0> : tile<f32>
    yield %x_else, %y_else : tile<f32>, tile<f32>
}
```

See [cuda_tile.yield_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-yield-0) for the full example listing.
