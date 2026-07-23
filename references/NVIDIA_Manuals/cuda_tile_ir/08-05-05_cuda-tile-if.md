---
title: "8.5.5. cuda_tile.if"
section: "8.5.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-if"
---

### [8.5.5. cuda_tile.if](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-if)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-if "Permalink to this headline")

_Conditional execution_

```default
cuda_tile.if %condition
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id130)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id130 "Permalink to this headline")

- **condition** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The condition of the if operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id131)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id131 "Permalink to this headline")

- **results** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The results of the if operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id132)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id132 "Permalink to this headline")

The `if` operation represents an if-then-else construct.

The *if* operation consists of (1) a control operand which is a `tile<i1>` value, (2) a true branch `thenRegion`
and (3) an optional false branch `elseRegion`.

The `if` operation may produce results by yielding values in each branch using [cuda_tile.yield](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-yield).

If yielding value(s) the types of yielded values must match and the result
result type of the `if` operation will be the same as the yielded values.

If yielding values the else branch is required and must also yield a value.

The values returned will be dependent on which branch is taken.

> **Warning**
>
> The `if` operation has a set of additional restrictions today:
>
> - Results of `if` must not be a [tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) or view type.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id133)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id133 "Permalink to this headline")

- All regions must have zero arguments.
- Operation must provide custom parsing and printing methods.
- Operation effects must include all nested operations’ effects.
- Each region must contain exactly one block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id134)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id134 "Permalink to this headline")

```mlir
%condition = constant <i1: 1> : tile<i1>

// A simple if operation that conditionally executes a region.
if %condition  {
  // ...
}

// An if operation with an "else" branch.
if %condition  {
  // ...
} else {
  // ...
}

// An if operation that returns mixed types (f32,i32)
%x, %y = if %condition -> (tile<f32>, tile<i32>) {
  %x_then = constant <f32: 1.0> : tile<f32>
  %y_then = constant <i32: 2> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
} else {
  %x_then = constant <f32: 1.0> : tile<f32>
  %y_then = constant <i32: 42> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
}
```

See [cuda_tile.if_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-if-0) for the full example listing.
