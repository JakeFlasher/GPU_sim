---
title: "8.5.6. cuda_tile.loop"
section: "8.5.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-loop"
---

### [8.5.6. cuda_tile.loop](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-loop)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-loop "Permalink to this headline")

_Loop until a break operation_

```default
cuda_tile.loop %initValues
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id135)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id135 "Permalink to this headline")

- **initValues** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The initial values of the loop. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id136)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id136 "Permalink to this headline")

- **resultValues** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The result values of the loop. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id137)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id137 "Permalink to this headline")

The `loop` operation represents an, unstructured, infinite loop that executes
until a [cuda_tile.break](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-break) is reached.

The loop consists of a (1) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and
(2) a region which represents the loop body.

The loop will execute the body of the loop until a [cuda_tile.break](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-break) is dynamically executed.

Each control path of the loop must be terminated by:

- a [cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-continue) that yields the next iteration’s value for each loop carried variable.
- a [cuda_tile.break](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-break) that terminates the loop and yields the final loop carried values.

As long as each loop iteration is terminated by one of these operations they may be combined with other control
flow operations to express different control flow patterns.

The loop operation produces one return value for each loop carried variable. The type of the \(\(i\)\)th return
value is that of the \(\(i\)\)th loop carried variable and its value is the final value of the
\(\(i\)\)th loop carried variable.

> **Warning**
>
> Loop operations have a set of additional restrictions today:
>
> - Early returns from inside loops are not supported, a code generator must first terminate the loop and then return if they wish to end the
> function execution entirely.
> - Loop carried variables can not be a [tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) or view type.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id138)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id138 "Permalink to this headline")

- Operation must define scope where stack allocations are automatically freed.
- Operation must provide custom parsing and printing methods.
- Operation effects must include all nested operations’ effects.
- Each region must contain exactly one block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id139)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id139 "Permalink to this headline")

```mlir
// A simple "while-do" loop.
loop {
    %cond = constant <i1: 1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See [cuda_tile.loop_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-loop-0) for the full example listing.

```mlir
// A simple "do-while" loop.
loop {
    //... body of the loop.

    %cond = constant <i1: 1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See [cuda_tile.loop_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-loop-1) for the full example listing.

```mlir
%initValue0 = constant <f32: 0.0> : tile<f32>
// A loop that yields carried-iteration values, returning the final values.
%results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
    %cond = constant <i1: 1> : tile<i1>
    if %cond {
        %loopValue0 = constant <f32: 0.0> : tile<f32>
        continue %loopValue0 : tile<f32>
    }
    break %value0 : tile<f32>
}
```

See [cuda_tile.loop_2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-loop-2) for the full example listing.

```mlir
%initValue0 = constant <i32: 0> : tile<i32>
// A loop that uses loop-carried values and returns a different type.
%results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
    %cond = constant <i1: 1> : tile<i1>

    if %cond {
        %newLoopValue = constant <i32: 0> : tile<i32>
        continue %newLoopValue : tile<i32>
    }

    %finalReturnValue = constant <f32: 0.0> : tile<f32>
    break %finalReturnValue : tile<f32>
}
```

See [cuda_tile.loop_3](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-loop-3) for the full example listing.
