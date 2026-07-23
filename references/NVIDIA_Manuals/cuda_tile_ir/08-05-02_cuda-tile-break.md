---
title: "8.5.2. cuda_tile.break"
section: "8.5.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-break"
---

### [8.5.2. cuda_tile.break](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-break)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-break "Permalink to this headline")

_Break from loop_

```default
cuda_tile.break %operands
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id115)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id115 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The operands to yield to the parent loop upon termination. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id116)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id116 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id117)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id117 "Permalink to this headline")

The `break` operation is a terminator operation of a [cuda_tile.loop](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-loop).

It may yield any number of `$operands` to the parent loop upon termination. The number of values yielded
and the execution semantics of how they are yielded are determined by the parent loop.

The `break` operation always returns control to the innermost enclosing loop operation,
even when it is nested within other control constructs such as `if` or additional loops.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id118)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id118 "Permalink to this headline")

- Operation must terminate its parent basic block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id119)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id119 "Permalink to this headline")

```mlir
// Break from the body of a loop.
loop {
    break
}

// Break from an if nested within the loop.
loop  {
    %condition = constant <i1: 1> : tile<i1>
    if %condition  {
        break
    }
    // ...
}

%initValue0 = constant <f32: 0.0> : tile<f32>
// Break from an if nested within the loop, while yielding values.
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant <i1: 1> : tile<i1>
    if %condition  {
        // ...
        yield
    } else {
        // %if.loopValue0 = ...
        %loopValue0 = constant <f32: 1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant <f32: 1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

See [cuda_tile.break_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-break-0) for the full example listing.
