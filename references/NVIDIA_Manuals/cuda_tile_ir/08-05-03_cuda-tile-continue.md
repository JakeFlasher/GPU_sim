---
title: "8.5.3. cuda_tile.continue"
section: "8.5.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-continue"
---

### [8.5.3. cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-continue)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-continue "Permalink to this headline")

_Continue to next loop iteration_

```default
cuda_tile.continue %operands
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id120)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id120 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The values to yield to the parent loop. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id121)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id121 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id122)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id122 "Permalink to this headline")

The `continue` operation represents a block terminator that returns control to
a loop operation, such as [cuda_tile.for](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-for) and [cuda_tile.loop](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-loop). The operation
may yield any number of `$operands` to the parent loop upon termination.

The requirements and semantics of the `continue` operation are defined by the parent loop
operation, see the loop operation’s description for particular semantics.

The `continue` operation always returns control to the innermost enclosing loop operation,
even when it is nested within other control constructs such as `if` or additional loops.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id123)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id123 "Permalink to this headline")

- Operation must terminate its parent basic block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id124)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id124 "Permalink to this headline")

```mlir
  %lowerBound = constant <i32: 0> : tile<i32>
  %upperBound = constant <i32: 10> : tile<i32>
  %step = constant <i32: 1> : tile<i32>
  %condition = constant <i1: 1> : tile<i1>
  // Continue from the body of a loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      continue
  }

  // Continue from an if nested within the loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      if %condition  {
          continue
      }
      // ...
  }

// Continue from an if nested within the loop, while yielding values.
%initVar0 = constant <f32: 0.0> : tile<f32>
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
          iter_values(%var0 = %initVar0) -> (tile<f32>)
  {
      if %condition {
          // ...
          yield
      } else {
          %loopValue0 = constant <f32: 1.0> : tile<f32>
          continue %loopValue0 : tile<f32>
      }
      %loopValue1 = constant <f32: 1.0> : tile<f32>
      continue %loopValue1 : tile<f32>
  }
```

See [cuda_tile.continue_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-continue-0) for the full example listing.
