---
title: "8.5.4. cuda_tile.for"
section: "8.5.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-for"
---

### [8.5.4. cuda_tile.for](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-for)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-for "Permalink to this headline")

_For loop over integer range_

```default
cuda_tile.for %lowerBound %upperBound %step %initValues %unsignedCmp
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id125)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id125 "Permalink to this headline")

- **lowerBound** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The lower bound of the loop. 13.1
- **upperBound** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The upper bound of the loop. 13.1
- **step** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The step of the loop. 13.1
- **initValues** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The initial values of the loop-carried values. 13.1
- **unsignedCmp** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If present, use unsigned integer comparison for loop termination. 13.2

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id126)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id126 "Permalink to this headline")

- **resultValues** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The values of the loop-carried variables after loop termination. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id127)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id127 "Permalink to this headline")

The `for` operation is a structured range-based sequential loop.

The loop operation consists of (1) a range formed by `lowerBound`, `upperBound`, and `step`,
(2) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and
(3) a region which represents the loop body.

The iteration space is defined by the interval \(\([lowerBound, upperBound)\)\) with each value
separated by `step`.

$$
\[range(L_b, U_b, S) = \{ L_b + i \cdot S \mid i \in \mathbb{Z}, L_b + i \cdot S < U_b \}\]
$$

`lowerBound`, `upperBound`, and `step` must be of the same type.
`lowerBound` and `upperBound` specify a half-open (or exclusive) range: the range
includes the `lowerBound` but does not include the `upperBound`.
`step` must be positive but the bounds may be negative or zero.

The `lowerBound`, `upperBound`, and `step` operands are interpreted as signed integers.

The first iteration of the loop receives the induction variable initialized to the value of `lowerBound`
and the loop-carried values initialized to the values of `initValues`.

The loop body is executed for each value in the range, receiving an integer induction variable
incremented by `step` on each iteration and the loop-carried values which correspond to the
loop-carried values yielded by the previous loop iteration.

The loop terminates when the induction variable is greater than or equal to
`upperBound`. By default, signed comparison is used between the
upperBound and the induction variable. To use unsigned comparison instead,
specify the optional `unsigned` unit attribute.

The body of the loop must be terminated by a [cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-continue) that yields
the next iteration’s value for each loop carried variable.

The for operation produces one return value for each loop carried variable. The type of the \(\(i\)\)-th return
value is that of the \(\(i\)\)-th loop carried variable and its value is the final value of the
\(\(i\)\)-th loop carried variable.

> **Warning**
>
> - Loop carried variables can not be a [tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) or view type.
> - `for` operations cannot terminate early and must end in a [cuda_tile.continue](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-continue).

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id128)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id128 "Permalink to this headline")

- Operation must define scope where stack allocations are automatically freed.
- `lowerBound`, `upperBound` and `step` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>).
- `initValues` and `resultValues` must have the same shape and element type ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>).
- Operation must provide custom parsing and printing methods.
- Operation effects must include all nested operations’ effects.
- Each region must contain exactly one block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id129)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id129 "Permalink to this headline")

```mlir
%lowerBound = constant <i32: 0> : tile<i32>
%upperBound = constant <i32: 10> : tile<i32>
%step = constant <i32: 1> : tile<i32>

// A simple loop iterating over an i32 range.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    continue
}

%initVal0 = constant <f32: 0.0> : tile<f32>
// A similar loop to the above, but with a loop carried value, val0.
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                    iter_values(%val00 = %initVal0) -> (tile<f32>) {
  %loopVal0 = constant <f32: 1.0> : tile<f32>
  continue %loopVal0 : tile<f32>
}
```

See [cuda_tile.for_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-for-0) for the full example listing.
