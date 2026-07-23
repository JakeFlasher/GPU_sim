---
title: "8.5.7. cuda_tile.return"
section: "8.5.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-return"
---

### [8.5.7. cuda_tile.return](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-return)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-return "Permalink to this headline")

_Return value(s) from a function_

```default
cuda_tile.return %operands
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id140)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id140 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>) - The values to return. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id141)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id141 "Permalink to this headline")

No results.

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id142)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id142 "Permalink to this headline")

The `return` operation returns control to the caller of a function.

> **Warning**
>
> Currently `return` implements restricted return semantics, notably:
>
> - [cuda_tile.entry](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-entry) operations do not produce return value(s) and thus
> `return` may be used to terminate the execution of the kernel by invoking
> the operation with no operands
> - `return` can not be directly used inside of loop bodies to terminate the
> the execution of the kernel

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id143)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id143 "Permalink to this headline")

- Operation must terminate its parent basic block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id144)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id144 "Permalink to this headline")

```mlir
entry @foo() {
  %0 = constant <i32: 0> : tile<i32>
  %1 = constant <f16: 0.0> : tile<f16>
  // ...
  return
}
```

See [cuda_tile.return_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-return-0) for the full example listing.
