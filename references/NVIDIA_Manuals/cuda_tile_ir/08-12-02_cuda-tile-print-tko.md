---
title: "8.12.2. cuda_tile.print_tko"
section: "8.12.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-print-tko"
---

### [8.12.2. cuda_tile.print_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-print-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-print-tko "Permalink to this headline")

_Print a formatted string (token-ordered)_

```default
cuda_tile.print_tko %str %args %token
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id415)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id415 "Permalink to this headline")

- **str** ([String](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-string)) - The format string. 13.1
- **args** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)>) - The arguments to format and print. 13.1
- **token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The optional token for operation ordering. 13.2

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id416)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id416 "Permalink to this headline")

- **result_token** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The result token for synchronization. 13.2

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id417)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id417 "Permalink to this headline")

The `print_tko` operation prints a C-printf-style format string,
interleaved with the given operands. The number of format expressions
(starting with the `%` character) must match the number of operands.
If a format expression is not applicable to its respective operand, then
the output is undefined.

Token-ordered print operations are not constrained by program order. The
compiler may reorder them (i.e., move them earlier or later in the program)
unless further constrained by tokens.

This operation is meant for debugging. Its implementation is not optimized
for performance, so it should not be used in production mode. Prints are
not guaranteed to be atomic. I.e., the output of prints that execute
simultaneously may be interleaved.

> **Note**
>
> This op was renamed from `print` to `print_tko` in 13.2. The
> op code did not change.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id418)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id418 "Permalink to this headline")

- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id419)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id419 "Permalink to this headline")

```mlir
print_tko "Hello world: %f\n", %arg : tile<4xf32> -> token
print_tko "%+08.3f", %arg : tile<4xf32> -> token
```

See [cuda_tile.print_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-print-tko-0) for the full example listing.
