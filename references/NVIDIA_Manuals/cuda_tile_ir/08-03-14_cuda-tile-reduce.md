---
title: "8.3.14. cuda_tile.reduce"
section: "8.3.14"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-reduce"
---

### [8.3.14. cuda_tile.reduce](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reduce)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reduce "Permalink to this headline")

_Variadic tile reduction across dimensions_

```default
cuda_tile.reduce %operands %dim %identities
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id55)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id55 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)>) - The set of tiles to reduce. 13.1
- **dim** ([i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)) - The index of the dimension to perform reduction on. 13.1
- **identities** ([Array](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-array)) - The reduction identities for each operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id56)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id56 "Permalink to this headline")

- **results** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)>) - The set of reduced tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id57)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id57 "Permalink to this headline")

The `reduce` operation applies a custom reduction function along a specified dimension of
one or more input tiles, producing the same number of output tiles.

The reduction function must be an associative operation defined within the `reduce`
operation’s region. A single reduction operation can reduce over any number of input tiles in
parallel, producing a reduced output tile for each.

> **Note**
>
> Only pure operations are allowed in the body of `reduce`.

All input tiles must have the same shape. The output tiles will have a matching shape in every
dimension except the one being reduced, which is removed.

For each input tile, a constant identity value must be provided that matches the element type of
the input tile. Identity `i` of `identities` corresponds to input tile
`i` of `operands`. The correct identity value is a property of the reduction
function in the `body`. (For example, if the reduction function performs `min`,
the identity is `+inf`, while if the reduction function performs a `sum`,
the identity is `0`.)

The reduction function must expect `2N` arguments, where `N` is the number of input tiles.
Each pair of reduction arguments `2i` and `2i+1` will correspond to the `i`-th input tile.
The first argument of each pair is an element of the input tile; the second is the accumulator from all
prior reductions along the specified dimension. This second value might be input element, the identity value,
or the result of a previous reduction iteration. The reduction function should yield the new accumulator value
for each input tile.

> **Note**
>
> There are no guarantees on the order of element reduction along the specified dimension.
> However, the result is deterministic across different runs of the same kernel on the same device.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id58)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id58 "Permalink to this headline")

- Operation must provide custom parsing and printing methods.
- Operation effects must include all nested operations’ effects.
- All operands must have identical shapes.
- Each region must contain exactly one block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id59)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id59 "Permalink to this headline")

```mlir
%input = constant <f32: 0.0> : tile<8xf32>
%0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
  (%input_arg: tile<f32>, %input_accum: tile<f32>) {
    %add_result = addf %input_arg, %input_accum : tile<f32>
    yield %add_result : tile<f32>
  }
```

See [cuda_tile.reduce_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-reduce-0) for the full example listing.

```mlir
%input = constant <f32: 0.0> : tile<8x64xf32>
%0 = reduce %input dim=1 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
  (%input_arg: tile<f32>, %input_accum: tile<f32>) {
    %add_result = addf %input_arg, %input_accum : tile<f32>
    yield %add_result : tile<f32>
  }
```

See [cuda_tile.reduce_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-reduce-1) for the full example listing.
