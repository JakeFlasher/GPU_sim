---
title: "8.3.3. cuda_tile.constant"
section: "8.3.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-constant"
---

### [8.3.3. cuda_tile.constant](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-constant)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-constant "Permalink to this headline")

_Construct a constant tile_

```default
cuda_tile.constant %value
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id5)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id5 "Permalink to this headline")

- **value** ([DenseConstant](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-denseconstant)) - The constant value to create. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id6)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id6 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The constant tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id7)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id7 "Permalink to this headline")

The `constant` operation creates a tile initialized by `$value`.

There are two main forms of using the operation:

- One where the value is a single constant specified by `<D: c>`
and the tile is filled with identical values for all elements with element type `D`.
- One where the value is a list of constants specified by `dense<D: [c0, c1, c2, ...]>`
and the constant value’s shape must match the tile’s shape with the element type `D`.

The annotated type of the tile constrains its rank, shape, and element type.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id8)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id8 "Permalink to this headline")

- Operation must have zero operands and be foldable at compile time.
- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `value` and `result` must have the same shape and element type ([DenseConstant](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-denseconstant)).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id9)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id9 "Permalink to this headline")

```mlir
%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i64: 1> : tile<i64>
%c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
%c3 = constant <f32: 0.0> : tile<2x4xf32>
%c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
```

See [cuda_tile.constant_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-constant-0) for the full example listing.
