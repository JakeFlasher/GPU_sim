---
title: "8.7.19. cuda_tile.mmaf"
section: "8.7.19"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-mmaf"
---

### [8.7.19. cuda_tile.mmaf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mmaf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mmaf "Permalink to this headline")

_Floating-point matrix-multiply-accumulate_

```default
cuda_tile.mmaf %lhs %rhs %acc
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id241)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id241 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>>) - The left hand side matrix operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>>) - The right hand side matrix operand. 13.1
- **acc** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The accumulator matrix operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id242)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id242 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result matrix after multiplication and accumulation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id243)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id243 "Permalink to this headline")

The `mmaf` operation implements an MMA (matrix-multiply-accumulate) operation for floating-point tiles.
It performs matrix multiplication on the floating-point tiles `lhs` and `rhs`, then adds the tile `acc` to the result.
`lhs`, `rhs`, and `acc` must be 2D tiles or 3D tiles. The latter case
indicates a batched matrix multiplication.

$$
\[\text{mmaf}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}\]
$$

The types of all operands must be a supported combination (see [mmaf Supported Data Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#table-cuda-tile-mmaf-0)).

Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands `lhs`, `rhs`, and `acc` to have shapes `M x K`,
`K x N`, and `M x N` (respectively). Batched (3D) MMA expects the operands
to have shapes `B x M x K`, `B x K x N`, and `B x M x N` (respectively).

The table below shows the supported output types for each possible `mmaf` input type. Input operands must be of the same element type.

| Input Type | Supported Output Types |
| --- | --- |
| `f8E4M3FN` | `f16` or `f32` |
| `f8E5M2` | `f16` or `f32` |
| `f16` | `f16` or `f32` |
| `bf16` | `f32` |
| `tf32` | `f32` |
| `f32` | `f32` |
| `f64` | `f64` |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id244)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id244 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `acc` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- `lhs` and `rhs` must have the same element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>>).
- `lhs`, `rhs` and `acc` must have the same rank.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id245)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id245 "Permalink to this headline")

```mlir
%lhs0 = constant <f16: 0.0> : tile<4x8xf16>
%rhs0 = constant <f16: 0.0> : tile<8x2xf16>
%acc0 = constant <f32: 0.0> : tile<4x2xf32>

%0 = mmaf %lhs0, %rhs0, %acc0
    : tile<4x8xf16>, tile<8x2xf16>,
      tile<4x2xf32>

%lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
%rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
%acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

%1 = mmaf %lhs1, %rhs1, %acc1
    : tile<2x4x8xf16>, tile<2x8x2xf16>,
      tile<2x4x2xf32>
```

See [cuda_tile.mmaf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-mmaf-0) for the full example listing.
