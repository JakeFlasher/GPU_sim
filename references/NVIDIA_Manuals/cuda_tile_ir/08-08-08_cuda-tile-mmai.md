---
title: "8.8.8. cuda_tile.mmai"
section: "8.8.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-mmai"
---

### [8.8.8. cuda_tile.mmai](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mmai)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mmai "Permalink to this headline")

_Integer matrix-multiply-accumulate_

```default
cuda_tile.mmai %lhs %rhs %acc %signedness_lhs %signedness_rhs
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id322)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id322 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side matrix operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side matrix operand. 13.1
- **acc** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The accumulator matrix operand. 13.1
- **signedness_lhs** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-mmai-signedness-attr)) - The signedness of the `lhs` operand. 13.1
- **signedness_rhs** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-mmai-signedness-attr)) - The signedness of the `rhs` operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id323)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id323 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result matrix after multiplication and accumulation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id324)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id324 "Permalink to this headline")

The `mmai` operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles.
It performs matrix multiplication on the integer tiles `lhs` and `rhs`, then adds the tile `acc` to the result.
`lhs`, `rhs`, and `acc` must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

$$
\[\text{mmai}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}\]
$$

Input tiles `lhs` and `rhs` must be of integer type `i8`. The signedness of
`lhs` and `rhs` are specified separately by the `signedness_lhs` and
`signedness_rhs` attributes, respectively. The accumulator tile `acc` must be
of type `i32` and is always interpreted as signed. The output tile `result`
is of type `i32` and is always interpreted as signed.

Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands `lhs`, `rhs`, and `acc` to have shapes `M x K`,
`K x N`, and `M x N` (respectively). Batched (3D) MMA expects the operands
to have shapes `B x M x K`, `B x K x N`, and `B x M x N` (respectively).

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id325)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id325 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `acc` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- `lhs` and `rhs` must have the same element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- `lhs`, `rhs` and `acc` must have the same rank.
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id326)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id326 "Permalink to this headline")

```mlir
%lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
%rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
%acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

%0 = mmai %lhs0, %rhs0, %acc0 signed signed
    : tile<4x8xi8>, tile<8x2xi8>,
      tile<4x2xi32>

%lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
%rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
%acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

%1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
    : tile<2x4x8xi8>, tile<2x8x2xi8>,
      tile<2x4x2xi32>
```

See [cuda_tile.mmai_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-mmai-0) for the full example listing.
