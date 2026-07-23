---
title: "8.8.6. cuda_tile.maxi"
section: "8.8.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-maxi"
---

### [8.8.6. cuda_tile.maxi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-maxi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-maxi "Permalink to this headline")

_Element-wise integer maximum_

```default
cuda_tile.maxi %lhs %rhs %signedness
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id312)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id312 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-maxi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id313)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id313 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the maxi operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id314)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id314 "Permalink to this headline")

The `maxi` operation computes the element-wise maximum between two input integer tiles.

$$
\[\begin{split}\text{maxi}(x, y)_i = \begin{cases}
  x_i & \text{if } x_i \geq y_i \\
  y_i & \text{if } x_i < y_i
\end{cases}\end{split}\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id315)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id315 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id316)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id316 "Permalink to this headline")

```mlir
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from them.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
// Signless i32 treated as unsigned
%4 = maxi %2, %3 unsigned : tile<2x4xi32>
// Signless i32 treated as signed
%5 = maxi %2, %3 signed : tile<2x4xi32>
```

See [cuda_tile.maxi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-maxi-0) for the full example listing.
