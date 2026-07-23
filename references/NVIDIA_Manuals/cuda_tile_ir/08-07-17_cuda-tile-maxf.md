---
title: "8.7.17. cuda_tile.maxf"
section: "8.7.17"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-maxf"
---

### [8.7.17. cuda_tile.maxf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-maxf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-maxf "Permalink to this headline")

_Element-wise floating-point maximum_

```default
cuda_tile.maxf %lhs %rhs %propagate_nan %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id231)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id231 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **propagate_nan** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - When set, `maxf` (or `minf`) returns a `NaN` if either of the two compared elements is `NaN`. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id232)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id232 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the `maxf` operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id233)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id233 "Permalink to this headline")

The `maxf` operation computes the element-wise maximum of two input
tiles with floating-point element types.

The `propagate_nan` controls how `maxf` will interpret `NaN`. If
the `propagate_nan` modifier is set, `maxf` returns a canonical `NaN`
if either of the compared elements is `NaN` (IEEE 754-2019’s maximum). While if
the `propagate_nan` modifier is not set, `maxf` returns a canonical `NaN`
only if both elements are `NaN`; otherwise, it returns the non-`NaN` element (IEEE
754-2019’s maximumNumber).

If neither element is `NaN`, `maxf` will return the greater of the
inputs. `+0.0` is considered greater than `-0.0`.

If the `flush_to_zero` modifier is specified, denormal numbers are
flushed to sign-preserving zero. The `flush_to_zero` modifier applies
only to the f32 data type.

$$
\[\begin{split}\text{maxi}(x, y)_i = \begin{cases}
  x_i & \text{if } x_i \geq y_i \\
  y_i & \text{if } x_i < y_i
\end{cases}\end{split}\]
$$

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id234)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id234 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id235)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id235 "Permalink to this headline")

```mlir
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from partition views.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
// IEEE 754-2019's maximum
%4 = maxf %2, %3 propagate_nan : tile<2x4xf32>
// IEEE 754-2019's maximumNumber
%5 = maxf %2, %3 : tile<2x4xf32>
// flush denormal to positive zero
%6 = maxf %2, %3 flush_to_zero : tile<2x4xf32>
```

See [cuda_tile.maxf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-maxf-0) for the full example listing.
