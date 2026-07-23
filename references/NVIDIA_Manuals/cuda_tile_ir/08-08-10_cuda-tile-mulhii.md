---
title: "8.8.10. cuda_tile.mulhii"
section: "8.8.10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-mulhii"
---

### [8.8.10. cuda_tile.mulhii](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mulhii)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mulhii "Permalink to this headline")

_Element-wise high bits of integer multiplication_

```default
cuda_tile.mulhii %x %y
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id331)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id331 "Permalink to this headline")

- **x** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side input integer tile. 13.1
- **y** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side input integer tile. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id332)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id332 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The most significant bits of the product of the input tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id333)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id333 "Permalink to this headline")

The `mulhii` operation produces the most significant N bits of the 2N-bit
product of two N-bit integer tiles. For `i64`, this is the most significant 64
bits of the full 128-bit product; for `i8`, it is the most significant 8
bits of the full 16-bit product; etc.

This is in contrast to `muli`, which produces the lower N bits of the 2N-bit
product.

The `mulhii` operation is only defined for unsigned integers.

$$
\[\text{mulhii}(x_i, y_i) = x_i \times y_i >> \text{bitwidth}(\text{type}(x_i))\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id334)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id334 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `x`, `y` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id335)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id335 "Permalink to this headline")

```mlir
// 2^31 * 2 = 2^32, or 0x100000000.
// The most significant 32 bits of the product are 0x00000001.
// The lower 32 bits of the product are 0x00000000.
%a = constant <i32: 2147483648> : tile<i32>  // %a = 2^31
%b = constant <i32: 2> : tile<i32>           // %b = 2
%res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
%res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
```

See [cuda_tile.mulhii_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-mulhii-0) for the full example listing.
