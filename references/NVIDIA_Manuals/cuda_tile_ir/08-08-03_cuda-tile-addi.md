---
title: "8.8.3. cuda_tile.addi"
section: "8.8.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-addi"
---

### [8.8.3. cuda_tile.addi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-addi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-addi "Permalink to this headline")

_Element-wise integer addition_

```default
cuda_tile.addi %lhs %rhs %overflow
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id299)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id299 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **overflow** ([IntegerOverflow](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-addi-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id300)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id300 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The sum of the input tiles. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id301)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id301 "Permalink to this headline")

The `addi` operation computes the element-wise addition of two tiles with integer element types.

$$
\[\text{addi}(x, y)_i = x_i + y_i\]
$$

Element-wise integer arithmetic operations are performed by the target architecture’s native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-integer-arithmetic) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation
respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id302)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id302 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
