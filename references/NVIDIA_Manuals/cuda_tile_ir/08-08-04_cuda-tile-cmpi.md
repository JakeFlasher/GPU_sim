---
title: "8.8.4. cuda_tile.cmpi"
section: "8.8.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-cmpi"
---

### [8.8.4. cuda_tile.cmpi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cmpi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cmpi "Permalink to this headline")

_Element-wise integer comparison_

```default
cuda_tile.cmpi %comparison_predicate %lhs %rhs %signedness
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id303)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id303 "Permalink to this headline")

- **comparison_predicate** ([ComparisonPredicate](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-cmpi-comparisonpredicate-attr)) - The comparison predicate. 13.1
- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-cmpi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id304)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id304 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the comparison. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id305)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id305 "Permalink to this headline")

The `cmpi` operation is a generic comparison for integer-like types. The
operands must have the same shape and type, and this type must be an integer type.
The result type has i1 element type and the same shape as the operands.

The result is `1` if the comparison is true and `0` otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.

$$
\[\begin{split}\text{cmpi}(x, y, \text{pred})_i = \begin{cases}
  1 & \text{if } x_i \text{ pred } y_i \\
  0 & \text{otherwise}
\end{cases}\end{split}\]
$$

The `comparison_predicate` attribute specifies the kind of comparison to be performed.

- `equal` - Equal comparison.
- `not_equal` - Not equal comparison.
- `less_than` - Less than comparison.
- `less_than_or_equal` - Less than or equal comparison.
- `greater_than` - Greater than comparison.
- `greater_than_or_equal` - Greater than or equal comparison.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id306)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id306 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs` and `rhs` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Result type has i1 element type and same shape as operands
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id307)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id307 "Permalink to this headline")

```mlir
%lhs0 = constant <i16: 0> : tile<i16>
%rhs0 = constant <i16: 0> : tile<i16>

// Scalar "signed less than" comparison.
%x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

%lhs1 = constant <i64: 0> : tile<2x2xi64>
%rhs1 = constant <i64: 0> : tile<2x2xi64>

// Tile equality comparison.
// There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
%x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
```

See [cuda_tile.cmpi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-cmpi-0) for the full example listing.
