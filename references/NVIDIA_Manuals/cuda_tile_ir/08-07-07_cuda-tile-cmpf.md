---
title: "8.7.7. cuda_tile.cmpf"
section: "8.7.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-cmpf"
---

### [8.7.7. cuda_tile.cmpf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cmpf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cmpf "Permalink to this headline")

_Element-wise floating-point comparison_

```default
cuda_tile.cmpf %comparison_predicate %comparison_ordering %lhs %rhs
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id185)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id185 "Permalink to this headline")

- **comparison_predicate** ([ComparisonPredicate](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-cmpf-comparisonpredicate-attr)) - The comparison predicate. 13.1
- **comparison_ordering** ([ComparisonOrdering](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-cmpf-comparisonordering-attr)) - The comparison ordering. 13.1
- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The left hand side operand. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The right hand side operand. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id186)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id186 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the comparison. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id187)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id187 "Permalink to this headline")

The `cmpf` operation is a generic comparison for float-like types. The
operands must have the same shape and type, and this type must be a float type.

The result is `1` if the comparison is true and `0` otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.

$$
\[\begin{split}\text{cmpf}(x, y, \text{pred})_i = \begin{cases}
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

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison.
- `ordered` - Ordered comparison.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id188)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id188 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs` and `rhs` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Result type has i1 element type and same shape as operands
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id189)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id189 "Permalink to this headline")

```mlir
%lhs0 = constant <f16: 0.0> : tile<f16>
%rhs0 = constant <f16: 0.0> : tile<f16>

// Custom form of scalar "ordered equal" comparison.
%x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

%lhs1 = constant <f16: 0.0> : tile<2x2xf16>
%rhs1 = constant <f16: 0.0> : tile<2x2xf16>

// Custom form of scalar "unordered less than" comparison.
%x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

%lhs2 = constant <f64: 0.0> : tile<2x2xf64>
%rhs2 = constant <f64: 0.0> : tile<2x2xf64>
```

See [cuda_tile.cmpf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-cmpf-0) for the full example listing.
