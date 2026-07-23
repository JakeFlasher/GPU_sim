---
title: "8.7.10. cuda_tile.divf"
section: "8.7.10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-divf"
---

### [8.7.10. cuda_tile.divf](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-divf)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-divf "Permalink to this headline")

_Element-wise floating-point division_

```default
cuda_tile.divf %lhs %rhs %rounding_mode %flush_to_zero
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id199)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id199 "Permalink to this headline")

- **lhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The dividend input floating-point tile. 13.1
- **rhs** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The divisor input floating-point tile. 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-divf-roundingmode-attr)) - The rounding mode for the operation. 13.1
- **flush_to_zero** ([Flag](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-flag)) - If set, flushes subnormal inputs and results to sign-preserving zero. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id200)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id200 "Permalink to this headline")

- **result** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result of the `divf` operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id201)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id201 "Permalink to this headline")

The `divf` operation computes the element-wise division of two input tiles
with floating-point element types.

The `approx` rounding mode implements a fast approximation of divide,
computed as a multiplication by reciprocal. For `|rhs|` in normalized range
`[2^(-126), 2^(126)]` the maximum ULP (Unit in the Last Place) error is `2`.
For `2^(126) < |rhs| < 2^(128)`, if `lhs` is infinity the operation returns `NaN`,
otherwise `0`.

The `full` rounding mode implements a relatively fast, full-range
approximation that scales operands to achieve better accuracy, but is not fully
IEEE 754 compliant. The maximum ulp error is 2 across the full range of inputs.

$$
\[\text{div(lhs, rhs)}_i = \text{lhs}_i / \text{rhs}_i\]
$$

Element-wise floating-point arithmetic operations are performed by the target architecture’s native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See [Floating-Point Arithmetic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-section-floating-point-arithmetic) for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

The below table shows the supported modifiers and rounding modes for each data type. Entries with ‘*’ are emulated in f32.

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
| --- | --- | --- | --- | --- |
| `flush_to_zero` | yes | no | no | no |
| `approx` | yes | no | no | no |
| `full` | yes | no | no | no |
| `rounding<nearest_even>` | yes | yes | yes* | yes* |
| `rounding<zero>` | yes | yes | yes* | yes* |
| `rounding<negative_inf>` | yes | yes | yes* | yes* |
| `rounding<positive_inf>` | yes | yes | yes* | yes* |

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id202)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id202 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>).
- Operation must infer result types from operands and attributes.
