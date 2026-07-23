---
title: "8.4.3. cuda_tile.ftof"
section: "8.4.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-ftof"
---

### [8.4.3. cuda_tile.ftof](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ftof)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ftof "Permalink to this headline")

_Convert between floating-point types_

```default
cuda_tile.ftof %from %rounding_mode
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id82)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id82 "Permalink to this headline")

- **from** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-ftof-roundingmode-attr)) - The rounding mode for the operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id83)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id83 "Permalink to this headline")

- **to** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result floating-point tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id84)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id84 "Permalink to this headline")

The `ftof` operation converts a tile of a given floating-point element type into one
of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding_mode` attribute specifies the rounding behavior for the operation.
Only `NEAREST_EVEN` rounding mode is supported.

> **Warning**
>
> Different floating-point types have different conversion behaviors for out-of-finite-range
> values and special values. See [Floating-Point Conversion: Special Value and Saturation Behavior](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#table-float-conversion-semantics) for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id85)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id85 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
