---
title: "8.4.4. cuda_tile.ftoi"
section: "8.4.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-ftoi"
---

### [8.4.4. cuda_tile.ftoi](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ftoi)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ftoi "Permalink to this headline")

_Convert a tile from floating-point values to integer values_

```default
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id86)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id86 "Permalink to this headline")

- **from** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input floating-point tile. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-ftoi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-ftoi-roundingmode-attr)) - The rounding mode for the operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id87)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id87 "Permalink to this headline")

- **to** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The result integer tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id88)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id88 "Permalink to this headline")

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a [cuda_tile.bitcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-bitcast) which is bits preserving, this preserves the numerical
value of the tile, rounded towards zero to the nearest integer of the provided type.

The `rounding_mode` attribute specifies the rounding behavior for the operation.
Only `NEAREST_INT_TO_ZERO` rounding mode is supported.

> **Warning**
>
> If the input floating-point value, after being rounded, is outside the
> (signed or unsigned) range of the target integer type, the closest
> representable value is used instead. `NaN` values are converted to 0.
> Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id89)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id89 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
