---
title: "8.4.5. cuda_tile.itof"
section: "8.4.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-itof"
---

### [8.4.5. cuda_tile.itof](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-itof)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-itof "Permalink to this headline")

_Convert integer to floating-point_

```default
cuda_tile.itof %from %signedness %rounding_mode
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id90)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id90 "Permalink to this headline")

- **from** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input integer tile. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-itof-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1
- **rounding_mode** ([RoundingMode](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-itof-roundingmode-attr)) - The rounding mode for the operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id91)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id91 "Permalink to this headline")

- **to** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[f16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [bf16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [f64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e4m3fn](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [fp8e5m2](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [tf32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The converted floating-point tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id92)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id92 "Permalink to this headline")

The `itof` operation converts an integer tile into a float tile.
In contrast to [cuda_tile.bitcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-bitcast), this preserves the numerical value of the tile,
rounded to the nearest floating-point number of the provided type.

> **Warning**
>
> Different floating-point types have different conversion behaviors for out-of-finite-range
> values and special values. See [Floating-Point Conversion: Special Value and Saturation Behavior](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#table-float-conversion-semantics) for more details.

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

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id93)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id93 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
