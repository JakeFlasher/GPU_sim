---
title: "8.4.2. cuda_tile.exti"
section: "8.4.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-exti"
---

### [8.4.2. cuda_tile.exti](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-exti)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-exti "Permalink to this headline")

_Extend the width of an integer tile_

```default
cuda_tile.exti %from %signedness
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id78)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id78 "Permalink to this headline")

- **from** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The input integer tile to extend. 13.1
- **signedness** ([Signedness](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-exti-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id79)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id79 "Permalink to this headline")

- **to** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The extended integer tile. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id80)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id80 "Permalink to this headline")

The `exti` operation converts a tile of integers of a given width to a
strictly larger width. Zero-extension is used
for `unsigned` integers and sign-extension is used for `signed`
integers.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id81)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id81 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
