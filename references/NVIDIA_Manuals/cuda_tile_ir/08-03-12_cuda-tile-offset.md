---
title: "8.3.12. cuda_tile.offset"
section: "8.3.12"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-offset"
---

### [8.3.12. cuda_tile.offset](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-offset)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-offset "Permalink to this headline")

_Offsets a tile of pointers_

```default
cuda_tile.offset %ptr %offset
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id46)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id46 "Permalink to this headline")

- **ptr** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The base pointer tile to advance. 13.1
- **offset** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[i1](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i8](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i16](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) | [i64](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)>) - The offset tile to add to the pointer. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id47)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id47 "Permalink to this headline")

- **result** ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)) - The resulting pointer tile after advancement. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id48)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id48 "Permalink to this headline")

`offset` advances a tile of pointers. It takes `ptr` as base
and `offset` as increment, and performs element-wise addition of
`ptr` by `offset`:

$$
\[\text{offset}(\text{ptr}, \text{offset})_i = \text{ptr}_i + \text{offset}_i \times \text{bitwidth}\]
$$

```mlir
result[i,j] = ptr[i,j] + offset[i,j] * bitwidth
```

`ptr` is interpreted as an unsigned integer. `offset` is
interpreted as a signed integer. `bitwidth` is the storage bitwidth
of the pointee type. The multiplication must not overflow (wrap-around) in
a signed sense. The addition must not overflow (wrap-around) in an unsigned
sense. In case of an overflow, the result is undefined.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id49)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id49 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must apply element-wise to its operands.
- `ptr`, `offset` and `result` must have the same shape.
- `result` and `ptr` must have the same shape and element type ([ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)).
- Operation must infer result types from operands and attributes.
