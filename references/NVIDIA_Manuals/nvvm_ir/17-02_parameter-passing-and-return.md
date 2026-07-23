---
title: "17.2. Parameter Passing and Return"
section: "17.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#parameter-passing-and-return"
---

## [17.2. Parameter Passing and Return](https://docs.nvidia.com/cuda/nvvm-ir-spec#parameter-passing-and-return)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#parameter-passing-and-return "Permalink to this headline")

The following table shows the mapping of function argument and return types in NVVM IR to PTX types.

| Source Type | Size in Bits | PTX Type |
| --- | --- | --- |
| Integer types | <= 32 | `.u32` or `.b32` (zero-extended if unsigned)   `.s32` or `.b32` (sign-extended if signed) |
| 64 | `.u64` or `.b64` (if unsigned)   `.s64` or `.b64` (if signed) |  |
| Pointer types (without `byval` attribute) | 32 | `.u32` or `.b32` |
|  | 64 | `.u64` or `.b64` |
| Floating-point types | 32 | `.f32` or `.b32` |
| 64 | `.f64` or `.b64` |  |
| Aggregate types | Any size | `.align`_align_ `.b8`_name_[_size_]   Where _align_ is overall aggregate or vector alignment in bytes, _name_ is variable name associated with aggregate or vector, and _size_ is the aggregate or vector size in bytes. |
| Pointer types to aggregate with `byval` attribute | 32 or 64 |  |
| Vector type | Any size |  |
