---
title: "Operation structs"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html#operation-structs"
---

### [Operation structs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#operation-structs)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#operation-structs "Permalink to this headline")

#### [Location of files](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#location-of-files)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#location-of-files "Permalink to this headline")

CuTe provides its Operations structs in the
[`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch)
directory, in header files starting with `mma`.

#### [Operation struct’s name](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#operation-struct-s-name)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#operation-struct-s-name "Permalink to this headline")

A CuTe Operation struct’s name principally encodes the PTX instruction it wraps.
These often include

- its first supported architecture,
- the M, N, and K dimensions that it accepts,
- the types that it takes, and
- the arrangement of the A and B inputs.

For example, the Volta section below will refer to the
`SM70_8x8x4_F32F16F16F32_NT` Operation struct defined in
[`include/cute/arch/mma_sm70.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/mma_sm70.hpp).

- “SM70” refers to Volta.
- “8x8x4” refers to M = 8, N = 8, and K = 4,
the dimensions of the MMA operation that the quadpair performs
(see below). This is reflected in the PTX as `.m8n8k4.`.
- “F32F16F16F32” refers to the element types
of the four matrix operands A, B, C, and D.
An MMA computes D = C + A * B,
so we read the types from left to right:
D is F32 (`float`), A is F16 (half),
B is F16 (half), and C is F32 (`float`). This is reflected in the PTX instruction name as `.f32.f16.f16.f32`.
- “NT” means that the PTX instruction is designed for inputs A as M-major (not transposed, column-major)
and inputs B as N-major (transposed, row-major). This is reflected in the PTX instruction name as `.col.row.`.

#### [Contents](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#contents)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#contents "Permalink to this headline")

An Operation struct has the following members.

##### [Type aliases](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#type-aliases)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#type-aliases "Permalink to this headline")

An Operation struct has four public type aliases:
`DRegisters`, `ARegisters`, `BRegisters`, and `CRegisters`.
For example, the `SM70_8x8x4_F32F16F16F32_NT` Operation struct defined in
[`include/cute/arch/mma_sm70.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/mma_sm70.hpp)
defines these as follows.

```c++
using DRegisters = float[8];
using ARegisters = uint32_t[2];
using BRegisters = uint32_t[2];
using CRegisters = float[8];
```

This shows how many values each thread will pass into the PTX instruction
for each of the matrices A, B, C, and D.  For this Operation,
each thread passes 8 F32 values each for C and D (hence `float[8]`),
and 4 F16 values each for A and B (hence `uint32_t[2]`;
the instruction packs two 16-bit F16 values
in each of the two 32-bit `uint32_t` values).

##### [fma static member device function](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#fma-static-member-device-function)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#fma-static-member-device-function "Permalink to this headline")

An operation struct defines a public `static void fma` function.
It is marked with the `CUTE_HOST_DEVICE` macro,
which adds the `__host__ __device__` annotations.
Different Operations define `fma` to take different numbers of arguments,
depending on the PTX MMA instruction.
The implementation protects use of the PTX instruction with a macro,
and raises an `assert` if `fma` is called when the macro is not defined.
This ensures that tests and examples that use this Operation in an Atom
can still compile, even if the PTX instruction is not available.
