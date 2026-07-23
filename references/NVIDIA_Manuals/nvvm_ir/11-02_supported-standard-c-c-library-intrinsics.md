---
title: "11.2. Supported Standard C/C++ Library Intrinsics"
section: "11.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-standard-c-c-library-intrinsics"
---

## [11.2. Supported Standard C/C++ Library Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-standard-c-c-library-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-standard-c-c-library-intrinsics "Permalink to this headline")

- `llvm.copysign`

This is only supported in the modern NVVM IR dialect.
- `llvm.memcpy`

Note that the constant address space cannot be used as the destination since it is read-only.
- `llvm.memmove`

Note that the constant address space cannot be used since it is read-only.
- `llvm.memset`

Note that the constant address space cannot be used since it is read-only.
- `llvm.sqrt`

Supported for float/double and vector of float/double. Mapped to PTX `sqrt.rn.f32` and `sqrt.rn.f64`.
- `llvm.fma`

Supported for float/double and vector of float/double. Mapped to PTX `fma.rn.f32` and `fma.rn.f64`
