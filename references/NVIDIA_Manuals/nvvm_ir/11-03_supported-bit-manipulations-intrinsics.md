---
title: "11.3. Supported Bit Manipulations Intrinsics"
section: "11.3"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-bit-manipulations-intrinsics"
---

## [11.3. Supported Bit Manipulations Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-bit-manipulations-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-bit-manipulations-intrinsics "Permalink to this headline")

- `llvm.bitreverse`

Supported for `i8`, `i16`, `i32`, and `i64`.
- `llvm.bswap`

Supported for `i16`, `i32`, and `i64`.
- `llvm.ctpop`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.ctlz`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.cttz`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.fshl`

Supported for `i8`, `i16`, `i32`, and `i64`.
- `llvm.fshr`

Supported for `i8`, `i16`, `i32`, and `i64`.
