---
title: "15.1. Atomic"
section: "15.1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#atomic"
---

## [15.1. Atomic](https://docs.nvidia.com/cuda/nvvm-ir-spec#atomic)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#atomic "Permalink to this headline")

Besides the atomic instructions, the following extra atomic intrinsic functions are supported.

```text
declare float @llvm.nvvm.atomic.load.add.f32.p0f32(float* address, float val)
declare float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* address, float val)
declare float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* address, float val)
declare double @llvm.nvvm.atomic.load.add.f64.p0f64(double* address, double val)
declare double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* address, double val)
declare double @llvm.nvvm.atomic.load.add.f64.p3f64(double addrspace(3)* address, double val)
```

reads the single/double precision floating point value `old` located at the address `address`, computes `old+val`, and stores the result back to memory at the same address. These operations are performed in one atomic transaction. The function returns `old`.

```text
declare i32 @llvm.nvvm.atomic.load.inc.32.p0i32(i32* address, i32 val)
declare i32 @llvm.nvvm.atomic.load.inc.32.p1i32(i32 addrspace(1)* address, i32 val)
declare i32 @llvm.nvvm.atomic.load.inc.32.p3i32(i32 addrspace(3)* address, i32 val)
```

reads the 32-bit word `old` located at the address `address`, computes `((old >= val) ? 0 : (old+1))`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

```text
declare i32 @llvm.nvvm.atomic.load.dec.32.p0i32(i32* address, i32 val)
declare i32 @llvm.nvvm.atomic.load.dec.32.p1i32(i32 addrspace(1)* address, i32 val)
declare i32 @llvm.nvvm.atomic.load.dec.32.p3i32(i32 addrspace(3)* address, i32 val)
```

reads the 32-bit word `old` located at the address `address`, computes `(((old == 0) | (old > val)) ? val : (old-1) )`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.
