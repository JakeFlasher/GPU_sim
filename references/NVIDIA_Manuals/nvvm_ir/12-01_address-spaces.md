---
title: "12.1. Address Spaces"
section: "12.1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-spaces"
---

## [12.1. Address Spaces](https://docs.nvidia.com/cuda/nvvm-ir-spec#address-spaces)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-spaces "Permalink to this headline")

NVVM IR has a set of predefined memory address spaces, whose semantics are similar to those defined in CUDA C/C++, OpenCL C and PTX. Any address space not listed below is not supported .

| Name | Address Space Number | Semantics/Example |
| --- | --- | --- |
| code | 0 | functions, code   - CUDA C/C++ function - OpenCL C function |
| generic | 0 | Can only be used to qualify the pointee of a pointer   - Pointers in CUDA C/C++ |
| global | 1 | - CUDA C/C++ __device__ - OpenCL C global |
| shared | 3 | - CUDA C/C++ __shared__ - OpenCL C local |
| constant | 4 | - CUDA C/C++ __constant__ - OpenCL C constant |
| local | 5 | - CUDA C/C++ local - OpenCL C private |
| <reserved> | 2, 101 and above |  |

Each global variable, that is not an intrinsic global variable, can be declared to reside in a specific non-zero address space, which can only be one of the following: `global`, `shared` or `constant`.

If a non-intrinsic global variable is declared without any address space number or with the address space number 0, then this global variable resides in address space `global` and the pointer of this global variable holds a generic pointer value.

The predefined NVVM memory spaces are needed for the language front-ends to model the memory spaces in the source languages. For example,

```text
// CUDA C/C++
__constant__ int c;
__device__ int g;

; NVVM IR
@c = addrspace(4) global i32 0, align 4
@g = addrspace(1) global [2 x i32] zeroinitializer, align 4
```

Address space numbers 2 and 101 or higher are reserved for NVVM compiler internal use only. No language front-end should generate code that uses these address spaces directly.
