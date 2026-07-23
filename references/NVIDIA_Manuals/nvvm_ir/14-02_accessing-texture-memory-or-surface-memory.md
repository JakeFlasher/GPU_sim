---
title: "14.2. Accessing Texture Memory or Surface Memory"
section: "14.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#accessing-texture-memory-or-surface-memory"
---

## [14.2. Accessing Texture Memory or Surface Memory](https://docs.nvidia.com/cuda/nvvm-ir-spec#accessing-texture-memory-or-surface-memory)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#accessing-texture-memory-or-surface-memory "Permalink to this headline")

Texture memory and surface memory can be accessed using texture or surface handles. NVVM provides the following intrinsic function to get a texture or surface handle from a texture or surface variable.

```c
delcare i64 %llvm.nvvm.texsurf.handle.p1i64(metadata, i64 addrspace(1)*)
```

The first argument to the intrinsic is a metadata holding the texture or surface variable. Such a metadata may hold only one texture or one surface variable. The second argument to the intrinsic is the texture or surface variable itself. The intrinsic returns a handle of `i64` type.

The returned handle value from the intrinsic call can be used as an operand (with a constraint of l) in a PTX inline asm to access the texture or surface memory.
