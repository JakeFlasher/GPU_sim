---
title: "10.6. Memory Access and Addressing Operations"
section: "10.6"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#memory-access-and-addressing-operations"
---

## [10.6. Memory Access and Addressing Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#memory-access-and-addressing-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#memory-access-and-addressing-operations "Permalink to this headline")

### [10.6.1. alloca Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#alloca-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#alloca-instruction "Permalink to this headline")

The `alloca` instruction returns a generic pointer to the local address space. The `inalloca` attribute is not supported. Maximum alignment supported is 2^23. The `addrspace(<num>)` specifier is supported only if `num` is 0.

### [10.6.2. load Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#load-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#load-instruction "Permalink to this headline")

`load atomic` is not supported.

### [10.6.3. store Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#store-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#store-instruction "Permalink to this headline")

`store atomic` is not supported.

### [10.6.4. fence Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#fence-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#fence-instruction "Permalink to this headline")

Not supported. Use NVVM intrinsic functions instead.

### [10.6.5. cmpxchg Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#cmpxchg-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#cmpxchg-instruction "Permalink to this headline")

Supported for `i32`, `i64`, and `i128` types, with the following restrictions:

- The pointer operand must be either a global pointer, a shared pointer, or a generic pointer that points to either the `global` address space or the `shared` address space.
- The `weak` marker and the `failure ordering` are accepted and ignored.
- The `i128` type is only supported on `compute_90` and above.
- The pointer passed into `cmpxchg` must have alignment greater than or equal to the size in memory of the operand.

### [10.6.6. atomicrmw Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#atomicrmw-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#atomicrmw-instruction "Permalink to this headline")

Only the following operations are supported:

- `xchg`
- `add`
- `sub`
- `and`
- `or`
- `xor`
- `max`
- `min`
- `umax`
- `umin`

All other operations are not supported.

The operations support the `i32` and `i64` types. The `xchg` operation additionally supports `i128` on `compute_90` and above.

- The pointer operand must be either a global pointer, a shared pointer, or a generic pointer that points to either the `global` address space or the `shared` address space.
- The pointer passed into `atomicrmw` must have alignment greater than or equal to the size in memory of the operand.

### [10.6.7. getelementptr Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#getelementptr-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#getelementptr-instruction "Permalink to this headline")

Supported.
