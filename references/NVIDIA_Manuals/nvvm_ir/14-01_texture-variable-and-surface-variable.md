---
title: "14.1. Texture Variable and Surface Variable"
section: "14.1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#texture-variable-and-surface-variable"
---

## [14.1. Texture Variable and Surface Variable](https://docs.nvidia.com/cuda/nvvm-ir-spec#texture-variable-and-surface-variable)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#texture-variable-and-surface-variable "Permalink to this headline")

A texture or a surface variable can be declared/defined as a global variable of `i64` type with annotation `texture` or `surface` in the `global` address space.

A texture or surface variable must have a name, which must follow identifier naming conventions.

It is illegal to store to or load from the address of a texture or surface variable. A texture or a surface variable may only have the following uses:

- In a metadata node
- As an intrinsic function argument as shown below
- In `llvm.used` Global Variable
