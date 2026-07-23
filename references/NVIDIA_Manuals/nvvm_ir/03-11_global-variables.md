---
title: "3.11. Global Variables"
section: "3.11"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-variables"
---

## [3.11. Global Variables](https://docs.nvidia.com/cuda/nvvm-ir-spec#global-variables)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#global-variables "Permalink to this headline")

A global variable, that is not an intrinsic global variable, may be optionally declared to reside in one of the following address spaces:

- `global`
- `shared`
- `constant`

If no address space is explicitly specified, the global variable is assumed to reside in the `global` address space with a generic address value. See [Address Space](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#address-space) for details.

`thread_local` variables are not supported.

No explicit section (except for the metadata section) is allowed.

Initializations of `shared` variables are not supported. Use undef initialization.
