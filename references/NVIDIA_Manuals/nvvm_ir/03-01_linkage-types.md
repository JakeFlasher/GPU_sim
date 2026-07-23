---
title: "3.1. Linkage Types"
section: "3.1"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#linkage-types"
---

## [3.1. Linkage Types](https://docs.nvidia.com/cuda/nvvm-ir-spec#linkage-types)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#linkage-types "Permalink to this headline")

Supported:

- `private`
- `internal`
- `available_externally`
- `linkonce`
- `weak`
- `common`
- `linkonce_odr`
- `weak_odr`
- `external`

All other linkage types are not supported.

See [NVVM ABI for PTX](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-abi-for-ptx) for details on how linkage types are translated to PTX.
