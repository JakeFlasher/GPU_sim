---
title: "1.2. Differences between cuobjdump and nvdisasm"
section: "1.2"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#differences-between-cuobjdump-and-nvdisasm"
---

## [1.2. Differences between cuobjdump and nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities#differences-between-cuobjdump-and-nvdisasm)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#differences-between-cuobjdump-and-nvdisasm "Permalink to this headline")

CUDA provides two binary utilities for examining and disassembling cubin files and host executables: `cuobjdump` and `nvdisasm`. Basically, `cuobjdump` accepts both cubin files and host binaries while `nvdisasm` only accepts cubin files; but `nvdisasm` provides richer output options.

Here’s a quick comparison of the two tools:

|  | `cuobjdump` | `nvdisasm` |
| --- | --- | --- |
| Disassemble cubin | Yes | Yes |
| Extract ptx and extract and disassemble cubin from the following input files:   - Host binaries   - Executables   - Object files   - Static libraries - External fatbinary files | Yes | No |
| Control flow analysis and output | No | Yes |
| Advanced display options | No | Yes |
