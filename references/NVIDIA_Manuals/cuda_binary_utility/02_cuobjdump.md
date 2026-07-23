---
title: "2. cuobjdump"
section: "2"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump"
---

# [2. cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities#cuobjdump)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump "Permalink to this headline")

`cuobjdump` extracts information from CUDA binary files (both standalone and those embedded in host binaries) and presents them in human readable format. The output of `cuobjdump` includes CUDA assembly code for each kernel, CUDA ELF section headers, string tables, relocators and other CUDA specific sections. It also extracts embedded ptx text from host binaries.

For a list of CUDA assembly instruction set of each GPU architecture, see [Instruction Set Reference](https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-ref).
