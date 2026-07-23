---
title: "3. nvdisasm"
section: "3"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvdisasm"
---

# [3. nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities#nvdisasm)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvdisasm "Permalink to this headline")

`nvdisasm` extracts information from standalone cubin files and presents them in human readable format. The output of `nvdisasm` includes CUDA assembly code for each kernel, listing of ELF data sections and other CUDA specific sections. Output style and options are controlled through `nvdisasm` command-line options. `nvdisasm` also does control flow analysis to annotate jump/branch targets and makes the output easier to read.

> **Note**
>
> `nvdisasm` requires complete relocation information to do control flow analysis. If this information is missing from the CUDA binary, either use the `nvdisasm` option `-ndf` to turn off control flow analysis, or use the `ptxas` and `nvlink` option `-preserve-relocs` to re-generate the cubin file.

For a list of CUDA assembly instruction set of each GPU architecture, see [Instruction Set Reference](https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-ref).
