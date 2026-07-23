---
title: "7.4. Printing Code Generation Statistics"
section: "7.4"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#printing-code-generation-statistics"
---

## [7.4. Printing Code Generation Statistics](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#printing-code-generation-statistics)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#printing-code-generation-statistics "Permalink to this headline")

A summary on the amount of used registers and the amount of memory needed per compiled device function can be printed by passing option `--resource-usage` to `nvcc`:

```text
$ nvcc --resource-usage acos.cu -arch sm_80
ptxas info    : 1536 bytes gmem
ptxas info    : Compiling entry function 'acos_main' for 'sm_80'
ptxas info    : Function properties for acos_main
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 6 registers, 1536 bytes smem, 32 bytes cmem[0]
```

As shown in the above example, the amount of statically allocated global memory (gmem) is listed.

Global memory and some of the constant banks are module scoped resources and not per kernel resources. Allocation of constant variables to constant banks is profile specific.

Followed by this, per kernel resource information is printed.

Stack frame is per thread stack usage used by this function. Spill stores and loads represent stores and loads done on stack memory which are being used for storing variables that couldn’t be allocated to physical registers.

Similarly number of registers, amount of shared memory and total space in constant bank allocated is shown.
