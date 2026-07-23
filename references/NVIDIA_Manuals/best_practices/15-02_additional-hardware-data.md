---
title: "15.2. Additional Hardware Data"
section: "15.2"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#additional-hardware-data"
---

## [15.2. Additional Hardware Data](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#additional-hardware-data)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#additional-hardware-data "Permalink to this headline")

Certain hardware features are not described by the compute capability. For example, the ability to overlap kernel execution with asynchronous data transfers between the host and the device is available on most but not all GPUs irrespective of the compute capability. In such cases, call `cudaGetDeviceProperties()` to determine whether the device is capable of a certain feature. For example, the `asyncEngineCount` field of the device property structure indicates whether overlapping kernel execution and data transfers is possible (and, if so, how many concurrent transfers are possible); likewise, the `canMapHostMemory` field indicates whether zero-copy data transfers can be performed.
