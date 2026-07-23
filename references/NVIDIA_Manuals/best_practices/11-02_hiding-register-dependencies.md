---
title: "11.2. Hiding Register Dependencies"
section: "11.2"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#hiding-register-dependencies"
---

## [11.2. Hiding Register Dependencies](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#hiding-register-dependencies)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#hiding-register-dependencies "Permalink to this headline")

> **Note**
>
> **Medium Priority:** To hide latency arising from register dependencies, maintain sufficient numbers of active threads per multiprocessor (i.e., sufficient occupancy).

Register dependencies arise when an instruction uses a result stored in a register written by an instruction before it. The latency of most arithmetic instructions is typically 4 cycles on devices of compute capability 7.0. So threads must wait approximatly 4 cycles before using an arithmetic result. However, this latency can be completely hidden by the execution of threads in other warps. See [Registers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#registers) for details.
