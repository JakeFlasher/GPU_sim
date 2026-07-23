---
title: "10.33. Special Registers: %current_graph_exec"
section: "10.33"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-current-graph-exec"
---

## [10.33. Special Registers: %current_graph_exec](https://docs.nvidia.com/cuda/parallel-thread-execution#special-registers-current-graph-exec)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-current-graph-exec "Permalink to this headline")

**`%current_graph_exec`**

An Identifier for currently executing CUDA device graph.

**Syntax (predefined)**

```text
.sreg .u64 %current_graph_exec;
```

**Description**

A predefined, read-only special register initialized with the identifier referring to the CUDA
device graph being currently executed. This register is 0 if the executing kernel is not part of a
CUDA device graph.

Refer to the _CUDA Programming Guide_ for more details on CUDA device graphs.

**PTX ISA Notes**

Introduced in PTX ISA version 8.0.

**Target ISA Notes**

Requires `sm_50` or higher.

**Examples**

```text
mov.u64  r1, %current_graph_exec;
```
