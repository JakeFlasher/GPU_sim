---
title: "13.3. Supported Properties"
section: "13.3"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-properties"
---

## [13.3. Supported Properties](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-properties)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-properties "Permalink to this headline")

| Property Name | Annotated On | Description |
| --- | --- | --- |
| `maxntid{x, y, z}` | kernel function | Maximum expected CTA size from any launch. |
| `reqntid{x, y, z}` | kernel function | Minimum expected CTA size from any launch. |
| `cluster_dim_{x,y,z}` | kernel function | Support for cluster dimensions for Hopper+. If any dimension is specified as 0, then all dimensions must be specified as 0. |
| `cluster_max_blocks` | kernel function | Maximum number of blocks per cluster. Must be non-zero. Only supported for Hopper+. |
| `minctasm` | kernel function | Hint/directive to the compiler/driver, asking it to put at least these many CTAs on an SM. |
| `grid_constant` | kernel function | The argument is a metadata node, which contains a list of integers, where each integer n denotes that the nth parameter has the grid_constant annotation (numbering from 1). The parameter’s type must be of pointer type with byval attribute set. It is undefined behavior to write to memory pointed to by the parameter. This property is only supported for Volta+. |
| `maxnreg` | function | Maximum number of registers for function. |
| `kernel` | function | Signifies that this function is a kernel function. |
| `align` | function | Signifies that the value in low 16-bits of the 32-bit value contains alignment of n th parameter type if its alignment is not the natural alignment. n is specified by high 16-bits of the value. For return type, n is 0. |
| `texture` | global variable | Signifies that variable is a texture. |
| `surface` | global variable | Signifies that variable is a surface. |
| `managed` | global variable | Signifies that variable is a UVM managed variable. |
