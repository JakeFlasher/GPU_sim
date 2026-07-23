---
title: "3.15. Named Metadata"
section: "3.15"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#named-metadata"
---

## [3.15. Named Metadata](https://docs.nvidia.com/cuda/nvvm-ir-spec#named-metadata)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#named-metadata "Permalink to this headline")

Accepted and ignored, except for the following:

- `!nvvm.annotations`: see [Global Property Annotation](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#global-property-annotation-chapter-11)
- `!nvvmir.version`
- `!llvm.dbg.cu`
- `!llvm.module.flags`

The NVVM IR version is specified using a named metadata called `!nvvmir.version`. The `!nvvmir.version` named metadata may have one metadata node that contains the NVVM IR version for that module. If multiple such modules are linked together, the named metadata in the linked module may have more than one metadata node with each node containing a version. A metadata node with NVVM IR version takes either of the following forms:

- It may consist of two i32 values—the first denotes the NVVM IR major version number and the second denotes the minor version number. If absent, the version number is assumed to be 1.0, which can be specified as:

```text
!nvvmir.version = !{!0}
!0 = !{i32 1, i32 0}
```
- It may consist of four i32 values—the first two denote the NVVM IR major and minor versions respectively. The third value denotes the NVVM IR debug metadata major version number, and the fourth value denotes the corresponding minor version number. If absent, the version number is assumed to be 1.0, which can be specified as:

```text
!nvvmir.version = !{!0}
!0 = !{i32 1, i32 0, i32 1, i32 0}
```

The version of NVVM IR described in this document is 2.0. The version of NVVM IR debug metadata described in this document is 3.1.
