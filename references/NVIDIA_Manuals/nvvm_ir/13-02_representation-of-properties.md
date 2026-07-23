---
title: "13.2. Representation of Properties"
section: "13.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#representation-of-properties"
---

## [13.2. Representation of Properties](https://docs.nvidia.com/cuda/nvvm-ir-spec#representation-of-properties)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#representation-of-properties "Permalink to this headline")

For each translation unit (that is, per bitcode file), there is a named metadata called `nvvm.annotations`.

This named metadata contains a list of MDNodes.

The first operand of each MDNode is an entity that the node is annotating using the remaining operands.

Multiple MDNodes may provide annotations for the same entity, in which case their first operands will be same.

The remaining operands of the MDNode are organized in order as <property-name, value>.

- The property-name operand is MDString, while the value is `i32`.
- Starting with the operand after the annotated entity, every alternate operand specifies a property.
- The operand after a property is its value.

The following is an example.

```text
!nvvm.annotations = !{!12, !13}
  !12 = !{void (i32, i32)* @_Z6kernelii, !"kernel", i32 1}
  !13 = !{void ()* @_Z7kernel2v, !"kernel", i32 1, !"maxntidx", i32 16}
```

If two bitcode files are being linked and both have a named metadata `nvvm.annotations`, the linked file will have a single merged named metadata. If both files define properties for the same entity foo , the linked file will have two MDNodes defining properties for foo . It is illegal for the files to have conflicting properties for the same entity.
