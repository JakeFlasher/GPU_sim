---
title: "3.2. Calling Conventions"
section: "3.2"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#calling-conventions"
---

## [3.2. Calling Conventions](https://docs.nvidia.com/cuda/nvvm-ir-spec#calling-conventions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#calling-conventions "Permalink to this headline")

All LLVM calling convention markings are accepted and ignored. Functions and calls are generated according to the PTX calling convention.

### [3.2.1. Rules and Restrictions](https://docs.nvidia.com/cuda/nvvm-ir-spec#rules-and-restrictions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#rules-and-restrictions "Permalink to this headline")

1. When an argument with width less than 32-bit is passed, the `zeroext/signext` parameter attribute should be set. `zeroext` will be assumed if not set.
2. When a value with width less than 32-bit is returned, the `zeroext/signext` parameter attribute should be set. `zeroext` will be assumed if not set.
3. Arguments of aggregate or vector types that are passed by value can be passed by pointer with the `byval` attribute set (referred to as the `by-pointer-byval` case below). The align attribute must be set if the type requires a non-natural alignment (natural alignment is the alignment inferred for the aggregate type according to the [Data Layout](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout) section).
4. If a function has an argument of aggregate or vector type that is passed by value directly and the type has a non-natural alignment requirement, the alignment must be annotated by the global property annotation <`align`, alignment>, where alignment is a 32-bit integer whose upper 16 bits represent the argument position (starting from 1) and the lower 16 bits represent the alignment.
5. If the return type of a function is an aggregate or a vector that has a non-natural alignment, then the alignment requirement must be annotated by the global property annotation <`align`, alignment>, where the upper 16 bits is 0, and the lower 16 bits represent the alignment.
6. It is not required to annotate a function with <`align`, alignment> otherwise. If annotated, the alignment must match the natural alignment or the align attribute in the `by-pointer-byval` case.
7. For an indirect call instruction of a function that has a non-natural alignment for its return value or one of its arguments that is not expressed in alignment in the `by-pointer-byval` case, the call instruction must have an attached metadata of kind `callalign`. The metadata contains a sequence of `i32` fields each of which represents a non-natural alignment requirement. The upper 16 bits of an `i32` field represent the argument position (0 for return value, 1 for the first argument, and so on) and the lower 16 bits represent the alignment. The `i32` fields must be sorted in the increasing order.

For example,

```c
%call = call %struct.S %fp1(%struct.S* byval align 8 %arg1p, %struct.S %arg2),!callalign !10
!10 = !{i32 8, i32 520};
```
8. It is not required to have an `i32` metadata field for the other arguments or the return value otherwise. If presented, the alignment must match the natural alignment or the align attribute in the `by-pointer-byval case`.
9. It is not required to have a `callalign` metadata attached to a direct call instruction. If attached, the alignment must match the natural alignment or the alignment in the `by-pointer-byval` case.
10. The absence of the metadata in an indirect call instruction means using natural alignment or the align attribute in the `by-pointer-byval` case.
