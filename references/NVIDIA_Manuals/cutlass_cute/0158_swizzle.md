---
title: "Swizzle"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#swizzle"
---

### [Swizzle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#swizzle)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#swizzle "Permalink to this headline")

`Swizzle` is a transformation that permutes the elements of a layout. Swizzles are used to rearrange data elements to improve memory access patterns and computational efficiency, particularly for avoiding bank conflicts in shared memory.

**Swizzle Parameters:**

A swizzle is defined by three parameters:

- **MBase**: The number of least-significant bits to keep constant
- **BBits**: The number of bits in the mask
- **SShift**: The distance to shift the mask

**Bit Pattern:**

```text
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase (least-sig bits kept constant)
                 ^-^       ^-^     BBits (number of bits in mask)
                   ^---------^     SShift (distance to shift YYY)
                                      (positive: right, negative: left)

Given:    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
Result:   0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx
          where AA = ZZ xor YY
```

**Usage:**

Swizzles are typically created using CuTe’s swizzle factory functions and composed with layouts to create optimized memory access patterns.
