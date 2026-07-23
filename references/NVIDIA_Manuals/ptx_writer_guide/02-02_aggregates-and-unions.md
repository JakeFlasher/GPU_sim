---
title: "2.2. Aggregates and Unions"
section: "2.2"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#aggregates-and-unions"
---

## [2.2. Aggregates and Unions](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#aggregates-and-unions)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#aggregates-and-unions "Permalink to this headline")

Beyond the scalar types, PTX also supports native-vector types of these scalar types, with both its vector syntax and its byte-array syntax. For scalar types with a size no greater than four bytes, vector types with 1, 2, 3, and 4 elements exist; for all other types, only 1 and 2 element vector types exist.

All aggregates and unions can be supported in PTX with its byte-array syntax.

The following are the size-and-alignment rules for all aggregates and unions.

- For a non-native-vector type, an entire aggregate or union is aligned on the same boundary as its most strictly aligned member. This rule is not followed if the alignments are defined by the input language. For example, in OpenCL built-in vector data types have their alignment set to the size of the built-in data type in bytes.
- For a native vector type – discussed at the start of this section – the alignment is defined as follows. (For the definitions below, the native vector has n elements and has an element type t.)
  - For a vector with an odd number of elements, its alignment is the same as its member: alignof(t).
  - For a vector with an even number of elements, its alignment is set to number of elements times the alignment of its member: n*alignof(t).
- Each member is assigned to the lowest available offset with the appropriate alignment. This may require internal padding, depending on the previous member.
- The size of an aggregate or union, if necessary, is increased to make it a multiple of the alignment of the aggregate or union. This may require tail padding, depending on the last member.
