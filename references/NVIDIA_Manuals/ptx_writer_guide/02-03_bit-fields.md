---
title: "2.3. Bit Fields"
section: "2.3"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#bit-fields"
---

## [2.3. Bit Fields](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#bit-fields)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#bit-fields "Permalink to this headline")

C structure and union definitions may have bit fields that define integral objects with a specified number of bits.

| Bit Field Type | Width w | Range |
| --- | --- | --- |
| signed char | 1 to 8 | -2<sup>w-1</sup> to 2<sup>w-1</sup> - 1 |
| unsigned char | 1 to 8 | 0 to 2<sup>w</sup> - 1 |
| signed short | 1 to 16 | -2<sup>w-1</sup> to 2<sup>w-1</sup> - 1 |
| unsigned short | 1 to 16 | 0 to 2<sup>w</sup> - 1 |
| signed int | 1 to 32 | -2<sup>w-1</sup> to 2<sup>w-1</sup> - 1 |
| unsigned int | 1 to 32 | 0 to 2<sup>w</sup> - 1 |
| signed long long | 1 to 64 | -2<sup>w-1</sup> to 2<sup>w-1</sup> - 1 |
| unsigned long long | 1 to 64 | 0 to 2<sup>w</sup> - 1 |

Current GPUs only support little-endian memory, so the below assumes little-endian layout.

The following are rules that apply to bit fields.

- Plain bit fields (neither signed nor unsigned is specified) are treated as signed.
- When no type is provided (e.g., signed : 6 is specified), the type defaults to int.

Bit fields obey the same size and alignment rules as other structure and union members, with the following modifications.

- Bit fields are allocated in memory from right to left (least to more significant) for little endian.
- A bit field must entirely reside in a storage unit appropriate for its declared type. A bit field should never cross its unit boundary.
- Bit fields may share a storage unit with other structure and union members, including members that are not bit fields, as long as there is enough space within the storage unit.
- Unnamed bit fields do not affect the alignment of a structure or union.
- Zero-length bit fields force the alignment of the following member of a structure to the next alignment boundary corresponding to the bit-field type. An unnamed, zero-length bit field will not force the external alignment of the structure to that boundary. If an unnamed, zero-length bit field has a stricter alignment than the external alignment, there is no guarantee that the stricter alignment will be maintained when the structure or union gets allocated to memory.

The following figures contain examples of bit fields. Figure 1 shows the byte offsets (upper corners) and the bit numbers (lower corners) that are used in the examples. The remaining figures show different bit-field examples.

![Bit Numbering](images/bit-fields_1.png)

Bit Numbering

![Bit-field Allocation](images/bit-fields_2.png)

Bit-field Allocation

![Boundary Alignment](images/bit-fields_3.png)

Boundary Alignment

![Storage Unit Sharing](images/bit-fields_4.png)

Storage Unit Sharing

![Union Allocation](images/bit-fields_5.png)

Union Allocation

![Unnamed Bit Fields](images/bit-fields_6.png)

Unnamed Bit Fields
