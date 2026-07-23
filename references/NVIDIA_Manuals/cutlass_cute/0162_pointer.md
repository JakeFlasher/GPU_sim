---
title: "Pointer"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#pointer"
---

### [Pointer](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#pointer)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#pointer "Permalink to this headline")

`Pointer` represents a memory address with specific properties. Pointers are a fundamental type of iterator/engine that support random-access operations. They can be offset by elements of a layout’s codomain and dereferenced to produce values.

**Properties:**

- `dtype` - The type of value this pointer points to
- `type` - The MLIR type of the pointer
- `memspace` - The memory space where the pointer data resides (e.g., `gmem`, `smem`, `rmem`)
- `alignment` - The alignment of the pointer in bytes
- `max_alignment` - The maximum alignment of the pointer in bytes

**Operations:**

```python
# Pointer arithmetic
ptr2 = ptr + offset     # Offset pointer forward
ptr3 = offset + ptr     # Offset pointer forward (commutative)
ptr4 = ptr - offset     # Offset pointer backward

# Convert pointer to integer
int_addr = ptr.toint()

# Align pointer to specified byte boundary
aligned_ptr = ptr.align(16)  # Align to 16-byte boundary
```

**Tensor Composition:**

When composed with a layout, a pointer forms a tensor: `T = E ∘ L`, where `E` is the pointer (engine) and `L` is the layout. The tensor evaluates the layout by mapping a coordinate `c` to the codomain, offsets the pointer accordingly, and dereferences the result:

```text
T(c) = (E ∘ L)(c) = *(E + L(c))
```

**Methods:**

- `llvm_ptr` - Get the LLVM pointer representation (low-level use only)
- `align(min_align)` - Align pointer to specified byte alignment (must be power of 2)
- `toint()` - Convert pointer to integer address (Int64 for gmem/generic, Int32 otherwise)

**Examples:**

```python
# Create a pointer from a tensor's data
ptr = tensor.data()

# Offset the pointer
offset_ptr = ptr + 16

# Check pointer properties
print(f"Memory space: {ptr.memspace}")
print(f"Alignment: {ptr.alignment}")
print(f"Data type: {ptr.dtype}")
```
