---
title: "union"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#union"
---

### [union](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#union)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#union "Permalink to this headline")

The `union` decorator abstracts C unions in Python DSL. Similar to `struct`, but all members start at offset 0, and the size is the maximum size of all members.

**Layout Characteristics:**

- All objects start at offset 0
- Alignment is the maximum alignment of all objects
- Size is the maximum size of all objects

**Usage:**

```python
# Define a union with scalar elements
@cute.union
class value_union:
    as_int : cutlass.Int32
    as_float : cutlass.Float32

# Allocate union
val = allocator.allocate(value_union)

# Access different interpretations of same memory
val.as_int = 42
float_val = val.as_float.ptr.load()  # Interpret same bits as float
```

**Methods:**

Same as `struct`:

- `__sizeof__()` - Returns the size of the union in bytes
- `__alignof__()` - Returns the alignment of the union in bytes
