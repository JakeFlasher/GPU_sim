---
title: "struct"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#struct"
---

### [struct](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#struct)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#struct "Permalink to this headline")

The `struct` decorator abstracts C structures in Python DSL. It allows you to define structured data types with precise control over layout, alignment, and nesting.

**Supported Elements:**

- Base DSL scalar int/float elements
- Arrays (MemRange)
- Nested structures
- Aligned elements

**Basic Usage:**

```python
# Define a simple struct
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32

# Define a struct with arrays and nested structures
@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16
```

**Alignment Control:**

```python
# Define a struct with explicit alignment
@cute.struct
class StorageB:
    a: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_a], 1024
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_b], 1024
    ]
    x: cute.struct.Align[cutlass.Int32, 16]
    compA: cute.struct.Align[complex, 16]
```

**Static Queries:**

```python
# Get size and alignment at compile time
size = StorageB.__sizeof__()
align = StorageB.__alignof__()
```

**Allocation and Access:**

```python
# Allocate and reference elements
storage = allocator.allocate(StorageB)

# Access struct members
storage.a[0] = ...
storage.x = ...
... = storage.compA.real.ptr
... = storage.x.ptr.load()
```

**Methods:**

- `__sizeof__()` - Returns the size of the struct in bytes
- `__alignof__()` - Returns the alignment of the struct in bytes
- `size_in_bytes()` - Returns the size of the struct in bytes

#### [struct.MemRange](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#struct-memrange)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#struct-memrange "Permalink to this headline")

`MemRange` defines a contiguous range of memory with a specific element type and size.

**Syntax:**

```python
cute.struct.MemRange[dtype, size]
```

**param dtype:**
: The data type (must be a DSL scalar type)

**type dtype:**
: Type[Numeric]

**param size:**
: The number of elements in the range

**type size:**
: int

**Properties:**

- `size` - Number of elements in the range
- `elem_width` - Width of each element in bits
- `size_in_bytes` - Total size in bytes

**Methods:**

- `data_ptr()` - Returns a pointer to the start of the memory range
- `get_tensor(layout, swizzle=None, dtype=None)` - Creates a tensor from the memory range
- `__getitem__(index)` - Returns the element at the specified index

**Examples:**

```python
@cute.struct
class Buffer:
    data : cute.struct.MemRange[cutlass.Float32, 128]

# Allocate buffer
buf = allocator.allocate(Buffer)

# Get pointer to data
ptr = buf.data.data_ptr()

# Access individual elements
element = buf.data[5]

# Create tensor from memory range
layout = cute.make_layout((8, 16))
tensor = buf.data.get_tensor(layout)
```

#### [struct.Align](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#struct-align)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#struct-align "Permalink to this headline")

`Align` specifies explicit alignment requirements for struct members.

**Syntax:**

```python
cute.struct.Align[dtype, alignment]
```

**param dtype:**
: The type to align (scalar, MemRange, or struct)

**type dtype:**
: Type

**param alignment:**
: The alignment in bytes (must be > 0)

**type alignment:**
: int

**Properties:**

- `dtype` - The data type being aligned
- `align` - The alignment value

**Examples:**

```python
@cute.struct
class AlignedStorage:
    # Align scalar to 16 bytes
    counter: cute.struct.Align[cutlass.Int32, 16]

    # Align array to 1024 bytes
    buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 256], 1024
    ]
```
