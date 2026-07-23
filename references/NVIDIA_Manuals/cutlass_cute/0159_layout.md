---
title: "Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#layout"
---

### [Layout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#layout "Permalink to this headline")

`Layout` is CuTe’s core abstraction for representing tensor layouts. A Layout maps from a logical coordinate space to an index space, defined by a pair of (Shape, Stride). Layouts present a common interface to multidimensional array access that abstracts away the details of how array elements are organized in memory.

**Key Concepts:**

- **Shape**: Defines the abstract dimensions of the Layout
- **Stride**: Defines how coordinates within the Shape map to linear indices
- **Hierarchical Structure**: CuTe layouts are inherently hierarchical, constructed from smaller nested layouts

**Properties:**

- `shape` - An IntTuple representing the dimensions of the layout
- `stride` - An IntTuple representing the strides of the layout
- `max_alignment` - The maximum alignment of the layout in bytes

**Examples:**

```python
# Creating a layout with shape (4,8) and default stride (column major)
layout = cute.make_layout((4, 8))

# Creating a layout with explicit shape and stride (row major)
layout = cute.make_layout((4, 8), stride=(8, 1))

# Accessing layout properties
shape = layout.shape      # Returns (4, 8)
stride = layout.stride    # Returns (8, 1)

# Mapping a coordinate to an index: (2, 3) -> 2 * 8 + 3 * 1 = 19
idx = cute.crd2idx((2, 3), layout)
```

**Layout Operations:**

Layouts support a rich algebra of operations:

- **Concatenation**: Combining layouts along dimensions
- **Coalescence**: Merging adjacent modes
- **Composition**: Composing layouts with functions or other layouts
- **Complement**: Computing the complement space
- **Inversion**: Inverting the layout mapping

**String Representation:**

```python
layout = cute.make_layout((4, 8), stride=(1, 4))
print(layout)  # Prints "shape:stride" format, e.g., "(4,8):(1,4)"
```
