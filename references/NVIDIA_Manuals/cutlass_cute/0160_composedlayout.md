---
title: "ComposedLayout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#composedlayout"
---

### [ComposedLayout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#composedlayout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#composedlayout "Permalink to this headline")

`ComposedLayout` represents a composition of layouts and transformations. It is a generalization of normal layouts that can support arbitrary function mappings from coordinate to coordinate as an inner layout.

**Structure:**

A ComposedLayout consists of three components:

- **inner**: The inner transformation (Swizzle or Layout)
- **offset**: An offset applied to coordinates
- **outer**: The outer layout

**Properties:**

- `inner` - Returns the inner transformation (Union[Swizzle, Layout])
- `offset` - Returns the offset as an IntTuple
- `outer` - Returns the outer layout
- `shape` - Returns the shape of the composed layout
- `max_alignment` - Returns the maximum alignment
- `is_normal` - Returns `True` if this is a normal layout (not a general composition)

**Examples:**

```python
# ComposedLayouts are typically created through composition operations
# For example, composing a layout with a swizzle
layout = cute.make_layout((8, 8))
swizzle = cute.make_swizzle(...)
composed = cute.composition(swizzle, layout)

# Accessing components
inner = composed.inner      # Returns the swizzle
outer = composed.outer      # Returns the layout
offset = composed.offset    # Returns the offset
```

**String Representation:**

```python
print(composed)  # Prints "inner o offset o outer" format
```
