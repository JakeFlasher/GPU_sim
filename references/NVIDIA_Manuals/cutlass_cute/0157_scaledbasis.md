---
title: "ScaledBasis"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#scaledbasis"
---

### [ScaledBasis](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#scaledbasis)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#scaledbasis "Permalink to this headline")

`ScaledBasis` represents a scaled basis element in CuTe’s layout algebra. It consists of a scale value and a mode that identifies which basis element in the layout algebra is being referenced. ScaledBasis elements are fundamental to CuTe’s coordinate system representation.

**Constructor:**

```python
sb = cute.ScaledBasis(value, mode)
```

**param value:**
: The scale value

**type value:**
: Union[int, Integer, Ratio, ir.Value]

**param mode:**
: The mode identifying the basis element

**type mode:**
: Union[int, List[int]]

**raises TypeError:**
: If mode is not an integer or list of integers

**Examples:**

```python
# Create a scaled basis with integer scale and mode
sb1 = cute.ScaledBasis(2, 0)  # 2 * E(0)

# Create a scaled basis with a Ratio scale
sb2 = cute.ScaledBasis(cute.Ratio(1, 2), 1)  # (1/2) * E(1)

# Create a scaled basis with a list of modes
sb3 = cute.ScaledBasis(4, [0, 1])  # 4 * E([0, 1])

# Scaled basis elements are commonly used in layout strides
layout = cute.make_layout((4, 8), stride=(cute.ScaledBasis(2, 0), cute.ScaledBasis(1, 1)))

# This creates a layout with strides (2@0, 1@1) representing
# a coordinate system where each dimension has its own basis

# Example: Mapping coordinates to indices using the layout
coord = (2, 3)
idx = cute.crd2idx(coord, layout)  # Maps (2, 3) to (4, 3)
```

**Properties:**

- `value` - Get the scale value
- `mode` - Get the mode as a list of integers
- `is_static()` - Returns `True` if the value is statically known

**Methods:**

- `to(dtype)` - Convert to another type (ScaledBasis or internal _ScaledBasis)

**Operations:**

```python
# Right multiplication by a scale factor
sb = cute.ScaledBasis(2, 0)
result = 3 * sb  # Creates ScaledBasis(6, 0)
```

**Utility Function:**

```python
# Create a basis element with unit scale
basis = cute.E(mode)  # Equivalent to ScaledBasis(1, mode)
```
