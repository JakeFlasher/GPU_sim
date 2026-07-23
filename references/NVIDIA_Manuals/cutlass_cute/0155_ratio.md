---
title: "Ratio"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#ratio"
---

### [Ratio](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#ratio)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#ratio "Permalink to this headline")

`Ratio` represents a rational number as a ratio of two integers. It is used in CuTe to represent exact fractional values that arise in tensor layout operations, particularly in composition operations where divisibility conditions may not be satisfied.

**Constructor:**

```python
ratio = cute.Ratio(numerator, denominator)
```

**param numerator:**
: The numerator of the ratio

**type numerator:**
: int

**param denominator:**
: The denominator of the ratio

**type denominator:**
: int

**raises TypeError:**
: If numerator or denominator are not integers

**Methods:**

- `is_integral()` - Returns `True` if the ratio represents an integer value (numerator divisible by denominator)
- `reduced()` - Returns a new Ratio with numerator and denominator reduced to lowest terms
- `to(dtype)` - Converts the ratio to another type (Ratio, float, or int)

**Arithmetic Operations:**

```python
# Multiplication with another ratio
ratio1 = cute.Ratio(1, 2)
ratio2 = cute.Ratio(3, 4)
result = ratio1 * ratio2  # Returns Ratio(3, 8)

# Multiplication with integer
ratio = cute.Ratio(2, 3)
result = ratio * 5  # Returns Ratio(10, 3)
result = 5 * ratio  # Returns Ratio(10, 3)
```

**Type Conversion:**

```python
ratio = cute.Ratio(3, 2)

# Convert to float
float_val = ratio.to(float)  # Returns 1.5

# Convert to int (floor division)
int_val = ratio.to(int)  # Returns 1
```
