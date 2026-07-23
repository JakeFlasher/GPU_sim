---
title: "IntValue"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#intvalue"
---

### [IntValue](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#intvalue)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#intvalue "Permalink to this headline")

`IntValue` is an internal representation of constrained integer types with divisibility information. It serves as a proxy for constrained integer types in the CuTe IR, automatically tracking divisibility constraints that are crucial for layout operations.

**Key Features:**

- Inherits from `ArithValue` with extensions for divisibility tracking
- Automatically emits `cute.get_scalars` operations in the IR
- Supports arithmetic operations that propagate divisibility information
- Used internally for type-safe integer operations in layout algebra

**API Methods:**

- `get_typed_value()` - Returns the value as an IntTupleType
- `get_divisibility()` - Returns the divisibility constraint of the value
- `divisibility` - Property that returns the divisibility constraint

**Supported Operations:**

The `IntValue` type supports standard arithmetic operations with divisibility tracking:

```python
# Addition, subtraction, multiplication, division, and modulo
result = int_val1 + int_val2
result = int_val1 - int_val2
result = int_val1 * int_val2
result = int_val1 // int_val2
result = int_val1 % int_val2
```

**String Representation:**

```python
# IntValue with divisibility 1
str(int_val)  # Returns "?"

# IntValue with divisibility 4
str(int_val)  # Returns "?{div=4}"
```
