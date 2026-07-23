---
title: "Type Hierarchies and Relationships"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#type-hierarchies-and-relationships"
---

## [Type Hierarchies and Relationships](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#type-hierarchies-and-relationships)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#type-hierarchies-and-relationships "Permalink to this headline")

**Type Protocol Support:**

Many CuTe types implement standard Python protocols for integration:

- `__str__()` - String representation for debugging
- `__eq__()` / `__ne__()` - Equality comparison
- `__getitem__()` - Indexing operations
- `__add__()` / `__sub__()` / `__mul__()` / `__floordiv__()` / `__mod__()` - Arithmetic

**MLIR Integration:**

Internal types like `IntValue`, `Layout`, `Pointer`, and `ComposedLayout` are registered as MLIR value casters, enabling seamless integration with the underlying compiler infrastructure.
