---
title: "Best Practices"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#best-practices"
---

## [Best Practices](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#best-practices)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#best-practices "Permalink to this headline")

**Choosing Between Static and Dynamic:**

- Use static values (Python `int`) when dimensions are known at compile time for maximum optimization
- Use dynamic values (`IntValue`) when dimensions must be determined at runtime
- Refer to [Static vs Dynamic layouts](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html) for detailed guidance on static vs dynamic layouts

**Memory Alignment:**

- Always specify alignment requirements for shared memory structures to avoid bank conflicts
- Use `struct.Align` to enforce alignment constraints
- Check `max_alignment` properties to verify pointer and layout alignment

**Layout Operations:**

- Prefer built-in layout operations (`make_layout`, `composition`, etc.) over manual construction
- Use `ScaledBasis` for explicit control over stride modes in multi-modal layouts
- Leverage `ComposedLayout` for complex transformations like swizzling

**Type Safety:**

- Use type annotations in `@jit` and `@kernel` functions
- Let the DSL infer types when possible for cleaner code
- Check `dtype` and `memspace` properties when working with pointers
