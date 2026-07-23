---
title: "Constant Expressions & Objects"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#constant-expressions-objects"
---

### [Constant Expressions & Objects](https://docs.nvidia.com/cuda/cutile-python#constant-expressions-objects)[](https://docs.nvidia.com/cuda/cutile-python/#constant-expressions-objects "Permalink to this headline")

Some facilities require parameters whose values are known at compile time.
_Constant expressions_ produce _constant objects_ suitable for such parameters.
Constant expressions are:

- A literal object.
- Integer arithmetic expressions where all the operands are literal objects.
- A local object or parameter that is assigned from a literal object or constant expression.
- A global object that is defined at the time of compilation or launch.

By default, numeric constants are _loosely typed_: integer constants have
infinite precision and floating-point constants are stored in IEEE 754 double
precision, until used in a context that requires a specific-width type.

A _strictly typed_ constant is created by calling a dtype constructor,
e.g. `ct.int16(5)`. Combining a strictly typed constant with a loosely typed
constant yields a strictly typed constant:
`ct.int16(5) + 2` produces a strictly typed `int16` constant 7.

Combining two strictly typed constants also produces a strictly typed constant,
with the regular [type promotion](https://docs.nvidia.com/cuda/cutile-python/data.html#data-arithmetic-promotion) rules applied.
For example, `ct.int16(5) + ct.int32(7)` produces a strictly typed `int32`
constant 12.
