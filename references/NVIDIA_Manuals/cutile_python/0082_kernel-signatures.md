---
title: "Kernel Signatures"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/compilation.html#kernel-signatures"
---

## [Kernel Signatures](https://docs.nvidia.com/cuda/cutile-python#kernel-signatures)[](https://docs.nvidia.com/cuda/cutile-python/#kernel-signatures "Permalink to this headline")

There are two ways to construct a [`KernelSignature`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation.KernelSignature") object for use with
[`export_kernel()`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.export_kernel "cuda.tile.compilation.export_kernel"). The recommended way is to do it explicitly, by instantiating
a [`KernelSignature`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation.KernelSignature") object and providing a list of manually constructed
[`ParameterConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ParameterConstraint "cuda.tile.compilation.ParameterConstraint") objects.

Alternatively, one may use [`KernelSignature.from_kernel_args()`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature.from_kernel_args "cuda.tile.compilation.KernelSignature.from_kernel_args") to obtain a signature
that would be used if the kernel was compiled just-in-time for the given example arguments.
While convenient, this approach may create undesired assumptions on kernel parameters. For example,
if the base address of an example array argument happens to be divisible by 16, an assumption may
be made that it will always be so. Launching the exported kernel with an array that doesn’t
satisfy this assumption would then result in undefined behavior. It is therefore recommended to limit
the use of this approach to testing or prototyping.

```
_`class`_`cuda.tile.compilation.``KernelSignature`(_`parameters`_, _`calling_convention`_, _`symbol``=``None`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "Link to this definition")
```

Signature of a compiled kernel.

**Parameters:**
: - **parameters** (_Sequence__[__ParameterConstraint__|__bool__|__int__|__float__]_) – For each parameter of the kernel’s Python function, a corresponding
[`ParameterConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ParameterConstraint "cuda.tile.compilation.ParameterConstraint") instance. If a parameter is marked with
[`ct.Constant`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.Constant "cuda.tile.Constant"), the corresponding constraint must be
a [`ConstantConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ConstantConstraint "cuda.tile.compilation.ConstantConstraint") or a `bool`, `int` or `float` value
that specifies the compile-time constant.
Otherwise, it must be either a [`ScalarConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ScalarConstraint "cuda.tile.compilation.ScalarConstraint"),
[`ArrayConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ArrayConstraint "cuda.tile.compilation.ArrayConstraint") or [`ListConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ListConstraint "cuda.tile.compilation.ListConstraint").
Passing a `bool`, `int` or `float` value as a constraint is convenience notation
that is equivalent to passing an instance of [`ConstantConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ConstantConstraint "cuda.tile.compilation.ConstantConstraint")
that wraps said value.
- **calling_convention** ([_CallingConvention_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.CallingConvention "cuda.tile.compilation.CallingConvention")) – [Calling convention](https://docs.nvidia.com/cuda/cutile-python/#compilation-callconv) to use.
- **symbol** (_str__|__None_) – Symbol name to use for the exported kernel. Set to *None* to automatically
generate it from the Python function name and this signature, using a name mangling
algorithm defined by the selected calling convention.

```
`with_mangled_symbol`(_`function_name`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature.with_mangled_symbol "Link to this definition")
```

Returns a copy of *self* with the *symbol* attribute replaced with a mangled name.

**Parameters:**
: **function_name** (_str_) – Function name to use as the base of the mangled symbol.

**Returns:**
: KernelSignature

**Return type:**
: [_KernelSignature_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation._signature.KernelSignature")

```
`with_symbol`(_`symbol`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature.with_symbol "Link to this definition")
```

Returns a copy of *self* with the *symbol* attribute replaced with the given value.

**Parameters:**
: **symbol** (_str__|__None_) – The new symbol name.

**Returns:**
: KernelSignature

**Return type:**
: [_KernelSignature_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation._signature.KernelSignature")

```
_`static`_`from_kernel_args`(
```

Returns the signature that would be used if the kernel was compiled just-in-time
for the given arguments.o

> **Warning**
>
> It is recommended to limit the use of this function to testing or prototyping.
> Deriving a kernel signature from example arguments may create unexpected assumptions
> on kernel parameters.
>
> For example, if the base address of an example array argument happens to be
> divisible by 16, an assumption may be made that it will always be so.
> Launching the exported kernel with an array that doesn’t satisfy this assumption
> would result in undefined behavior.

**Parameters:**
: - **kernel** ([_cuda.tile.kernel_](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel "cuda.tile.kernel")) – A kernel function decorated with [`@ct.kernel`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel "cuda.tile.kernel").
- **kernel_args** (_Sequence__[__Any__]_) – Tuple of kernel arguments, as if it were be passed to
[`ct.launch()`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.launch "cuda.tile.launch").
- **calling_convention** ([_CallingConvention_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.CallingConvention "cuda.tile.compilation.CallingConvention")) – [Calling convention](https://docs.nvidia.com/cuda/cutile-python/#compilation-callconv) to use.
- **symbol** (_str__|__None_) – Specifies the *symbol* attribute of the returned signature.
If set to *None*, the returned symbol will be automatically filled
using a name mangling algorithm defined by the selected [calling convention](https://docs.nvidia.com/cuda/cutile-python/#compilation-callconv).

**Returns:**
: KernelSignature

**Return type:**
: [_KernelSignature_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation._signature.KernelSignature")

The [`ParameterConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ParameterConstraint "cuda.tile.compilation.ParameterConstraint") type alias is used as a type hint for a kernel parameter
constraint:

```
`cuda.tile.compilation.``ParameterConstraint`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ParameterConstraint "Link to this definition")
```

alias of [`ScalarConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ScalarConstraint "cuda.tile.compilation._signature.ScalarConstraint") | [`ArrayConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ArrayConstraint "cuda.tile.compilation._signature.ArrayConstraint") | [`ListConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ListConstraint "cuda.tile.compilation._signature.ListConstraint") | [`ConstantConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ConstantConstraint "cuda.tile.compilation._signature.ConstantConstraint")

```
_`class`_`cuda.tile.compilation.``ScalarConstraint`(_`dtype`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ScalarConstraint "Link to this definition")
```

Describes a scalar kernel parameter and associated compile-time assumptions.

**Parameters:**
: **dtype** ([_DType_](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.DType "cuda.tile._datatype.DType")) – Data type of the scalar.

```
_`class`_`cuda.tile.compilation.``ArrayConstraint`(
```

Describes an array kernel parameter and associated compile-time assumptions.

**Parameters:**
: - **dtype** ([_DType_](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.DType "cuda.tile.DType")) – Data type of the array.
- **ndim** (_int_) – Number of dimensions of the array, also known as rank.
- **index_dtype** ([_DType_](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.DType "cuda.tile.DType")) – Data type used to represent array’s shape, strides and indices.
Currently, only [`ct.int32`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.int32 "cuda.tile.int32") is supported.
- **stride_lower_bound_incl** (_Sequence__[__int__|__None__]__|__int__|__None_) – For each dimension of the array, an optional inclusive lower bound for its stride.
If all dimensions have the same lower bound, a single number can be passed
instead of a sequence. For example, passing *0* specifies that all strides
are non-negative.
- **alias_groups** (_Sequence__[__str__]_) – When set to an empty sequence, specifies that this array may not alias
any other parameter. Otherwise, it must be a sequence of arbitrary strings,
referred to as “alias groups”. Two parameters are allowed to alias
each other if and only if they have an alias group in common.
- **may_alias_internally** (_bool_) – Indicates whether two distinct in-bounds indices are allowed
to point to the same memory location. For example, this can happen if the
array has a zero stride. For most arrays produced by major tensor libraries,
this can be assumed to be false. Setting this to True may disable certain
optimizations of loads and stores to/from this array.
- **stride_constant** (_Sequence__[__int__|__None__]__|__None_) – For each dimension of the array, an optional constant value of its stride.
For example, if the array is known to have a C-contiguous layout, the stride of
the last dimension can be set to 1, which may enable certain optimizations of loads
and stores from/to this array. Can be set to *None* if none of the dimensions
have known strides (this is the default).
- **stride_divisible_by** (_Sequence__[__int__]__|__int_) – For each dimension of the array, a factor by which its stride is assumed
to be divisible. The value is given in array elements, not bytes.
For example, a value of *8* for a *float16* array indicates divisibility by
16 bytes, since each element of the array is 2 bytes wide. Value of 1
indicates that no assumption is made regarding the stride divisibility
(this is the default).
- **shape_divisible_by** (_Sequence__[__int__]__|__int_) – For each dimension of the array, a factor by which its length is assumed
to be divisible. The value is given in array elements, not bytes.
For example, a value of *8* for a *float16* array indicates divisibility by
16 bytes, since each element of the array is 2 bytes wide. Value of 1
indicates that no assumption is made regarding the shape divisibility
(this is the default).
- **base_addr_divisible_by** (_int_) – Factor by which the array’s base address is assumed
to be divisible. Value of 1 indicates that no assumption is made regarding
the base address divisibility (this is the default).

```
_`class`_`cuda.tile.compilation.``ListConstraint`(_`element`_, _`*`_, _`alias_groups`_, _`elements_may_alias`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ListConstraint "Link to this definition")
```

Describes a list kernel parameter and associated compile-time assumptions.

**Parameters:**
: - **element** ([_ArrayConstraint_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ArrayConstraint "cuda.tile.compilation.ArrayConstraint")) – Describes the element of this list. Currently, this must be an *ArrayConstraint*,
since only lists of arrays are supported as kernel arguments.
- **alias_groups** (_Sequence__[__str__]_) – Describes which other parameters the storage of this list is allowed to alias.
Note that this is different from `element.alias_groups`, which sets aliasing
assumptions on the list elements.
When set to an empty sequence, specifies that this list may not alias
any other parameter. Otherwise, it must be a sequence of arbitrary strings,
referred to as “alias groups”. Two parameters are allowed to alias
each other if and only if they have an alias group in common.
- **elements_may_alias** (_bool_) – Specifies whether two distinct elements of this list are allowed to alias each other.

```
_`class`_`cuda.tile.compilation.``ConstantConstraint`(_`value`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ConstantConstraint "Link to this definition")
```

Specifies the constant value of a kernel parameter
marked with [`ct.Constant`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.Constant "cuda.tile.Constant").

**Parameters:**
: **value** (_bool__|__int__|__float_) – The value of the compile-time constant.
