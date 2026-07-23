---
title: "Calling Conventions"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/compilation.html#calling-conventions"
---

## [Calling Conventions](https://docs.nvidia.com/cuda/cutile-python#calling-conventions)[](https://docs.nvidia.com/cuda/cutile-python/#calling-conventions "Permalink to this headline")

A calling convention defines three aspects of the binary interface provided by an exported kernel:

- The binary format and the order of kernel arguments, e.g. as passed to the `cuLaunchKernel()`
CUDA Driver API function.
- The set of supported parameter constraints.
- The name mangling algorithm used to automatically derive a symbol name from the kernel’s
function name and a kernel signature.

The only currently implemented calling convention is `cutile_python_v1`.
According to this convention, the binary kernel arguments are passed in the same order
as the kernel parameters are declared in the Python kernel function, except that parameters
annotated with [`ct.Constant`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.Constant "cuda.tile.Constant") are omitted. The following table
lists the supported parameter constraints, as well as the corresponding binary format
of the kernel arguments:

| Constraint Class | Binary Format of Arguments |
| --- | --- |
| [`ScalarConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ScalarConstraint "cuda.tile.compilation.ScalarConstraint") | Passed as a single argument of the corresponding type. For example, if the constraint’s *dtype* is [`ct.int32`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.int32 "cuda.tile.int32"), the corresponding C type is `int32_t`; [`ct.float64`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.float64 "cuda.tile.float64") corresponds to C’s `double` and so on. |
| [`ArrayConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ArrayConstraint "cuda.tile.compilation.ArrayConstraint") | Passed as *1 + 2n* arguments, where *n* is the number of dimensions (*ndim*) of the array. The first argument is the device pointer to the base of the array’s data. It is followed by *n* arguments representing the shape of the array. Finally, the last *n* arguments represent the strides of the array. The type of shape and stride arguments is determined by the `index_dtype` field of the constraint:   \| Constraint’s `index_dtype` \| C Type of Shape & Strides \| \| --- \| --- \| \| [`ct.int32`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.int32 "cuda.tile.int32") \| `int32_t` \| \| [`ct.uint32`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.uint32 "cuda.tile.uint32") \| `uint32_t` \| \| [`ct.int64`](https://docs.nvidia.com/cuda/cutile-python/data.html#cuda.tile.int64 "cuda.tile.int64") \| `int64_t` \| |
| [`ListConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ListConstraint "cuda.tile.compilation.ListConstraint") with an [`ArrayConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ArrayConstraint "cuda.tile.compilation.ArrayConstraint") element | Passed as two arguments: a device pointer to the base of the list data and an `int32_t` denoting the length of the list. The base pointer must point to an 8-byte aligned contiguous buffer in the global GPU memory, consisting of *(1 + 2n) * L* 64-bit words, where *L* is the length of the list and *n* is the *ndim* of the element array constraint. Each element array of the list is represented by *(1 + 2n)* words in this buffer. The first word stores a device pointer to the base of array; the next *n* signed integers store the shape of the array; the final *n* signed integers store the strides of the array. Even though 64-bit integers are always used for the shape and the strides, they are truncated to the `index_dtype` of the element constraint. |
| [`ConstantConstraint`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.ConstantConstraint "cuda.tile.compilation.ConstantConstraint") | Omitted from the launch arguments. |

Calling conventions are represented by the [`CallingConvention`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.CallingConvention "cuda.tile.compilation.CallingConvention") class:

```
_`class`_`cuda.tile.compilation.``CallingConvention`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.CallingConvention "Link to this definition")
```

```
_`static`_`cutile_python_v1`()[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.CallingConvention.cutile_python_v1 "Link to this definition")
```

Returns the `cutile_python_v1` calling convention.
