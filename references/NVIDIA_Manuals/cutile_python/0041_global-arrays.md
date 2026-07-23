---
title: "Global Arrays"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#global-arrays"
---

## [Global Arrays](https://docs.nvidia.com/cuda/cutile-python#global-arrays)[](https://docs.nvidia.com/cuda/cutile-python/#global-arrays "Permalink to this headline")

A _global array_ (or _array_) is a container of elements of a specific [dtype](https://docs.nvidia.com/cuda/cutile-python/#data-data-types)
arranged in a logical multidimensional space.

An array’s _shape_ is a tuple of integers, each denoting the length of the
corresponding dimension. The length of the shape tuple equals the array’s number
of dimensions, and the product of its values equals the total number of logical
elements in the array.

Arrays are stored in global memory using a _strided memory layout_: in addition to a shape,
each array has an equally sized tuple of _strides_ that maps logical indices to physical memory
locations. For example, for a 3-dimensional *float32* array with strides *(s1, s2, s3)*, the
memory address of the element at index *(i1, i2, i3)* is:

```default
base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),
```

where `base_addr` is the base address of the array and `4` is the byte size of a `float32`
element.

New arrays can only be allocated by the host and passed to the tile kernel as arguments.
[Tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code) can only create new views of existing arrays, for example via
[`Array.slice()`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.slice "cuda.tile.Array.slice"). As in Python, assigning an array to another variable does not copy
the underlying data but creates another reference to the same array.

Any object that implements the [DLPack](https://dmlc.github.io/dlpack/latest/) interface or the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
can be passed as a kernel argument — for example, [CuPy](https://docs.cupy.dev/en/stable/) arrays and [PyTorch](https://pytorch.org/docs/stable/) tensors.

If two or more array arguments are passed to the kernel, their memory must not overlap.
Otherwise, the behavior is undefined.

An array’s shape can be queried with the [`Array.shape`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.shape "cuda.tile.Array.shape") attribute, which
returns a tuple of *int32* scalars. These are non-constant, runtime values.
Using *int32* improves performance at the cost of capping the maximum representable
shape at 2,147,483,647 elements. This limitation will be lifted in the future.

> **See also**
>
> [cuda.tile.Array class documentation](https://docs.nvidia.com/cuda/cutile-python/data/array.html#data-array-cuda-tile-array)
