---
title: "How do I print CuTe objects on host or device?"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html#how-do-i-print-cute-objects-on-host-or-device"
---

### [How do I print CuTe objects on host or device?](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#how-do-i-print-cute-objects-on-host-or-device)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#how-do-i-print-cute-objects-on-host-or-device "Permalink to this headline")

The `cute::print` function has overloads for almost all CuTe types, including Pointers, Integers, Strides, Shapes, Layouts, and Tensors.  When in doubt, try calling `print` on it.

CuTe’s print functions work on either host or device.
Note that on device, printing is expensive.
Even just leaving print code in place on device,
even if it is never called
(e.g., printing in an `if` branch that is not taken at run time),
may generate slower code.
Thus, be sure to remove code that prints on device after debugging.

You might also only want to print on thread 0 of each threadblock, or threadblock 0 of the grid.  The `thread0()` function returns true only for global thread 0 of the kernel, that is, for thread 0 of threadblock 0.  A common idiom for printing CuTe objects to print only on global thread 0.

```c++
if (thread0()) {
  print(some_cute_object);
}
```

Some algorithms depend on some thread or threadblock,
so you may need to print on threads or threadblocks other than zero.
The header file
[`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp),
among other utilities,
includes the function `bool thread(int tid, int bid)`
that returns `true` if running on thread `tid` and threadblock `bid`.

#### [Other output formats](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#other-output-formats)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#other-output-formats "Permalink to this headline")

Some CuTe types have special printing functions that use a different output format.

The `cute::print_layout` function will display any rank-2 layout in a plain text table. This is excellent for visualizing the map from coordinates to indices.

The `cute::print_tensor` function will display any rank-1, rank-2, rank-3, or rank-4 tensor in a plain text multidimensional table. The values of the tensor are printed so you can verify the tile of data is what you expect after a copy, for example.

The `cute::print_latex` function will print LaTeX commands that you can use to build a nicely formatted and colored tables via `pdflatex`. This work for `Layout`, `TiledCopy`, and `TiledMMA`, which can be very useful to get a sense of layout patterns and partitioning patterns within CuTe.
