---
title: "Data Model"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#data-model"
---

# [Data Model](https://docs.nvidia.com/cuda/cutile-python#data-model)[](https://docs.nvidia.com/cuda/cutile-python/#data-model "Permalink to this headline")

cuTile is an array-based programming model.
The fundamental data structure is the multidimensional array whose elements share a single
homogeneous type. cuTile Python exposes only arrays, not pointers.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked for safety and correctness.
- Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
- Python programmers are already familiar with array-based frameworks such as NumPy.
- Pointers are not a natural fit for Python.

Within [tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code), only the types described in this section are supported.
