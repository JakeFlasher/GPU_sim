---
title: "Object Model & Lifetimes"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#object-model-lifetimes"
---

### [Object Model & Lifetimes](https://docs.nvidia.com/cuda/cutile-python#object-model-lifetimes)[](https://docs.nvidia.com/cuda/cutile-python/#object-model-lifetimes "Permalink to this headline")

All objects created within [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code) are immutable.
Any operation that would conceptually modify an object instead creates and returns a new object.
Attributes cannot be added to objects dynamically.

Global [arrays](https://docs.nvidia.com/cuda/cutile-python/data.html#data-global-arrays) are views that can read and write global device memory, but the views themselves
are also immutable.

The caller of a [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) must ensure that:

- No [arrays](https://docs.nvidia.com/cuda/cutile-python/data.html#data-global-arrays) passed to the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) alias one another.
- All [arrays](https://docs.nvidia.com/cuda/cutile-python/data.html#data-global-arrays) remain valid until the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) completes execution.
