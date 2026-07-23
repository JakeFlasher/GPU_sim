---
title: "Constant Embedding"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#constant-embedding"
---

### [Constant Embedding](https://docs.nvidia.com/cuda/cutile-python#constant-embedding)[](https://docs.nvidia.com/cuda/cutile-python/#constant-embedding "Permalink to this headline")

If a [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) parameter is _constant embedded_, then:

- Every use of the parameter behaves as if replaced by its literal value.
- A distinct [machine representation](https://docs.nvidia.com/cuda/cutile-python/interoperability.html#interoperability-machine-representation) of the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) is generated for each unique value of the parameter. Note: The [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) is compiled once per unique value, even if JIT caching is enabled.
