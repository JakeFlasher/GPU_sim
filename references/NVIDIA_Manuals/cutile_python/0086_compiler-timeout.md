---
title: "Compiler Timeout"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/debugging.html#compiler-timeout"
---

## [Compiler Timeout](https://docs.nvidia.com/cuda/cutile-python#compiler-timeout)[](https://docs.nvidia.com/cuda/cutile-python/#compiler-timeout "Permalink to this headline")

```
`cuda.tile.``compiler_timeout`(_`timeout_sec`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compiler_timeout "Link to this definition")
```

Context manager that temporarily sets the compiler timeout.

> **Note**
>
> This function is not thread-safe. It modifies global state
> shared by all threads.

Example:

```default
with ct.compiler_timeout(10):
    ct.launch(stream, grid, kernel, args)
```
