---
title: "Tile Kernels"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-tile-kernels"
---

## [Tile Kernels](https://docs.nvidia.com/cuda/cutile-python#execution-tile-kernels)[](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels "Permalink to this headline")

```
_`class`_`cuda.tile.``kernel`(_`function``=``None`_, _`/`_, _`**``kwargs`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.kernel "Link to this definition")
```

A _tile kernel_ is a function executed by each [block](https://docs.nvidia.com/cuda/cutile-python/#block) in a [grid](https://docs.nvidia.com/cuda/cutile-python/#grid).

Functions with this decorator are [kernels](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels).

[Kernels](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) are the entry points of [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code).
Their [execution space](https://docs.nvidia.com/cuda/cutile-python/#execution-execution-spaces) shall be only [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code); they cannot be called from [host code](https://docs.nvidia.com/cuda/cutile-python/#host-code).

Kernels cannot be called directly. Instead, use [`launch()`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.launch "cuda.tile.launch") to
queue a kernel for execution over a grid.

The types usable as parameters to a [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) are described in the [data model](https://docs.nvidia.com/cuda/cutile-python/data.html#data-data-model).

**Parameters:**
: - **num_ctas** – Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
Default: None (auto).
- **occupancy** – Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
- **opt_level** – Optimization level [0, 3], default 3.

Target-specific values for the compiler options above can be provided
using a [`ByTarget`](https://docs.nvidia.com/cuda/cutile-python/performance.html#cuda.tile.ByTarget "cuda.tile.ByTarget") object.

```
`replace_hints`(_`**``hints`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.kernel.replace_hints "Link to this definition")
```

Return a new kernel with updated compiler hints.

Notes:

```default
Because hints affects compilation, the returned object will have its
own JIT cache.
```

Examples:

SnippetComplete Example

```python
@ct.kernel(occupancy=2)
def kernel():
    pass

# compile
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
# cache hit
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())

new_kernel = kernel.replace_hints(occupancy=4)

# compile with new hints
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
# cache hit
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
```

```python
import cuda.tile as ct
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

@ct.kernel(occupancy=2)
def kernel():
    pass

# compile
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
# cache hit
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())

new_kernel = kernel.replace_hints(occupancy=4)

# compile with new hints
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
# cache hit
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())

torch.cuda.synchronize()
```

```
`cuda.tile.``launch`(_`stream`_, _`grid`_, _`kernel`_, _`kernel_args`_, _`/`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.launch "Link to this definition")
```

Launch a cuTile kernel.

**Parameters:**
: - **stream** – The CUDA stream to execute the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) on.
- **grid** – Tuple of up to 3 grid dimensions to execute the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) over.
- **kernel** – The [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels) to execute.
- **kernel_args** – Positional arguments to pass to the kernel.
