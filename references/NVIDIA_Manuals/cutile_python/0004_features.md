---
title: "Features"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html#features"
---

### [Features](https://docs.nvidia.com/cuda/cutile-python/generated#features)[](https://docs.nvidia.com/cuda/cutile-python/generated/#features "Permalink to this headline")

- Add API for ahead-of-time compilation and export via [`compilation.export_kernel()`](https://docs.nvidia.com/cuda/cutile-python/compilation.html#cuda.tile.compilation.export_kernel "cuda.tile.compilation.export_kernel").
See the [Compilation and Export](https://docs.nvidia.com/cuda/cutile-python/compilation.html) section for more details.
- Add API for autotuning via [`tune.exhaustive_search()`](https://docs.nvidia.com/cuda/cutile-python/performance.html#cuda.tile.tune.exhaustive_search "cuda.tile.tune.exhaustive_search") and the following helpers:
  - Add API [`kernel.replace_hints()`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel.replace_hints "cuda.tile.kernel.replace_hints") to get a new kernel with updated hints.
  - Add API function [`compiler_timeout()`](https://docs.nvidia.com/cuda/cutile-python/debugging.html#cuda.tile.compiler_timeout "cuda.tile.compiler_timeout") for temporarily setting the
timeout on the tileiras compiler.
See the [Autotuning](https://docs.nvidia.com/cuda/cutile-python/performance.html#autotuning) section for more details.
- Add API [`Array.tiled_view()`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.tiled_view "cuda.tile.Array.tiled_view") to create a tiled view of an array
with a fixed tile shape and padding mode.
