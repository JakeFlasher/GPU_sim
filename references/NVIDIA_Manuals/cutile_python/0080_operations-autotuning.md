---
title: "Autotuning"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/operations.html#operations--autotuning"
---

## [Autotuning](https://docs.nvidia.com/cuda/cutile-python#autotuning)[](https://docs.nvidia.com/cuda/cutile-python/#autotuning "Permalink to this headline")

| [`tune.exhaustive_search`](https://docs.nvidia.com/cuda/cutile-python/performance.html#cuda.tile.tune.exhaustive_search "cuda.tile.tune.exhaustive_search") | Searches the entire search space and return the best configuration. |
| --- | --- |

| [`tune.TuningResult`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.TuningResult.html#cuda.tile.tune.TuningResult "cuda.tile.tune.TuningResult") | Holds the measurement result for each config. |
| --- | --- |
| [`tune.Measurement`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.Measurement.html#cuda.tile.tune.Measurement "cuda.tile.tune.Measurement") | Holds a configuration and its timing result. |

| [`kernel.replace_hints`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel.replace_hints "cuda.tile.kernel.replace_hints") | Return a new kernel with updated compiler hints. |
| --- | --- |
| [`compiler_timeout`](https://docs.nvidia.com/cuda/cutile-python/debugging.html#cuda.tile.compiler_timeout "cuda.tile.compiler_timeout") | Context manager that temporarily sets the compiler timeout. |
