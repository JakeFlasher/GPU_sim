---
title: "Autotuning"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/performance.html#autotuning"
---

## [Autotuning](https://docs.nvidia.com/cuda/cutile-python#autotuning)[](https://docs.nvidia.com/cuda/cutile-python/#autotuning "Permalink to this headline")

[`tune.exhaustive_search()`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.tune.exhaustive_search "cuda.tile.tune.exhaustive_search") provides a convenient way to measure kernel performance
on a finite space of configurations and return the best configuration.

```
`cuda.tile.tune.``exhaustive_search`(
```

Searches the entire search space and return the best configuration.

**Parameters:**
: - **search_space** (_Sequence__[__T__]_) – Sequence of configs to evaluate.
- **stream** – The CUDA stream to execute kernel on.
- **grid_fn** (_Callable__[__[__T__]__,__tuple__[__int__,__...__]__]_) – Maps a config to grid dimensions.
- **kernel** – The kernel to tune.
- **args_fn** (_Callable__[__[__T__]__,__tuple__[__Any__,__...__]__]_) – Maps a config to kernel arguments for timing.
- **hints_fn** (_Callable__[__[__T__]__,__dict__[__str__,__Any__]__]__|__None_) – Maps a config to compiler hints. Default: no hints.
- **quiet** (_bool_) – If true, avoid printing any progress or result.

**Returns:**
: TuningResult with the best config and its time in microseconds.

**Return type:**
: [_TuningResult_](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.TuningResult.html#cuda.tile.tune.TuningResult "cuda.tile.tune._tune.TuningResult")[_T_]

Examples:

SnippetComplete Example

```python
# Define the kernel

@ct.kernel
def matmul(X, Y, Out,
           tm: ct.Constant[int],
           tn: ct.Constant[int],
           tk: ct.Constant[int]):

    i, j =  ct.bid(0), ct.bid(1)

    x_view = X.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
    y_view = Y.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((tm, tn), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        ty = y_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    ct.store(Out, (i, j), acc.astype(Out.dtype))

# Tune the kernel

from itertools import product
from cuda.tile import ByTarget

def tune(x, y, out) -> ct.tune.TuningResult:
    keys = ("tm", "tn", "tk", "num_ctas")
    search_space = [dict(zip(keys, vals))
                    for vals in product(
                    (64, 128),
                    (64, 128),
                    (32, 64),
                    (1, 2))]
    grid = lambda cfg: (ct.cdiv(M, cfg['tm']), ct.cdiv(N, cfg['tn']))
    args = lambda cfg: (x, y, out.clone(), cfg['tm'], cfg['tn'], cfg['tk'])
    hints = lambda cfg: {'num_ctas': ByTarget(sm_100=cfg['num_ctas'])}
    stream = torch.cuda.current_stream()
    tuning_result = ct.tune.exhaustive_search(search_space,
                                              stream,
                                              grid,
                                              matmul,
                                              args,
                                              hints)
    return tuning_result

M, N, K = 1024, 256, 512
x = torch.rand((M, K), dtype=torch.float16, device='cuda')
y = torch.rand((K, N), dtype=torch.float16, device='cuda')
out = torch.zeros((M, N), dtype=torch.float16, device='cuda')

result = tune(x, y, out)
print(f"Best config: {result.best.config} ({result.best.mean_us:.1f}us)")

# Launch the kernel with tuned result

tm, tn, tk, num_ctas = result.best.config.values()
kernel = matmul.replace_hints(num_ctas=ByTarget(sm_100=num_ctas))
ct.launch(torch.cuda.current_stream(),
          (ct.cdiv(M, tm), ct.cdiv(N, tn)),
          kernel,
          (x, y, out, tm, tn, tk))

torch.testing.assert_close(out, x @ y)
```

```python
import cuda.tile as ct
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

# Define the kernel

@ct.kernel
def matmul(X, Y, Out,
           tm: ct.Constant[int],
           tn: ct.Constant[int],
           tk: ct.Constant[int]):

    i, j =  ct.bid(0), ct.bid(1)

    x_view = X.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
    y_view = Y.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((tm, tn), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        ty = y_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    ct.store(Out, (i, j), acc.astype(Out.dtype))

# Tune the kernel

from itertools import product
from cuda.tile import ByTarget

def tune(x, y, out) -> ct.tune.TuningResult:
    keys = ("tm", "tn", "tk", "num_ctas")
    search_space = [dict(zip(keys, vals))
                    for vals in product(
                    (64, 128),
                    (64, 128),
                    (32, 64),
                    (1, 2))]
    grid = lambda cfg: (ct.cdiv(M, cfg['tm']), ct.cdiv(N, cfg['tn']))
    args = lambda cfg: (x, y, out.clone(), cfg['tm'], cfg['tn'], cfg['tk'])
    hints = lambda cfg: {'num_ctas': ByTarget(sm_100=cfg['num_ctas'])}
    stream = torch.cuda.current_stream()
    tuning_result = ct.tune.exhaustive_search(search_space,
                                              stream,
                                              grid,
                                              matmul,
                                              args,
                                              hints)
    return tuning_result

M, N, K = 1024, 256, 512
x = torch.rand((M, K), dtype=torch.float16, device='cuda')
y = torch.rand((K, N), dtype=torch.float16, device='cuda')
out = torch.zeros((M, N), dtype=torch.float16, device='cuda')

result = tune(x, y, out)
print(f"Best config: {result.best.config} ({result.best.mean_us:.1f}us)")

# Launch the kernel with tuned result

tm, tn, tk, num_ctas = result.best.config.values()
kernel = matmul.replace_hints(num_ctas=ByTarget(sm_100=num_ctas))
ct.launch(torch.cuda.current_stream(),
          (ct.cdiv(M, tm), ct.cdiv(N, tn)),
          kernel,
          (x, y, out, tm, tn, tk))

torch.testing.assert_close(out, x @ y)

torch.cuda.synchronize()
```

Output

```none
16 succeeded, 0 failed
...
Best config: {'tm': ..., 'tn': ..., 'tk': ..., 'num_ctas': ...} (...us)
```

To achieve consistent result with tuning, it is best to fix GPU clock and memory clock.

Enable persistent mode:

```default
nvidia-smi -i <GPU_ID> -pm 1
```

Query supported clocks:

```default
nvidia-smi -i <GPU_ID> --query-supported-clocks=graphics,memory --format=csv | head
```

Fix graphics and memory clocks:

```default
nvidia-smi -i <GPU_ID> -lgc <MIN_CLOCK>,<MAX_CLOCK>
nvidia-smi -i <GPU_ID> -lmc <MIN_CLOCK>,<MAX_CLOCK>
```
