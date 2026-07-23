---
title: "Tiles and Scalars"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#tiles-and-scalars"
---

## [Tiles and Scalars](https://docs.nvidia.com/cuda/cutile-python#tiles-and-scalars)[](https://docs.nvidia.com/cuda/cutile-python/#tiles-and-scalars "Permalink to this headline")

A _tile_ is an immutable multidimensional collection of elements of a specific [dtype](https://docs.nvidia.com/cuda/cutile-python/#data-data-types).

A tile’s _shape_ is a tuple of integers, each denoting the length of the corresponding dimension.
The length of the shape tuple equals the tile’s number of dimensions, and the product of its values
equals the total number of elements in the tile.

The shape of a tile must be known at compile time. Each dimension of a tile must be a power of 2.

A tile’s dtype and shape can be queried with the `dtype` and `shape` attributes, respectively.
For example, if `x` is a *float32* tile, `x.dtype` returns a compile-time constant
equal to [`cuda.tile.float32`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.float32 "cuda.tile.float32").

A zero-dimensional tile is called a _scalar_. A scalar has exactly one element and its shape is
the empty tuple *()*. Numeric literals like *7* or *3.14* are treated as constant scalars,
i.e. zero-dimensional tiles.

Because scalars are tiles, they differ slightly from Python’s `int`/`float` objects.
For example, they have `dtype` and `shape` attributes:

```python
a = 0
# The following line will evaluate to cuda.tile.int32 in cuTile,
# but would raise an AttributeError in Python:
a.dtype
```

Tiles can only be used in [tile code](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-code), not in host code.
A tile’s contents do not necessarily have a physical representation in memory.
Tiles are created by loading from [global arrays](https://docs.nvidia.com/cuda/cutile-python/#data-global-arrays) using functions such as
[`cuda.tile.load()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html#cuda.tile.load "cuda.tile.load") and [`cuda.tile.gather()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.gather.html#cuda.tile.gather "cuda.tile.gather"), or with [factory](https://docs.nvidia.com/cuda/cutile-python/operations.html#operations-factory) functions
such as [`cuda.tile.zeros()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.zeros.html#cuda.tile.zeros "cuda.tile.zeros").

Tiles can be stored back into global arrays using functions such as [`cuda.tile.store()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html#cuda.tile.store "cuda.tile.store")
and [`cuda.tile.scatter()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.scatter.html#cuda.tile.scatter "cuda.tile.scatter").

Scalar constants are [loosely typed](https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-constant-expressions-objects) by default, for example, a literal `2` or
a constant attribute like `Tile.ndim`, `Tile.shape`, or `Array.ndim`.

> **See also**
>
> [cuda.tile.Tile class documentation](https://docs.nvidia.com/cuda/cutile-python/data/tile.html#data-tile-cuda-tile-tile)
