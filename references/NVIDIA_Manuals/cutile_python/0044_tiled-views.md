---
title: "Tiled Views"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#tiled-views"
---

## [Tiled Views](https://docs.nvidia.com/cuda/cutile-python#tiled-views)[](https://docs.nvidia.com/cuda/cutile-python/#tiled-views "Permalink to this headline")

A _tiled view_ represents the [tile space](https://docs.nvidia.com/cuda/cutile-python/#data-element-tile-space) of a [global array](https://docs.nvidia.com/cuda/cutile-python/#data-global-arrays).

A tiled view’s _num_tiles_ is a tuple of integers, each denoting the number of tiles along
the corresponding dimension. The length of the _num_tiles_ tuple equals the tile space’s number
of dimensions, and the product of its values equals the total number of tiles in the tile space.

A tile in the tiled view can be loaded or stored by its tile index.

> **See also**
>
> [cuda.tile.TiledView class documentation](https://docs.nvidia.com/cuda/cutile-python/data/tiled_view.html#data-tiled-view-cuda-tile-tiled-view)
>
> [`Array.tiled_view()`](https://docs.nvidia.com/cuda/cutile-python/data/array.html#cuda.tile.Array.tiled_view "cuda.tile.Array.tiled_view")
