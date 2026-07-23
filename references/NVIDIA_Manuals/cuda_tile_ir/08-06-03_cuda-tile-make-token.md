---
title: "8.6.3. cuda_tile.make_token"
section: "8.6.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-make-token"
---

### [8.6.3. cuda_tile.make_token](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-make-token)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-make-token "Permalink to this headline")

_Create a fresh token with no prior dependencies_

```default
cuda_tile.make_token
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id159)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id159 "Permalink to this headline")

No parameters.

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id160)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id160 "Permalink to this headline")

- **result** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - A fresh token with no prior dependencies. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id161)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id161 "Permalink to this headline")

The `make_token` operation creates a fresh token with no prior dependencies.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id162)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id162 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.
