---
title: "8.6.1. cuda_tile.join_tokens"
section: "8.6.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-join-tokens"
---

### [8.6.1. cuda_tile.join_tokens](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-join-tokens)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-join-tokens "Permalink to this headline")

_Product a new token which depends on the input tokens_

```default
cuda_tile.join_tokens %tokens
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id150)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id150 "Permalink to this headline")

- **tokens** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)>) - The input tokens to join. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id151)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id151 "Permalink to this headline")

- **result** ([token](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-token)) - The joined token. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id152)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id152 "Permalink to this headline")

The `join_tokens` operation produces a fresh token which depends on all input tokens.
Token-ordered operations which consume the new token will then be ordered with respect to all
joined tokens.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id153)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id153 "Permalink to this headline")

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.
