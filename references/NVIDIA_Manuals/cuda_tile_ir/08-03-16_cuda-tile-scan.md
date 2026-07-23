---
title: "8.3.16. cuda_tile.scan"
section: "8.3.16"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-scan"
---

### [8.3.16. cuda_tile.scan](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-scan)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-scan "Permalink to this headline")

_A parallel prefix sum operation_

```default
cuda_tile.scan %operands %dim %reverse %identities
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id65)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id65 "Permalink to this headline")

- **operands** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)>) - The a set of tiles to scan. 13.1
- **dim** ([i32](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types)) - The index of the dimension along which to scan. 13.1
- **reverse** ([bool](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-bool)) - Whether to scan in reverse order. 13.1
- **identities** ([Array](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-array)) - The identities of the scan operation. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id66)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id66 "Permalink to this headline")

- **results** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)>) - The resulting tiles from the scan operation. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id67)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id67 "Permalink to this headline")

The `scan` operation computes an inclusive parallel prefix along a given
dimension of the input tiles using a binary associative function and an identity.

The `scan` operation applies a scan function defined over a tile of elements
for a given type, utilizing an associative operation and an identity value. It
operates on `operands` and `identities` across the specified `dim`,
producing new `results` tile values. The exact evaluation order within each
prefix is implementation-defined but the result remains deterministic across different
runs of the same kernel on the same device.

> **Note**
>
> Only pure operations are allowed in the body of `scan`.

$$
\[\text{scan}(X, \text{dim}, \text{identity}, f)_{i_1,\ldots,i_d}[j] \;=\;
\text{fold}\!\left(f, \text{identity},
  \left(X_{i_1,\ldots,i_{\text{dim}-1}, 0, i_{\text{dim}+1},\ldots,i_d}, \ldots,
        X_{i_1,\ldots,i_{\text{dim}-1}, j, i_{\text{dim}+1},\ldots,i_d}\right)\right)\]
$$

The scan preserves all intermediate accumulator values:

$$
\[\begin{split}\text{result}[0] \;=\; f(\text{identity}, X[\ldots, 0, \ldots]) \\
\text{result}[1] \;=\; f(\text{result}[0], X[\ldots, 1, \ldots]) \\
\vdots \\
\text{result}[j] \;=\; f(\text{result}[j-1], X[\ldots, j, \ldots])\end{split}\]
$$

When `reverse` is `true`, the prefix is taken in decreasing index order.
Let \(\(N\)\) be the size of the scanned dimension; then:

$$
\[\text{scan}_{\text{rev}}(X)[j] \;=\;\
\text{fold}\!\left(f, \text{identity},
  \left(X[\ldots, N\!-\!1,\ldots], \ldots, X[\ldots, j,\ldots]\right)\right)\]
$$

The `identities` attribute is a list of identity elements for each input
tile; the identity at position `i` binds with the operand tile at the same
position. The correct identity is a property of the scan function in the `body`
(e.g., `sum` uses 0, `prod` uses 1, `min` uses +inf, `max` uses -inf).

The `body` region represents the binary associative operation. The region must
contain **Tile IR** operations with 0-rank tile types. Region arguments are bound in
operand order as `[op_0_current_iter, op_0_prev_iter, op_1_current_iter, op_1_prev_iter, ...]`,
where `op_i_current_iter` is the current element along `dim` and
`op_i_prev_iter` is the running accumulator for operand `i`. On the first
step, the accumulator is the corresponding identity element.

> **Note**
>
> Associativity of the binary operation permits the compiler to reorganize the
> applications of the operation to achieve efficient parallel prefix scans on the GPU.

> **Warning**
>
> The *scan* operation is restricted to only support single tile input.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id68)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id68 "Permalink to this headline")

- Operation must provide custom parsing and printing methods.
- Operation effects must include all nested operations’ effects.
- All operands must have identical shapes.
- Each region must contain exactly one block.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id69)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id69 "Permalink to this headline")

```mlir
%input = constant <f32: 0.0> : tile<8x16xf32>
%result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
(%acc: tile<f32>, %elem: tile<f32>) {
  %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
  yield %prod : tile<f32>
}
```

See [cuda_tile.scan_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-scan-0) for the full example listing.
