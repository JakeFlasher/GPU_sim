---
title: "8.11.5. cuda_tile.make_tensor_view"
section: "8.11.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-make-tensor-view"
---

### [8.11.5. cuda_tile.make_tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-make-tensor-view)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-make-tensor-view "Permalink to this headline")

_Create :code:`tensor_view` from a pointer to global memory_

```default
cuda_tile.make_tensor_view %base %dynamicShape %dynamicStrides
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id400)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id400 "Permalink to this headline")

- **base** ([tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[ptr](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr)>) - The scalar base pointer to a portion of global memory. 13.1
- **dynamicShape** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>>) - The array of values representing the shape of the view, may be fully dynamic. 13.1
- **dynamicStrides** ([Variadic](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-variadic)<[tile](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile)<[any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)>>) - The array of values representing the strides of the view, may be fully dynamic. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id401)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id401 "Permalink to this headline")

- **result** ([tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view)) - The constructed tensor_view. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id402)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id402 "Permalink to this headline")

The `make_tensor_view` operation constructs a `tensor_view` from a global
memory pointer, a dynamic shape and dynamic strides. See [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) for more details.

The constructor supports taking dynamic arrays for shapes and strides as part of the constructor
enabling workloads to take global memory tensors of dynamic shape and strides. If these arguments
are static they will be statically reflected in the type of the resulting `tensor_view`, if
they are dynamic they will appear as `?` in the type. See below for concrete examples.

The `dynamicShape` and `dynamicStrides` operands are interpreted as unsigned integers.

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id403)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id403 "Permalink to this headline")

- Operation must encode variadic operand segment sizes in attributes.
- Operation must not perform any memory side effects.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id404)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id404 "Permalink to this headline")

```mlir
  // tensor_view to a scalar tile of f32
  %a0 = make_tensor_view %base,
      shape = [], strides = [] : tensor_view<f32>

  // tensor_view to a tile of static shape and strides
  %a1 = make_tensor_view %base,
      shape = [32, 32], strides = [32, 1]
      : tensor_view<32x32xf32, strides=[32,1]>

%sh0 = constant <i32: 32> : tile<i32>
%sh1 = constant <i32: 32> : tile<i32>
%st0 = constant <i32: 32> : tile<i32>
%st1 = constant <i32: 1> : tile<i32>

  // tensor_view to a tile with partially dynamic shape and strides
  // all dynamic values must be of the same type, here tile<i32>
  %a2 = make_tensor_view %base,
          shape = [%sh0, %sh1], strides = [%st0, %st1]
          : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
```

See [cuda_tile.make_tensor_view_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-make-tensor-view-0) for the full example listing.
