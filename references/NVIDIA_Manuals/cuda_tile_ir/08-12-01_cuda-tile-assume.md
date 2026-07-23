---
title: "8.12.1. cuda_tile.assume"
section: "8.12.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-assume"
---

### [8.12.1. cuda_tile.assume](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-assume)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-assume "Permalink to this headline")

_Attach static information to an SSA value_

```default
cuda_tile.assume %value %predicate
```

#### [Parameters](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id410)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id410 "Permalink to this headline")

- **value** ([Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)) - The value to attach the predicate to. 13.1
- **predicate** ([AssumePredicate](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-attribute-cuda-tile-assume-assumepredicate-attr)) - The predicate to attach to the value. 13.1

#### [Results](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id411)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id411 "Permalink to this headline")

- **result** ([Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)) - The value with the attached predicate. 13.1

#### [Description](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id412)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id412 "Permalink to this headline")

The `assume` operation passes through `value` as the result and
attaches a predicate to it. The assumed predicate is a property of
`result`.

This operation can be used to inject static information into the compiler,
potentially resulting in more efficient code generation.

`predicate` must implement the `AssumePredicateAttrInterface`.

> **Note**
>
> `assume` does not check the correctness of the predicate.
> Incorrect predicates may inject incorrect static information and cause
> miscompilation. If an incorrect predicate is attached to an SSA value,
> the behavior of the program is undefined.

#### [AssumePredicate Implementers](https://docs.nvidia.com/cuda/tile-ir/latest/sections#assumepredicate-implementers)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#assumepredicate-implementers "Permalink to this headline")

#### [Bounded](https://docs.nvidia.com/cuda/tile-ir/latest/sections#bounded)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#bounded "Permalink to this headline")

```mlir
#bounded<(lb|?), (ub|?)>
```

The `bounded` attribute must be used as a predicate for
`cuda_tile.assume`. The predicated value must be a tile of integers.

`bounded` specifies a lower and upper bound for all elements of the
predicated tile when interpreted as signed integers. Bounds are optional:
it is possible to leave a bound unspecified, as indicated by “?” in the
assembly format. E.g., `#bounded<0, ?>`. Both lower bound and upper
bound are inclusive.

The lower bounds must be less than or equal to the upper bound. A lower/
upper bound that exceeds the range of valid values of the predicated value
is invalid.

```mlir
%1 = cuda_tile.assume #cuda_tile.bounded<0, ?>, %0
    : !cuda_tile.tile<4x8xi16>
```

#### [DivBy](https://docs.nvidia.com/cuda/tile-ir/latest/sections#divby)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#divby "Permalink to this headline")

```mlir
div_by< $divisor (, every $every^ along $along)?>
```

The `div_by` attribute must be used as a predicate for `cuda_tile.assume`
ops. The predicated value must be a `tile` of integers or pointers, or
a `tensor_view`.

If the predicated value is a `tile`, the attribute indicates that some
elements of the `tile` are divisible by `divisor`. If the predicated value
is a `tensor_view` the attribute indicates that the base address of the `tensor_view` is
divisible by `divisor`. `divisor` must be a positive power of `2`.

The `every` and `along` attributes control which elements are assumed to
satisfy the divisibility property. When splitting the tensor in groups of
size `every` along dimension `along`, the first element of each group is
assumed to satisfy the divisibility property. The other elements are
assumed to be monotonically increasing by `1` within the group. In case
of a `tile` of pointers, the elements are assumed to be monotonically
increasing by the byte width of the pointee type. The size of the last
group may be smaller than `every`.

The `every` and `along` attributes are optional. When missing, they are
assumed to have a default value of `1` and `0` in case of a `tile`.
I.e., all elements of the `tile` are assumed to satisfy the divisibility
property. (The value of `along` does not matter in that case.) If the
predicated value is a `tensor_view` or a 0D `tile`, `every` and `along` cannot be
used.

`every`, and `along` must be used together. If one is specified,
so must be the other.

> **Note**
>
> If the predicated value is a tile of integers, `every` is a property of
> the signed interpretation of the integer values. Otherwise, it is a
> property of the unsigned integer interpretation. E.g., `every = 4`
> is incorrect for the following sequence of “i8” values (written in binary
> form) because they wrap around when interpreted as signed integers:
> `[01111110, 01111111, 10000000, 10000001]`. `every = 2` would
> be correct.

The examples below demonstrate tensors that satisfy the assumed properties.

```mlir
// Example 1: Each pointer is divisible by 16.
// [ 0x10, 0x20, 0x80, 0x10, 0x0, 0x120, ... ]
%0 = cuda_tile.assume #cuda_tile.div_by<16>, %ptrs
    : !cuda_tile.tile<128x!cuda_tile.ptr<f32>>
// Note: Equivalent to #cuda_tile.div_by<16, every 1 along 0>.
```

```mlir
// Example 2: Each integer is divisible by 4.
// [ 16, 24, 8, 4, 12, 12, 0, 16, ... ]
%0 = cuda_tile.assume #cuda_tile.div_by<4>, %t
    : !cuda_tile.tile<128xi32>
```

```mlir
// Example 3: Group size [4].
// [7, 8, 9, 10, 23, 24, 25, 26, 0, 1, 2, 3, ...]
%0 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 0>, %t
    : !cuda_tile.tile<128xi32>
```

```mlir
// Example 4: 2-d Group size [1, 4] with divisibility 4.
// [ [  4,  5,  6,  7, 12, 13, 14, 15 ],
//   [  8,  9, 10, 11, 24, 25, 26, 27 ],
//   [ 24, 25, 26, 27, 64, 65, 66, 67 ],
//   [  0,  1,  2,  3,  4,  5,  6,  7 ] ]
%0 = cuda_tile.assume #cuda_tile.div_by<4, every 4 along 1>, %t
    : !cuda_tile.tile<4x8xi32>
```

```mlir
// Example 5: 2-d Group size [4, 1] with divisibility 32.
// Note that the elements within each column are monotonically increasing
// by the byte width of the pointee type f32, e.g., 0x20, 0x24, 0x28, 0x2c.
// [ [  0x20, 0x100,  0x40,  0x60,  0x40, 0x200, 0x340,  0x40 ],
//   [  0x24, 0x104,  0x44,  0x64,  0x44, 0x204, 0x344,  0x44 ],
//   [  0x28, 0x108,  0x48,  0x68,  0x48, 0x208, 0x348,  0x48 ],
//   [  0x2c, 0x10c,  0x4c,  0x6c,  0x4c, 0x20c, 0x34c,  0x4c ] ]
%0 = cuda_tile.assume #cuda_tile.div_by<32, every 4 along 0>, %ptrs
    : !cuda_tile.tile<4x8x!cuda_tile.ptr<f32>>
```

#### [SameElements](https://docs.nvidia.com/cuda/tile-ir/latest/sections#sameelements)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sameelements "Permalink to this headline")

```mlir
#same_elements< $values >
```

The `same_elements` attribute must be used as a predicate for
`cuda_tile.assume`. The predicated value must be a tensor of integers or
pointers.

`same_elements` is specified for each dimension. A value of C for a
dimension of size N indicates that, after dividing the respective
dimension into N/C groups of size C, each group consists of the same
elements. As N/C may not divide evenly, the last group may have fewer
than C elements.

If the “same elements” property does not hold along a dimension, the
respective value should be set to 1.
`#cuda_tile.same_elements<[1, 1, ..., 1]>` is a correct predicate for any
tensor of integers or pointers, where the number of ones matches the rank
of the tensor. (Size-1 groups always have the same elements.)

```mlir
// Integer tensor with same elements.
%0 = cuda_tile.constant <i16: [[0, 0, 0, 0, 10, 10, 10, 10],
                               [0, 0, 0, 0, 10, 10, 10, 10],
                               [5, 5, 5, 5, 93, 93, 93, 93],
                               [5, 5, 5, 5, 93, 93, 93, 93]]>
    : tile<4x8xi16>
%1 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %0
    : !cuda_tile.tile<4x8xi16>

// Pointer tensor with same elements.
%2 = cuda_tile.constant <i64: [[ 0,  0,  0,  0,  8,  8,  8,  8],
                               [ 0,  0,  0,  0,  8,  8,  8,  8],
                               [64, 64, 64, 64, 32, 32, 32, 32],
                               [64, 64, 64, 64, 32, 32, 32, 32]]>
    : tile<4x8xi64>
%3 = cuda_tile.bitcast %2
    : !cuda_tile.tile<4x8xi64>
      -> !cuda_tile.tile<!cuda_tile.ptr<f32>>
%4 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %3
    : !cuda_tile.tile<!cuda_tile.ptr<f32>>
```

#### [Constraints](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id413)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id413 "Permalink to this headline")

- `value` and `result` must have the same shape and element type ([Any](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-any)).
- Operation must infer result types from operands and attributes.

#### [Examples](https://docs.nvidia.com/cuda/tile-ir/latest/sections#id414)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#id414 "Permalink to this headline")

```mlir
// Assume that all integers are divisible by 32.
%int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
%div_by_1 = assume div_by<32>, %int_tile : tile<8xi16>

// Assume that every 4th element (starting with element 0) along
// dimension 0 is divisible by 32 that and all integers are
// montonically increasing by 1 within each group of 4.
%int_tile_2 = constant <i16: [96, 97, 98, 99, 64, 65, 66, 67]> : tile<8xi16>
%div_by_2 = assume div_by<32, every 4 along 0>, %int_tile_2 : tile<8xi16>

// Assume that every rectangular chunk of size [1, 4, 2] has the same
// values.
%same_elem = assume same_elements<[1, 4, 2]>, %ptr_3d : tile<1x8x8xptr<f32>>

// Assume that every value is greater or equal to 5.
%int_tile_3 = constant <i16: [5, 9, 10, 11, 6, 5, 5, 7]> : tile<8xi16>
%bounded = assume bounded<5, ?>, %int_tile_3 : tile<8xi16>
```

See [cuda_tile.assume_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#example-cuda-tile-assume-0) for the full example listing.
