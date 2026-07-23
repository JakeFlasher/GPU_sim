---
title: "Strides aren’t just integers"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html#strides-aren-t-just-integers"
---

### [Strides aren’t just integers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#strides-aren-t-just-integers)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#strides-aren-t-just-integers "Permalink to this headline")

Ordinary tensors have a layout that maps
a logical coordinate `(i,j)` into a 1-D linear index `k`.
This mapping is the inner-product of the coordinate with the strides.

TMA Tensors hold iterators of TMA coordinates.
Thus, a TMA Tensor’s Layout must map a logical coordinate
to a TMA coordinate, rather than to a 1-D linear index.

To do this, we can abstract what a stride is. Strides need not be integers, but rather any algebraic object that supports inner-product with the integers (the logical coordinate). The obvious choice is the `ArithmeticTuple` we used earlier since they can be added to each other, but this time additionally equipped with an `operator*` so it can also be scaled by an integer.

#### [Aside: Integer-module strides](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#aside-integer-module-strides)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#aside-integer-module-strides "Permalink to this headline")

A group of objects that support addition between elements and product between elements and integers is called an integer-module.

Formally, an integer-module is an abelian group `(M,+)` equipped with `Z*M -> M`, where `Z` are the integers. That is, an integer-module `M` is
a group that supports inner products with the integers.
The integers are an integer-module.
Rank-R tuples of integers are an integer-module.

In principle, layout strides may be any integer-module.

#### [Basis elements](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#basis-elements)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#basis-elements "Permalink to this headline")

CuTe’s basis elements live in the header file `cute/numeric/arithmetic_tuple.hpp`.
To make it easy to create `ArithmeticTuple`s that can be used as strides, CuTe defines normalized basis elements using the `E` type alias. “Normalized” means that the scaling factor of the basis element is the compile-time integer 1.

| C++ object | Description | String representation |
| --- | --- | --- |
| `E<>{}` | `1` | `1` |
| `E<0>{}` | `(1,0,...)` | `1@0` |
| `E<1>{}` | `(0,1,0,...)` | `1@1` |
| `E<0,0>{}` | `((1,0,...),0,...)` | `1@0@0` |
| `E<0,1>{}` | `((0,1,0,...),0,...)` | `1@1@0` |
| `E<1,0>{}` | `(0,(1,0,...),0,...)` | `1@0@1` |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1` |

The “description” column in the above table
interprets each basis element as an infinite tuple of integers,
where all the tuple’s entries not specified by the element’s type are zero.
We count tuple entries from left to right, starting with zero.
For example, `E<1>{}` has a 1 in position 1: `(0,1,0,...)`.
`E<3>{}` has a 1 in position 3: `(0,0,0,1,0,...)`.

Basis elements can be _nested_.
For instance, in the above table, `E<0,1>{}` means that
in position 0 there is a `E<1>{}`: `((0,1,0,...),0,...)`. Similarly,
`1@1@0` means that `1` is lifted to position 1 to create `1@1`: `(0,1,0,...)`
which is then lifted again to position 0.

Basis elements can be _scaled_.
That is, they can be multiplied by an integer _scaling factor_.
For example, in `5*E<1>{}`, the scaling factor is `5`.
`5*E<1>{}` prints as `5@1` and means `(0,5,0,...)`.
The scaling factor commutes through any nesting.
For instance, `5*E<0,1>{}` prints as `5@1@0`
and means `((0,5,0,...),0,...)`.

Basis elements can also be added together,
as long as their hierarchical structures are compatible.
For example, `3*E<0>{} + 4*E<1>{}` results in `(3,4,0,...)`.
Intuitively, “compatible” means that
the nested structure of the two basis elements
matches well enough to add the two elements together.

#### [Linear combinations of strides](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#linear-combinations-of-strides)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#linear-combinations-of-strides "Permalink to this headline")

Layouts work by taking the inner product
of the natural coordinate with their strides.
For strides made of integer elements, e.g., `(1,100)`,
the inner product of the input coordinate `(i,j)`
and the stride is `i + 100j`.
Offsetting an “ordinary” tensor’s pointer and this index
gives the pointer to the tensor element at `(i,j)`.

For strides of basis elements, we still compute the inner product of the natural coordinate with the strides.
For example, if the stride is `(1@0,1@1)`,
then the inner product of the input coordinate `(i,j)`
with the strides is `i@0 + j@1 = (i,j)`.
That translates into the (TMA) coordinate `(i,j)`.
If we wanted to reverse the coordinates,
then we could use `(1@1,1@0)` as the stride.
Evaluating the layout would give `i@1 + j@0 = (j,i)`.

A linear combination of basis elements
can be interpreted as a possibly multidimensional and hierarchical coordinate.
For instance, `2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0`
means `((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`
and can be interpreted as the coordinate `((7,4),23)`.

Thus, linear combinations of these strides can be used to generate TMA coordinates.
These coordinates, in turn, can be used to offset TMA coordinate iterators.
