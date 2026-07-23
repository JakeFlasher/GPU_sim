---
title: "Computing Composition"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#computing-composition"
---

### [Computing Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#computing-composition)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#computing-composition "Permalink to this headline")

First, a few observations:

- `B = (B_0, B_1, ...)`. A layout can be expressed as the concatenation of its sublayouts.
- `A o B = A o (B_0, B_1, ...) = (A o B_0, A o B_1, ...)`. When `B` is injective, composition is left-distributive with concatenation.

With the above, we can assume without loss of generality that `B = s:d` is a layout with integral shape and stride. We can also assume that `A` is a flattened, coalesced layout.

When `A` is integral, `A = a:b`, the result is rather trivial: `R = A o B = a:b o s:d = s:(b*d)`. Here, the result of the composition `R` are the first `s` elements of `A` strided by `d`.

When `A` is multimodal, we need to be more careful. Put into words, `A o B = A o s:d`, for integral `s` and `d` means that we want to:

1. Determine a layout that produces every `d`th element of `A`.

The shape of this intermediate layout can be calculated by progressively “dividing out” the first `d` elements from the shape of `A` starting from the left.

For example,

- `(6,2) /  2 => (3,2)`
- `(6,2) /  3 => (2,2)`
- `(6,2) /  6 => (1,2)`
- `(6,2) / 12 => (1,1)`
- `(3,6,2,8) /  3 => (1,6,2,8)`
- `(3,6,2,8) /  6 => (1,3,2,8)`
- `(3,6,2,8) /  9 => (1,2,2,8)`
- `(3,6,2,8) / 72 => (1,1,1,4)`

To compute the strides of the strided layout, the residues of the above operation are used to scale the strides of `A`. For instance, the last example `(3,6,2,8):(w,x,y,z) / 72` with strides `(w,x,y,z)` produces `(72*w,24*x,4*x,2*z)` as the strides of the strided layout.

As you may have noticed, we can only divide shapes by certain values and get a sensible result. This is called the **stride divisibility condition** and is statically checked in CuTe when possible.

2. Keep the first `s` elements of the newly strided `A` so that the result has a compatible shape with `B`. This can be computed by “modding out” the first `s` elements from the shape of `A` starting from the left.

For example,

- `(6,2) %  2 => (2,1)`
- `(6,2) %  3 => (3,1)`
- `(6,2) %  6 => (6,1)`
- `(6,2) % 12 => (6,2)`
- `(3,6,2,8) %  6 => (3,2,1,1)`
- `(3,6,2,8) %  9 => (3,3,1,1)`
- `(1,2,2,8) %  2 => (1,2,1,1)`
- `(1,2,2,8) % 16 => (1,2,2,4)`

This operation causes the result to have a shape that is compatible with `B`.

Again, this operation must satisfy a **shape divisibility condition** to yield a sensible result and is statically checked in CuTe when possible.

From the above examples, we can construct the composition `(3,6,2,8):(w,x,y,z) o 16:9 = (1,2,2,4):(9*w,3*x,y,z)`.

---

#### [Example 1 – Worked Example of Calculating a Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-1-worked-example-of-calculating-a-composition)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-1-worked-example-of-calculating-a-composition "Permalink to this headline")

We provide a more complex example of composition, where both operand layouts are multi-modal to illustrate the concepts introduced above.

```console
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

1. Using the left-distributive and concatenation properties for layouts we write the composition as,

R = A o B
  = (6,2):(8,2) o (4,3):(3,1)
  = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)

---
1. Compute `(6,2):(8,2) o 4:3`

- First, we compute the strided layout,

(6,2):(8,2) / 3 = (6/3,2):(8*3,2) = (2,2):(24,2)

- Next, we keep the shape compatible,

(2,2):(24,2) % 4 = (2,2):(24,2)

---
2. Compute `(6,2):(8,2) o 3:1`

- First, we compute the strided layout

(6,2):(8,2) / 1 = (6,2):(8,2)

- Next, we keep the shape compatible,

(6,2):(8,2) % 3 = (3,1):(8,2)

---

Putting this together and coalescing each mode, we obtain the result

R = A o B
  = ((2, 2), 3): ((24, 2), 8)
```

#### [Example 2 – Reshape a layout into a matrix](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-2-reshape-a-layout-into-a-matrix)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-2-reshape-a-layout-into-a-matrix "Permalink to this headline")

`20:2  o  (5,4):(4,1)`. Composition formulation.

This describes interpreting the layout `20:2`
as a 5x4 matrix in a row-major order.

1. ` = 20:2 o (5:4,4:1)`. Layout `(5,4):(4,1)` as concatenation of sublayouts.
2. ` = (20:2 o 5:4, 20:2 o 4:1)`. Left distributivity.
  - `20:2 o 5:4  =>  5:8`. Trivial case.
  - `20:2 o 4:1  =>  4:2`. Trivial case.
3. ` = (5:8, 4:2)`. Composed Layout as concatenation of sublayouts.
4. ` = (5,4):(8,2)`. Final composed layout.

#### [Example 3 – Reshape a layout into a matrix](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#example-3-reshape-a-layout-into-a-matrix)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#example-3-reshape-a-layout-into-a-matrix "Permalink to this headline")

`(10,2):(16,4)  o  (5,4):(1,5)`

This describes interpreting the layout `(10,2):(16,4)`
as a 5x4 matrix in a column-major order.

1. ` = (10,2):(16,4) o (5:1,4:5)`. Layout `(5,4):(1,5)` as concatenation of sublayouts.
2. ` = ((10,2):(16,4) o 5:1, (10,2):(16,4) o 4:5)`. Left distributivity.
  - `(10,2):(16,4) o 5:1 => (5,1):(16,4)`. Mod out the shape `5`.
  - `(10,2):(16,4) o 4:5 => (2,2):(80,4)`. Div out the stride `5`.
3. ` = ((5,1):(16,4), (2,2):(80,4))`. Composed Layout as concatenation of sublayouts.
4. ` = (5:16, (2,2):(80,4))`. By-mode coalesce.
5. ` = (5,(2,2))):(16,(80,4))`. Final composed layout.

We get exactly this result with CuTe
if we use compile-time shapes and strides.
The following C++ code prints `(_5,(_2,_2)):(_16,(_80,_4))`.

```cpp
Layout a = make_layout(make_shape (Int<10>{}, Int<2>{}),
                       make_stride(Int<16>{}, Int<4>{}));
Layout b = make_layout(make_shape (Int< 5>{}, Int<4>{}),
                       make_stride(Int< 1>{}, Int<5>{}));
Layout c = composition(a, b);
print(c);
```

If we use dynamic integers, the following C++ code prints `((5,1),(2,2)):((16,4),(80,4))`.

```cpp
Layout a = make_layout(make_shape (10, 2),
                       make_stride(16, 4));
Layout b = make_layout(make_shape ( 5, 4),
                       make_stride( 1, 5));
Layout c = composition(a, b);
print(c);
```

The results may _look_ different but are the mathematically the same. The 1s in the shape don’t affect the layout as a mathematical function from 1-D coordinates to integers or as a function from 2-D coordinates to integers. In the dynamic case, CuTe can not coalesce the dynamic size-1 modes to “simplify” the layout due to the static rank and type of the tuples containing them.
