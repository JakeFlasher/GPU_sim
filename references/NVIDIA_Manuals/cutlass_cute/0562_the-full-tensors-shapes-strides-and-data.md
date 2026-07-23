---
title: "The Full Tensors: Shapes, Strides, and Data"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#the-full-tensors-shapes-strides-and-data"
---

### [The Full Tensors: Shapes, Strides, and Data](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#the-full-tensors-shapes-strides-and-data)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#the-full-tensors-shapes-strides-and-data "Permalink to this headline")

Most GEMM interfaces list the matrices’ dimensions
in the order M, N, K. CuTe also uses this convention, but packages them
into a single `IntTuple`. In this example, they are dynamic values
defined at the top of the `gemm_nt` and `gemm_tn` host functions
that invoke the device kernel.

```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);    // (M, N, K)
```

Inside the kernel, the problem shape is checked against the preconditions and then used to construct each of the full matrices.

```cpp
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                      // (M, N, K)

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));            // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));            // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));            // dC strides for shape MN

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)
```

The appropriate modes of the `Shape` are selected to construct each of the tensors. The preconditions make sure that for every integer in the `Shape` there is a corresponding integer in the associated `Stride`.

Note that the comment after B says `(N,K)` rather than `(K,N)`.
This means that B is treated as an NxK matrix instead of a KxN matrix as is typical within BLAS and most other matrix-matrix multiplications.
CuTe follows the convention that the semantics of matrix modes is
`(M,K)` for `A`, `(N,K)` for `B`, and `(M,N)` for `C`, which we try to record in comments everywhere.

For each of the `(M,K)`, `(N,K)`, and `(M,N)` tensors, the `gemm_nt` and `gemm_tn` construct the strides those tensors will use. In `gemm_nt` the strides are defined as

```cpp
  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);    // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)
```

and in `gemm_tn` the strides are defined as

```cpp
  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});    // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)
```

#### [Aside: M-major, N-major, K-major](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#aside-m-major-n-major-k-major)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#aside-m-major-n-major-k-major "Permalink to this headline")

We’ve found that the BLAS convention of using “non-transposed” (N) and “transposed” (T) flags in conjunction with the mode conventions of `MxK * KxN` to confuse the core issue of “what layout does this matrix use” and “in which mode does my matrix have a stride-1?”. Indeed, the answer to those questions can always be found by inspecting the CuTe `Layout`.

Instead of row-major or column-major (or Transposed
and Not-Transposed), we have found it much more convenient to say that a matrix is “M-major” if it is stride-1 in the M-mode, “N-major” if it is stride-1 in the N-mode, or “K-major” if it is stride-1 in the K-mode. Furthermore, knowing that matrix multiply always performs a reduction in the K-mode, it is very convenient from a software perspective to always have the K-mode in the same place and adopt the mode convention `MxK * NxK`. Implementations will always reduce over the second mode (the K mode) of both input matrices and leads to cases where implementations can treat both input matrices the same way.

How do we translate this into the BLAS user’s experience?

| BLAS | A Majorness | A Layout | B Majorness | B Layout |
| --- | --- | --- | --- | --- |
| NT | M-major | `(M,K):(1,ldA)` | N-major | `(N,K):(1,ldB)` |
| TN | K-major | `(M,K):(ldA,1)` | K-major | `(N,K):(ldB,1)` |
| NN | M-major | `(M,K):(1,ldA)` | K-major | `(N,K):(ldB,1)` |
| TT | K-major | `(M,K):(ldA,1)` | N-major | `(N,K):(1,ldB)` |

Regardless, we’ll still use the BLAS “NT” and “TN” notations for high-level descriptions of kernels when it’s appropriate.
