---
title: "5.4.5.2. Built-in Atomic Functions"
section: "5.4.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-atomic-functions"
---

### [5.4.5.2. Built-in Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#built-in-atomic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#built-in-atomic-functions "Permalink to this headline")

CUDA 12.8 and later support CUDA compiler built-in functions for atomic operations, following the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) and the CUDA [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes). The functions follow the [GNU’s atomic built-in function signature](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) with an extra argument for thread scope.

`nvcc` defines the macro `__CUDACC_DEVICE_ATOMIC_BUILTINS__` when built-in atomic functions are supported.

Below are listed the raw enumerators for the [memory orders](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes), which are used as the `order` and `scope` arguments of the built-in atomic functions:

```cuda
// atomic memory orders
enum {
   __NV_ATOMIC_RELAXED,
   __NV_ATOMIC_CONSUME,
   __NV_ATOMIC_ACQUIRE,
   __NV_ATOMIC_RELEASE,
   __NV_ATOMIC_ACQ_REL,
   __NV_ATOMIC_SEQ_CST
};
```

```cuda
// thread scopes
enum {
   __NV_THREAD_SCOPE_THREAD,
   __NV_THREAD_SCOPE_BLOCK,
   __NV_THREAD_SCOPE_CLUSTER,
   __NV_THREAD_SCOPE_DEVICE,
   __NV_THREAD_SCOPE_SYSTEM
};
```

- The memory order corresponds to [C++ standard atomic operations’ memory order](https://en.cppreference.com/w/cpp/atomic/memory_order).
- The thread scope follows the `cuda::thread_scope` [definition](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).
- `__NV_ATOMIC_CONSUME` memory order is currently implemented using stronger `__NV_ATOMIC_ACQUIRE` memory order.
- `__NV_THREAD_SCOPE_THREAD` thread scope is currently implemented using wider `__NV_THREAD_SCOPE_BLOCK` thread scope.

Example:

```cuda
__device__ T __nv_atomic_load_n(T*  pointer,
                                int memory_order,
                                int thread_scope = __NV_THREAD_SCOPE_SYSTEM);
```

Atomic built-in functions have the following restrictions:

- They can only be used in device functions.
- They cannot operate on local memory.
- The addresses of these functions cannot be taken.
- The `order` and `scope` arguments must be integer literals; they cannot be variables.
- The thread scope `__NV_THREAD_SCOPE_CLUSTER` is supported on architectures `sm_90` and higher.

Example of unsupported cases:

```cuda
 // Not permitted in a host function
 __host__ void bar() {
     unsigned u1 = 1, u2 = 2;
     __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
 }

 // Not permitted to be applied to local memory
__device__ void foo() {
   unsigned a = 1, b;
   __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

 // Not permitted as a template default argument.
 // The function address cannot be taken.
 template<void *F = __nv_atomic_load_n>
 class X {
     void *f = F; // The function address cannot be taken.
 };

 // Not permitted to be called in a constructor initialization list.
 class Y {
     int a;
 public:
     __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
 };
```

#### [5.4.5.2.1. __nv_atomic_fetch_add(), __nv_atomic_add()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-add-nv-atomic-add)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-add-nv-atomic-add "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old + val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_add` returns the `old` value.
- `__nv_atomic_add` has no return value.

The functions support the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`, `double`.

#### [5.4.5.2.2. __nv_atomic_fetch_sub(), __nv_atomic_sub()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-sub-nv-atomic-sub)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-sub-nv-atomic-sub "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old - val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_sub` returns the `old` value.
- `__nv_atomic_sub` has no return value.

The functions support the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`, `double`.

#### [5.4.5.2.3. __nv_atomic_fetch_and(), __nv_atomic_and()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-and-nv-atomic-and)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-and-nv-atomic-and "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old & val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_and` returns the `old` value.
- `__nv_atomic_and` has no return value.

The functions support the following data types:

- Any integral type of size 4 or 8 bytes.

#### [5.4.5.2.4. __nv_atomic_fetch_or(), __nv_atomic_or()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-or-nv-atomic-or)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-or-nv-atomic-or "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old | val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_or` returns the `old` value.
- `__nv_atomic_or` has no return value.

The functions support the following data types:

- Any integral type of size 4 or 8 bytes.

#### [5.4.5.2.5. __nv_atomic_fetch_xor(), __nv_atomic_xor()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-xor-nv-atomic-xor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-xor-nv-atomic-xor "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old ^ val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_xor` returns the `old` value.
- `__nv_atomic_xor` has no return value.

The functions support the following data types:

- Any integral type of size 4 or 8 bytes.

#### [5.4.5.2.6. __nv_atomic_fetch_min(), __nv_atomic_min()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-min-nv-atomic-min)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-min-nv-atomic-min "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the minimum of `old` and `val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_min` returns the `old` value.
- `__nv_atomic_min` has no return value.

The functions support the following data types:

- `unsigned`, `int`, `unsigned long long`, `long long`.

#### [5.4.5.2.7. __nv_atomic_fetch_max(), __nv_atomic_max()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-fetch-max-nv-atomic-max)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-fetch-max-nv-atomic-max "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the maximum of `old` and `val`.
3. Stores the result back to memory at the same address.

- `__nv_atomic_fetch_max` returns the `old` value.
- `__nv_atomic_max` has no return value.

The functions support the following data types:

- `unsigned`, `int`, `unsigned long long`, `long long`

#### [5.4.5.2.8. __nv_atomic_exchange(), __nv_atomic_exchange_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-exchange-nv-atomic-exchange-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-exchange-nv-atomic-exchange-n "Permalink to this headline")

```cuda
__device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_exchange_n` stores `val` to where `address` points to.
| `__nv_atomic_exchange` stores `old` to where `ret` points to and stores the value located at the address `val` to where `address` points to.

- `__nv_atomic_exchange_n` returns the `old` value.
- `__nv_atomic_exchange` has no return value.

The functions support the following data types:

- Any data type of size of 4, 8 or 16 bytes.
- The 16-byte data type is supported on devices of compute capability 9.x and higher.

#### [5.4.5.2.9. __nv_atomic_compare_exchange(), __nv_atomic_compare_exchange_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-compare-exchange-nv-atomic-compare-exchange-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-compare-exchange-nv-atomic-compare-exchange-n "Permalink to this headline")

```cuda
__device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);

__device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Compare `old` with the value where `expected` points to.
3. If they are equal, the return value is `true` and `desired` is stored to where `address` points to. Otherwise, it returns `false` and `old` is stored to where `expected` points to.

The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

The functions support the following data types:

- Any data type of size of 2, 4, 8 or 16 bytes.
- The 16-byte data type is supported on devices with compute capability 9.x and higher.

#### [5.4.5.2.10. __nv_atomic_load(), __nv_atomic_load_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-load-nv-atomic-load-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-load-nv-atomic-load-n "Permalink to this headline")

```cuda
__device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_load` stores `old` to where `ret` points to.
| `__nv_atomic_load_n` returns `old`.

The functions support the following data types:

- Any data type of size 1, 2, 4, 8 or 16 bytes.

`order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.

#### [5.4.5.2.11. __nv_atomic_store(), __nv_atomic_store_n()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-store-nv-atomic-store-n)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-store-nv-atomic-store-n "Permalink to this headline")

```cuda
__device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. | `__nv_atomic_store`  reads the value where `val` points to and stores to where `address` points to.
| `__nv_atomic_store_n` stores `val` to where `address` points to.

`order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.

#### [5.4.5.2.12. __nv_atomic_thread_fence()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#nv-atomic-thread-fence)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#nv-atomic-thread-fence "Permalink to this headline")

```cuda
__device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. The thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.
