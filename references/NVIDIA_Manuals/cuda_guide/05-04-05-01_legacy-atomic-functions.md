---
title: "5.4.5.1. Legacy Atomic Functions"
section: "5.4.5.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#legacy-atomic-functions"
---

### [5.4.5.1. Legacy Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#legacy-atomic-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#legacy-atomic-functions "Permalink to this headline")

Legacy atomic functions perform atomic read-modify-write operations on a 32-, 64-, or 128-bit word stored in global or shared memory. For example, the `atomicAdd()` function reads a word at a specific address in global or shared memory, adds a number to it, and writes the result back to the same address.

- Atomic functions can only be used in device functions.
- For vector types such as `__half2`, `__nv_bfloat162`, `float2`, and `float4`, the read-modify-write operation is performed on each element of the vector. The entire vector is not guaranteed to be atomic in a single access.

The atomic functions described in this section have a [memory ordering](https://en.cppreference.com/w/cpp/atomic/memory_order) of `cuda::std::memory_order_relaxed` and are only atomic at a particular [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes):

- Atomic APIs without a suffix, for example `atomicAdd`, are atomic at scope `cuda::thread_scope_device`.
- Atomic APIs with the `_block` suffix, for example, `atomicAdd_block`, are atomic at scope `cuda::thread_scope_block`.
- Atomic APIs with the `_system` suffix, for example, `atomicAdd_system`, are atomic at scope  `cuda::thread_scope_system` if they meet particular [conditions](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity).

The following example shows the CPU and GPU atomically updating an integer value at address `addr`:

```cuda
#include <cuda_runtime.h>

__global__ void atomicAdd_kernel(int* addr) {
    atomicAdd_system(addr, 10);
}

void test_atomicAdd(int device_id) {
    int* addr;
    cudaMallocManaged(&addr, 4);
    *addr = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    if (deviceProp.concurrentManagedAccess != 1) {
        return; // the device does not coherently access managed memory concurrently with the CPU
    }

    atomicAdd_kernel<<<...>>>(addr);
    __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}
```

---

Note that any atomic operation can be implemented based on `atomicCAS()` (Compare and Swap). For example, `atomicAdd()` for single-precision floating-point numbers can be implemented as follows:

```cuda
#include <cuda/memory>
#include <cuda/std/bit>

__device__ float customAtomicAdd(float* d_ptr, float value) {
    volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
    unsigned  old_value      = *d_ptr_unsigned;
    unsigned  assumed;
    do {
        assumed                          = old_value;
        float    assumed_float           = cuda::std::bit_cast<float>(assumed);
        float    expected_value          = assumed_float + value;
        unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
        old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old_value);
    return cuda::std::bit_cast<float>(old_value);
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/676e5bc7a).

#### [5.4.5.1.1. atomicAdd()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicadd)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicadd "Permalink to this headline")

```cuda
T atomicAdd(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old + val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicAdd()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`, `double`, `__half2`, `__half`.
- `__nv_bfloat16`, `__nv_bfloat162` on devices of compute capability 8.x and higher.
- `float2`, `float4` on devices of compute capability 9.x and higher, and only supported for global memory addresses.

The atomicity of `atomicAdd()` applied to vector types, for example `__half2` or `float4`, is guaranteed separately for each of the components; the entire vector is not guaranteed to be atomic as a single access.

#### [5.4.5.1.2. atomicSub()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicsub)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicsub "Permalink to this headline")

```cuda
T atomicSub(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old - val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicSub()` supports the following data types:

- `int`, `unsigned`

#### [5.4.5.1.3. atomicInc()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicinc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicinc "Permalink to this headline")

```cuda
unsigned atomicInc(unsigned* address, unsigned val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old >= val ? 0 : (old + 1)`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

#### [5.4.5.1.4. atomicDec()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicdec)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicdec "Permalink to this headline")

```cuda
unsigned atomicDec(unsigned* address, unsigned val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `(old == 0 || old > val) ? val : (old - 1)`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

#### [5.4.5.1.5. atomicAnd()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicand)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicand "Permalink to this headline")

```cuda
T atomicAnd(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old & val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicAnd()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`.

#### [5.4.5.1.6. atomicOr()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicor "Permalink to this headline")

```cuda
T atomicOr(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old | val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicOr()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`.

#### [5.4.5.1.7. atomicXor()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicxor)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicxor "Permalink to this headline")

```cuda
T atomicXor(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old ^ val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicXor()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`.

#### [5.4.5.1.8. atomicMin()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicmin)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicmin "Permalink to this headline")

```cuda
T atomicMin(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the minimum of `old` and `val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicMin()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`, `long long`.

#### [5.4.5.1.9. atomicMax()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicmax)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicmax "Permalink to this headline")

```cuda
T atomicMax(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the maximum of `old` and `val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicMax()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`, `long long`.

#### [5.4.5.1.10. atomicExch()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomicexch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomicexch "Permalink to this headline")

```cuda
T atomicExch(T* address, T val);
```

```cuda
template<typename T>
T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Stores `val` back to memory at the same address.

The function returns the `old` value.

`atomicExch()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`, `float`.

The C++ template function `atomicExch()` supports 128-bit types with the following requirements:

- Compute capability 9.x and higher.
- `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.
- `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.
- For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.

#### [5.4.5.1.11. atomicCAS()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#atomiccas)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#atomiccas "Permalink to this headline")

```cuda
T atomicCAS(T* address, T compare, T val);
```

```cuda
template<typename T>
T atomicCAS(T* address, T compare, T val);  // only 128-bit types, compute capability 9.x and higher
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old == compare ? val : old`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicCAS()` supports the following data types:

- `int`, `unsigned`, `unsigned long long`, `unsigned short`.

The C++ template function `atomicCAS()` supports 128-bit types with the following requirements:

- Compute capability 9.x and higher.
- `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.
- `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.
- For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.
