---
title: "5.3.9.4. Variables"
section: "5.3.9.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#variables"
---

### [5.3.9.4. Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#variables "Permalink to this headline")

#### [5.3.9.4.1. Local Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#local-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#local-variables "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__`, and `__constant__` memory space specifiers are not allowed on  non-`extern` variable declarations within a function that executes on the host.

Examples:

```cuda
__host__ void host_function() {
    int x;                   // CORRECT, __host__ variable
    __device__   int y;      // ERROR,   __device__ variable declaration within a host function
    __shared__   int z;      // ERROR,   __shared__ variable declaration within a host function
    __managed__  int w;      // ERROR,   __managed__ variable  declaration within a host function
    __constant__ int h;      // ERROR,   __constant__ variable declaration within a host function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

The `__device__`, `__constant__`, and `__managed__` memory space specifiers are not allowed on variable declarations that are neither `extern` nor `static` within a function that executes on the device.

```cuda
__device__ void device_function() {
    int x;                   // CORRECT, __device__ variable
    __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
    __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

see also the [static variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#static-variables) section.

#### [5.3.9.4.2. const-qualified Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#const-qualified-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#const-qualified-variables "Permalink to this headline")

A `const`-qualified variable without memory space annotations (`__device__` or `__constant__`) declared at global, namespace, or class scope is considered to be a host variable. Device code cannot contain a reference or take the address of the variable.

The variable may be directly used in device code, if

- it has been initialized with a constant expression before the point of use,
- the type is not `volatile`-qualified, and
- it has one of the following types:
  - built-in integral type, or
  - built-in floating point type, except when the host compiler is Microsoft Visual Studio.

Starting with C++14, it is recommended to use `constexpr` or `inline constexpr` (C++17) variables instead of `const`-qualified ones. `constexpr` variables are not subject to the same type restrictions and can be utilized directly in device code.

`__managed__` variables don’t support `const`-qualified types.

Examples:

```cuda
const            int   ConstVar          = 10;
const            float ConstFloatVar     = 5.0f;
inline constexpr float ConstexprFloatVar = 5.0f; // C++17

struct MyStruct {
    static const            int   ConstVar          = 20;
//  static const             float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
    static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
};

extern const int ExternVar;

__device__ void foo() {
    int array1[ConstVar];                     // CORRECT
    int array2[MyStruct::ConstVar];           // CORRECT

    const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
    constexpr float var2 = ConstexprFloatVar; // CORRECT
//  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
//  int&            var4 = ConstVar;           // ERROR, reference to host variable
//  int*            var5 = &ConstVar;          // ERROR, address of host variable
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/eWG8KxK94).

#### [5.3.9.4.3. volatile-qualified Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#volatile-qualified-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#volatile-qualified-variables "Permalink to this headline")

> **Note**
>
> The `volatile` keyword is supported to maintain compatibility with ISO C++. However, few, if any, of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reading and writing to `volatile`-qualified objects are not atomic and are compiled into one or more [volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) that do not guarantee:

- ordering of memory operations, or
- that the number of memory operations performed by the hardware matches the number of PTX instructions.

CUDA C++ `volatile` is NOT suitable for:

- **Inter-Thread Synchronization**: Use atomic operations via [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions) instead.

Atomic memory operations provide inter-thread synchronization guarantees and deliver better performance than `volatile` operations.
However, CUDA C++ `volatile` operations do not provide any inter-thread synchronization guarantees and are therefore not suitable for this purpose.
The following example shows how to pass a message between two threads using atomic operations.

**cuda::atomic_ref**
| ```cuda #include <cuda/atomic>  __global__ void kernel(int* flag, int* data) {     cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref{*flag};     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomic_ref.load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         atomic_ref.store(1, cuda::memory_order_release);     } } ``` |
| --- |

**cuda::atomic**
| ```cuda #include <cuda/atomic>  __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(flag->load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         flag->store(1, cuda::memory_order_release);     } } ``` |
| --- |

**Atomic Functions (atomicAdd and atomicExch)**
| ```cuda __global__ void kernel(int* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomicAdd(flag, 0) == 0)             ;                // Load with Relaxed Read-Modify-Write         __threadfence();     // SequentiallyConsistent fence         if (*data != 42)             __trap();        // Errors if wrong data read     } else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         __threadfence();     // SequentiallyConsistent fence         atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write     } } ``` |
| --- |
- **Memory Mapped IO** (MMIO): Use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.

PTX MMIO operations strictly preserve the number of memory accesses performed.
However, CUDA C++ `volatile` operations do not preserve the number of memory accesses performed and may perform more or fewer accesses than requested in an undetermined way. This makes them unsuitable for MMIO.
The following example shows how to read from and write to a register using PTX MMIO operations.

```cuda
__global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
    // Write to MMIO register:
    int value = 13;
    asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
        :
        : "l"(mmio_reg0), "r"(value) : "memory");
    // Read MMIO register:
    asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
        : "=r"(value)
        : "l"(mmio_reg1) : "memory");
    if (value != 42)
        __trap(); // Errors if wrong data read
}
```

#### [5.3.9.4.4. static Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#static-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#static-variables "Permalink to this headline")

`static` variables are allowed in device code in the following cases:

- Within `__global__` or `__device__`-only functions.
- Within `__host__ __device__` functions:
  - `static` variables without an explicit memory space (automatic deduction).
  - `static` variables with an explicit memory space, such as `static __device__/__constant__/__shared__/__managed__`, are allowed only when `__CUDA_ARCH__` is defined.

A `static` variable within a `__host__ __device__` function holds a different value depending on the execution space.

Examples of legal and illegal uses of function-scope `static` variables are shown below.

```cuda
struct TrivialStruct {
    int x;
};

struct NonTrivialStruct {
    __device__ NonTrivialStruct(int x) {}
};

__device__ void device_function(int x) {
    static int v1;              // CORRECT, implicit __device__ memory space specifier
    static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
//  static int v3 = x;           // ERROR, dynamic initialization is not allowed

    static __managed__  int v4; // CORRECT, explicit
    static __device__   int v5; // CORRECT, explicit
    static __constant__ int v6; // CORRECT, explicit
    static __shared__   int v7; // CORRECT, explicit

    static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
    static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
//  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
//  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/TdYKaTq3f).

---

```cuda
__host__ __device__ void host_device_function() {
    static            int v1; // CORRECT, implicit __device__ memory space specifier
//  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
#ifdef __CUDA_ARCH__
    static __device__ int v3; // CORRECT, declaration is only visible during device compilation
#else
    static int v4;            // CORRECT, declaration is only visible during host compilation
#endif
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/18qhjn8P1).

---

```cuda
#include <cassert>

__host__ __device__ int host_device_function() {
    static int v = 0;
    v++;
    return v;
}

__global__ void kernel() {
    int ret = host_device_function(); // v = 1
    assert(ret == 4);                 // FAIL
}

int main() {
    host_device_function();           // v = 1
    host_device_function();           // v = 2
    int ret = host_device_function(); // v = 3
    assert(ret == 3);                 // OK
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/Wqo9WjvYY).

#### [5.3.9.4.5. extern Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#extern-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#extern-variables "Permalink to this headline")

When compiling in the [whole program compilation mode](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation), `__device__`, `__shared__`, `__managed__`, and `__constant__` variables cannot be defined with external linkage using the `extern` keyword.

The only exception is for dynamically allocated `__shared__` variables as described in the [Dynamic Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-dynamic-allocation-shared-memory) section.

```cuda
__device__        int x; // OK
extern __device__ int y; // ERROR in whole program compilation mode
extern __shared__ int z; // OK
```
