---
title: "5.3.9.5. Functions"
section: "5.3.9.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#functions"
---

### [5.3.9.5. Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#functions "Permalink to this headline")

#### [5.3.9.5.1. Recursion](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#recursion)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#recursion "Permalink to this headline")

`__global__` functions do not support recursion, while `__device__` and `__host__ __device__` functions do not have such restriction.

#### [5.3.9.5.2. External Linkage](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#external-linkage)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#external-linkage "Permalink to this headline")

Device variables or functions with external linkage require [separate compilation mode](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#nvcc-separate-compilation) across multiple translation units.

In separate compilation mode, if a `__device__` or `__global__` function definition is required to exist in a particular translation unit, then the parameters and return types of the function must be complete in that translation unit. The concept is also known as One Definition Rule-use, or ODR-use.

Example:

```cuda
//first.cu:
struct S;                   // forward declaration
__device__ void foo(S);     // ERROR, type 'S' is an incomplete type
__device__ auto* ptr = foo; // ODR-use, address taken

int main() {}
```

```cuda
//second.cu:
struct S {};               // struct definition
__device__ void foo(S) {}  // function definition
```

```console
# compiler invocation
$ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                 first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```

#### [5.3.9.5.3. Formal Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#formal-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#formal-parameters "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on formal parameters.

```cuda
void device_function1(__device__ int x) { } // ERROR, __device__ parameter
void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter
```

#### [5.3.9.5.4. __global__ Function Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#global-function-parameters)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#global-function-parameters "Permalink to this headline")

A `__global__` function has the following restrictions:

- It cannot have a variable number of arguments, namely the C ellipsis syntax `...` and the `va_list` type. C++11 variadic template is allowed, subject to the restrictions described in the [__global__ Variadic Template](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cpp11-variadic-template) section.
- Function parameters are passed to the device via [constant memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#constant-memory) and their total size is limited to 32,764 bytes.
- Function parameters cannot be pass-by-reference or by pass-by-rvalue reference.
- Function parameters cannot be of type `std::initializer_list`.
- Polymorphic class parameters (`virtual`) are considered undefined behavior.
- Lambda expressions and closure types are allowed, subject to the restrictions described in the [Lambda Expressions and __global__ Function Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#lambda-expressions-global) section.

#### [5.3.9.5.5. __global__ Function Arguments Passing](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#global-function-arguments-passing)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#global-function-arguments-passing "Permalink to this headline")

When launching a `__global__` function [from device code](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-kernels), each argument must be trivially copyable and trivially destructible.

When a `__global__` function is launched from host code, each argument type may be non-trivially copyable or non-trivially destructible. However, the processing of these types does not follow the standard C++ model, as described below. The user code must ensure that this workflow does not affect program correctness. The workflow diverges from standard C++ in two areas:

1. **Raw memory copy instead of copy constructor invocation**

The CUDA Runtime passes the kernel arguments to the `__global__` function by copying the raw memory content, eventually using `memcpy`. If an argument is non-trivially copyable and provides a user-defined copy constructor, the operations and side effects of the invocation are skipped in the host-to-device copy.

Example:

```cuda
#include <cassert>
struct MyStruct {
    int  value = 1;
    int* ptr;
    MyStruct() = default;
    __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
};
__global__ void device_function(MyStruct my_struct) {
    // this assert fails because "my_struct" is obtained by copying
    // the raw memory content and the copy constructor is skipped.
    assert(my_struct.ptr == &my_struct.value); // FAIL
}
void host_function(MyStruct my_struct) {
    assert(my_struct.ptr == &my_struct.value); // CORRECT
}
int main() {
    MyStruct my_struct;
    host_function(my_struct);
    device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/xhqe16dec).
2. **Destructor may be invoked before the** `__global__` **function has finished**

Kernel launches are asynchronous with host execution. As a result, if a `__global__` function argument has a non-trivial destructor, the destructor may execute in host code even before the `__global__` function has finished execution. This may break programs where the destructor has side effects.

Example:

```cuda
#include <cassert>
__managed__ int var = 0;
struct MyStruct {
    __host__ __device__ ~MyStruct() { var = 3; }
};
__global__ void device_function(MyStruct my_struct) {
    assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
}
int main() {
    MyStruct my_struct;
    // GPU kernel execution is asynchronous with host execution.
    // As a result, MyStruct::~MyStruct() could be executed before
    // the kernel finishes executing.
    device_function<<<1, 1>>>(my_struct);
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/cn6Y5W6zs).
