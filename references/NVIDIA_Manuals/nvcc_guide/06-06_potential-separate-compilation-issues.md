---
title: "6.6. Potential Separate Compilation Issues"
section: "6.6"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#potential-separate-compilation-issues"
---

## [6.6. Potential Separate Compilation Issues](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#potential-separate-compilation-issues)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#potential-separate-compilation-issues "Permalink to this headline")

### [6.6.1. Object Compatibility](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#object-compatibility)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#object-compatibility "Permalink to this headline")

Only relocatable device code with the same ABI version, link-compatible SM target architecture, and same pointer size (32 or 64) can be linked together. The toolkit version of the linker must be >= the toolkit version of the objects. Incompatible objects will produce a link error. Link-compatible SM architectures are ones that have compatible SASS binaries that can combine without translating, e.g. sm_86 and sm_80. An object could have been compiled for a different architecture but also have PTX available, in which case the device linker will JIT the PTX to cubin for the desired architecture and then link.

If Link Time Optimization is used with `-dlto`, the intermediate LTOIR is only guaranteed to be compatible within a major release (e.g. can link together 12.0 and 12.1 LTO intermediates, but not 12.1 and 11.6).

If a kernel is limited to a certain number of registers with the `launch_bounds` attribute or the `--maxrregcount` option, then all functions that the kernel calls must not use more than that number of registers; if they exceed the limit, then a link error will be given.

### [6.6.2. JIT Linking Support](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#jit-linking-support)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#jit-linking-support "Permalink to this headline")

JIT linking means doing an implicit relink of the code at load time. If the cubin does not match the target architecture at load time, the driver re-invokes the device linker to generate cubin for the target architecture, by first JIT’ing the PTX for each object to the appropriate cubin, and then linking together the new cubin. If PTX or cubin for the target architecture is not found for an object, then the link will fail. Implicit JIT linking of the LTO intermediates is not supported at this time, although they can be explicitly linked with the `nvJitLink` library.

### [6.6.3. Implicit CUDA Host Code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#implicit-cuda-host-code)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#implicit-cuda-host-code "Permalink to this headline")

A file like `b.cu` above only contains CUDA device code, so one might think that the b.o object doesn’t need to be passed to the host linker. But actually there is implicit host code generated whenever a device symbol can be accessed from the host side, either via a launch or an API call like `cudaGetSymbolAddress()`. This implicit host code is put into `b.o`, and needs to be passed to the host linker. Plus, for JIT linking to work all device code must be passed to the host linker, else the host executable will not contain device code needed for the JIT link. So a general rule is that the device linker and host linker must see the same host object files (if the object files have any device references in them—if a file is pure host then the device linker doesn’t need to see it). If an object file containing device code is not passed to the host linker, then you will see an error message about the function `__cudaRegisterLinkedBinary_name` calling an undefined or unresolved symbol `__fatbinwrap_name`.

### [6.6.4. Using __CUDA_ARCH__](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#using-cuda-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-cuda-arch "Permalink to this headline")

In separate compilation, `__CUDA_ARCH__` must not be used in headers such that different objects could contain different behavior. Or, it must be guaranteed that all objects will compile for the same compute_arch. If a weak function or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then the instances of that function in the objects could conflict if the objects are compiled for different compute arch. For example, if an a.h contains:

```c++
template<typename T>
__device__ T* getptr(void)
{
#if __CUDA_ARCH__ == 800
  return NULL; /* no address */
#else
  __shared__ T arr[256];
  return arr;
#endif
}
```

Then if a.cu and b.cu both include a.h and instantiate `getptr` for the same type, and b.cu expects a non-NULL address, and compile with:

```text
nvcc --gpu-architecture=compute_80 --device-c a.cu
nvcc --gpu-architecture=compute_86 --device-c b.cu
nvcc --gpu-architecture=sm_86 a.o b.o
```

At link time only one version of the getptr is used, so the behavior would depend on which version is picked. To avoid this, either a.cu and b.cu must be compiled for the same compute arch, or `__CUDA_ARCH__` should not be used in the shared header function.

### [6.6.5. Device Code in Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-code-in-libraries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-code-in-libraries "Permalink to this headline")

If a device function with non-weak external linkage is defined in a library as well as a non-library object (or another library), the device linker will complain about the multiple definitions (this differs from traditional host linkers that may ignore the function definition from the library object, if it was already found in an earlier object).
