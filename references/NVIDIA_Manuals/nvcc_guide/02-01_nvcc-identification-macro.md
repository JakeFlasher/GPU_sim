---
title: "2.1. NVCC Identification Macro"
section: "2.1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-identification-macro"
---

## [2.1. NVCC Identification Macro](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-identification-macro)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-identification-macro "Permalink to this headline")

`nvcc` predefines the following macros:

**`__NVCC__`**

  Defined when compiling C/C++/CUDA source files.

**`__CUDACC__`**

  Defined when compiling CUDA source files.

**`__CUDACC_RDC__`**

  Defined when compiling CUDA source files in relocatable device code mode (see [NVCC Options for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-options-for-separate-compilation)).

**`__CUDACC_EWP__`**

  Defined when compiling CUDA source files in extensible whole program mode (see [Options for Specifying Behavior of Compiler/Linker](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-altering-compiler-linker-behavior)).

**`__CUDACC_DEBUG__`**

  Defined when compiling CUDA source files in the device-debug mode (see [Options for Specifying Behavior of Compiler/Linker](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-altering-compiler-linker-behavior)).

**`__CUDACC_RELAXED_CONSTEXPR__`**

  Defined when the `--expt-relaxed-constexpr` flag is specified on the command line. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.

**`__CUDACC_EXTENDED_LAMBDA__`**

  Defined when the `--expt-extended-lambda` or `--extended-lambda` flag is specified on the command line. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.

**`__CUDACC_VER_MAJOR__`**

  Defined with the major version number of `nvcc`.

**`__CUDACC_VER_MINOR__`**

  Defined with the minor version number of `nvcc`.

**`__CUDACC_VER_BUILD__`**

  Defined with the build version number of `nvcc`.

**`__NVCC_DIAG_PRAGMA_SUPPORT__`**

  Defined when the CUDA frontend compiler supports diagnostic control with the `nv_diag_suppress`, `nv_diag_error`, `nv_diag_warning`, `nv_diag_default`, `nv_diag_once`, and `nv_diagnostic` pragmas.

**`__CUDACC_DEVICE_ATOMIC_BUILTINS__`**

  Defined when the CUDA frontend compiler supports device atomic compiler builtins. Refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for more details.
