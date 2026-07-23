---
title: "4.3. API Versioning"
section: "4.3"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#api-versioning"
---

## [4.3. API Versioning](https://docs.nvidia.com/cuda/ptx-compiler-api#api-versioning)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#api-versioning "Permalink to this headline")

The PTX compiler APIs are versioned so that any new features or API changes can be done by bumping up the API version.

**Functions**

**nvPTXCompileResult [nvPTXCompilerGetNumSupportedArchs](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga5bc657a75e33cad967df8da7d1387511)(int *numArchs)**

  Sets the output paramater `numArchs` with the number of architectures supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetNumSupportedVersions](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1gae6aa43abe2b0ab4bd948c58984cf44ae)(int *numVersions)**

  Sets the output paramater `numVersions` with the number of PTX ISA versions supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetSupportedArchs](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga0abce69c5235af0da34fb09171e46c2a)(int *supportedArchs)**

  Populates the array passed via output paramater `supportedArchs` with architectures supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetSupportedVersions](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga916f37c92d682ef18c552e9e767c465a)(int(*versions)[2])**

  Populates the 2D Array passed via output paramater `versions` with PTX ISA Versions major, minor pair supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetVersion](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1gadb846da0f180ce336bf1f050ce5b3e6b)(unsigned int *major, unsigned int *minor)**

  Queries the current `major` and `minor` version of PTX Compiler APIs being used.

### [4.3.1. Functions](https://docs.nvidia.com/cuda/ptx-compiler-api#functions)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#functions "Permalink to this headline")

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetNumSupportedArchs`(`int` `*``numArchs`)
```

Sets the output paramater `numArchs` with the number of architectures supported by PTX Compiler Library.

This can be used to pass an array to nvPTXCompilerGetSupportedArchs to get the supported architectures.

**Parameters**
: **numArchs** – **[out]** Number of supported architectures

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetNumSupportedVersions`(`int` `*``numVersions`)
```

Sets the output paramater `numVersions` with the number of PTX ISA versions supported by PTX Compiler Library.

This can be used to pass a 2D Array to nvPTXCompilerGetSupportedVersions. Number of rows in 2D Array will be equal to value populated in `numVersions`.

**Parameters**
: **numVersions** – **[out]** Number of supported PTX ISA versions

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetSupportedArchs`(`int` `*``supportedArchs`)
```

Populates the array passed via output paramater `supportedArchs` with architectures supported by PTX Compiler Library.

The array is sorted in the ascending order. The size of the array to be passed can be determined using nvPTXCompilerGetNumSupportedArchs.

**Parameters**
: **supportedArchs** – **[out]** Sorted array of supported architectures.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetSupportedVersions`(`int` `(``*``versions``)``[``2``]`)
```

Populates the 2D Array passed via output paramater `versions` with PTX ISA Versions major, minor pair supported by PTX Compiler Library.

The array is sorted in the ascending order. the row size of the 2D array can be determined using nvPTXCompilerGetNumSupportedVersions.

**Parameters**
: **versions** – **[out]** Sorted 2D array with column size 2, having pair of major and minor PTX ISA version supported.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetVersion`(`unsigned` `int` `*``major`, `unsigned` `int` `*``minor`)
```

Queries the current `major` and `minor` version of PTX Compiler APIs being used.

> **Note**
>
> The version of PTX Compiler APIs follows the CUDA Toolkit versioning. The PTX ISA version supported by a PTX Compiler API version is listed [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes).

**Parameters**
: - **major** – **[out]** Major version of the PTX Compiler APIs
- **minor** – **[out]** Minor version of the PTX Compiler APIs

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
