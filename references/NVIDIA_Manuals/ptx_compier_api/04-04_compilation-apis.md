---
title: "4.4. Compilation APIs"
section: "4.4"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-apis"
---

## [4.4. Compilation APIs](https://docs.nvidia.com/cuda/ptx-compiler-api#compilation-apis)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-apis "Permalink to this headline")

**Functions**

**nvPTXCompileResult [nvPTXCompilerCompile](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38)(nvPTXCompilerHandle compiler, int numCompileOptions, const char *const *compileOptions)**

  Compile a PTX program with the given compiler options.

**nvPTXCompileResult [nvPTXCompilerCreate](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga86b0c1159bfd34f0fdf632101f30ca26)(nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char *ptxCode)**

  Obtains the handle to an instance of the PTX compiler initialized with the given PTX program `ptxCode` .

**nvPTXCompileResult [nvPTXCompilerDestroy](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gabe1fd08fafc16166b553aa1f2ac132f4)(nvPTXCompilerHandle *compiler)**

  Destroys and cleans the already created PTX compiler.

**nvPTXCompileResult [nvPTXCompilerGetCompiledProgram](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga8f56c9115a219cb518d15ea8e21cd633)(nvPTXCompilerHandle compiler, void *binaryImage)**

  Obtains the image of the compiled program.

**nvPTXCompileResult [nvPTXCompilerGetCompiledProgramSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga026bf92499c9c0e362b4b899b1550f6a)(nvPTXCompilerHandle compiler, size_t *binaryImageSize)**

  Obtains the size of the image of the compiled program.

**nvPTXCompileResult [nvPTXCompilerGetErrorLog](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gaaf88564e76bf5314a597296ba1397942)(nvPTXCompilerHandle compiler, char *errorLog)**

  Query the error message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetErrorLogSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga197d976f96d939b6da998bd10c8b591f)(nvPTXCompilerHandle compiler, size_t *errorLogSize)**

  Query the size of the error message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetInfoLog](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gae35e6f03a99742a9c008c1a03b6dc35f)(nvPTXCompilerHandle compiler, char *infoLog)**

  Query the information message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetInfoLogSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gabff992a534a0d8fde596bfbe8da08449)(nvPTXCompilerHandle compiler, size_t *infoLogSize)**

  Query the size of the information message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerSetFlowCallback](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gae9e9942cf8b8363353cea54a9277e667)(nvPTXCompilerHandle compiler, int(*callback)(void *, void *), void *payload)**

  Register a callback function that the compiler will invoke at different phases of PTX Compilation during a call to [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) .

### [4.4.1. Functions](https://docs.nvidia.com/cuda/ptx-compiler-api#id1)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#id1 "Permalink to this headline")

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerCompile`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `int` `numCompileOptions`, `const` `char` `*``const` `*``compileOptions`)
```

Compile a PTX program with the given compiler options.

> **Note**
>
> &#8212;gpu-name (-arch) is a mandatory option.

**Parameters**
: - **compiler** – **[inout]** A handle to PTX compiler initialized with the PTX program which is to be compiled. The compiled program can be accessed using the handle
- **numCompileOptions** – **[in]** Length of the array `compileOptions`
- **compileOptions** – **[in]** Compiler options with which compilation should be done. The compiler options string is a null terminated character array. A valid list of compiler options is at [link](http://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compile-options).

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILATION_FAILURE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerCreate`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `*``compiler`, `size_t` `ptxCodeLen`, `const` `char` `*``ptxCode`)
```

Obtains the handle to an instance of the PTX compiler initialized with the given PTX program `ptxCode`.

**Parameters**
: - **compiler** – **[out]** Returns a handle to PTX compiler initialized with the PTX program `ptxCode`
- **ptxCodeLen** – **[in]** Size of the PTX program `ptxCode` passed as string
- **ptxCode** – **[in]** The PTX program which is to be compiled passed as string.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerDestroy`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `*``compiler`)
```

Destroys and cleans the already created PTX compiler.

**Parameters**
: **compiler** – **[in]** A handle to the PTX compiler which is to be destroyed

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetCompiledProgram`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `void` `*``binaryImage`)
```

Obtains the image of the compiled program.

> **Note**
>
> [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) API should be invoked for the handle before calling this API. Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **binaryImage** – **[out]** The image of the compiled program. Client should allocate memory for `binaryImage`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetCompiledProgramSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``binaryImageSize`)
```

Obtains the size of the image of the compiled program.

> **Note**
>
> [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) API should be invoked for the handle before calling this API. Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **binaryImageSize** – **[out]** The size of the image of the compiled program

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetErrorLog`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `char` `*``errorLog`)
```

Query the error message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **errorLog** – **[out]** The error log which was produced in previous call to nvPTXCompilerCompiler(). Clients should allocate memory for `errorLog`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetErrorLogSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``errorLogSize`)
```

Query the size of the error message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **errorLogSize** – **[out]** The size of the error log in bytes which was produced in previous call to nvPTXCompilerCompiler().

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetInfoLog`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `char` `*``infoLog`)
```

Query the information message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **infoLog** – **[out]** The information log which was produced in previous call to nvPTXCompilerCompiler(). Clients should allocate memory for `infoLog`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetInfoLogSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``infoLogSize`)
```

Query the size of the information message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **infoLogSize** – **[out]** The size of the information log in bytes which was produced in previous call to nvPTXCompilerCompiler().

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerSetFlowCallback`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `int` `(``*``callback``)``(``void``*``,` `void``*``)`, `void` `*``payload`)
```

Register a callback function that the compiler will invoke at different phases of PTX Compilation during a call to [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38).

The callback function decides to cancel the compilation by returning specific values.

Callback function must satisfy the following constraints

(1) Its signature should be

```c++
int callback(void* param1, void* param2);
```

 When invoking the callback, the compiler will always pass `payload` to param1 so that the callback may make decisions based on `payload` . It’ll always pass NULL to param2 for now which is reserved for future extensions.
(2) It must return 1 to cancel compilation or 0 to continue. Other return values are reserved for future use.

(3) It must return consistent values. Once it returns 1 at one point, it must return 1 in all following invocations during the current nvPTXCompilerCompile call in progress.

(4) It must be thread-safe.

(5) It must not invoke any nvrtc/libnvvm/ptx APIs.

**Parameters**
: - **compiler** – **[in]** A handle to an initialized PTX compiler in which to introduce the callback.
- **callback** – **[in]** Function pointer to the callback function.
- **payload** – **[in]** payload to be passed as a parameter when invoking the callback.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01) [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
