---
title: "4.2. Error codes"
section: "4.2"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#error-codes"
---

## [4.2. Error codes](https://docs.nvidia.com/cuda/ptx-compiler-api#error-codes)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#error-codes "Permalink to this headline")

**Enumerations**

**[nvPTXCompileResult](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)**

  The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result.

### [4.2.1. Enumerations](https://docs.nvidia.com/cuda/ptx-compiler-api#enumerations)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#enumerations "Permalink to this headline")

```
`enum` `nvPTXCompileResult`
```

The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result.

_Values:_

```
`enumerator` `NVPTXCOMPILE_SUCCESS`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INVALID_INPUT`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_COMPILATION_FAILURE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INTERNAL`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_OUT_OF_MEMORY`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_CANCELLED`
```

```
`enumerator` `NVPTXCOMPILE_PARSE_ONLY_SUCCESS`
```
