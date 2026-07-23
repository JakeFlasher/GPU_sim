---
title: "4.1. PTX-Compiler Handle"
section: "4.1"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#ptx-compiler-handle"
---

## [4.1. PTX-Compiler Handle](https://docs.nvidia.com/cuda/ptx-compiler-api#ptx-compiler-handle)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#ptx-compiler-handle "Permalink to this headline")

**Typedefs**

**[nvPTXCompilerHandle](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__handle_1ga78e3bba9412ef3e505a9a908089c883e)**

  nvPTXCompilerHandle represents a handle to the PTX Compiler.

### [4.1.1. Typedefs](https://docs.nvidia.com/cuda/ptx-compiler-api#typedefs)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#typedefs "Permalink to this headline")

```
`typedef` `struct` `nvPTXCompiler` `*``nvPTXCompilerHandle`
```

nvPTXCompilerHandle represents a handle to the PTX Compiler.

To compile a PTX program string, an instance of nvPTXCompiler must be created and the handle to it must be obtained using the API [nvPTXCompilerCreate()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga86b0c1159bfd34f0fdf632101f30ca26). Then the compilation can be done using the API [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38).
