---
title: "Compilation and Export"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/compilation.html#compilation-and-export"
---

# [Compilation and Export](https://docs.nvidia.com/cuda/cutile-python#compilation-and-export)[](https://docs.nvidia.com/cuda/cutile-python/#compilation-and-export "Permalink to this headline")

When a kernel function marked with [`@ct.kernel`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel "cuda.tile.kernel") is launched
using [`ct.launch()`](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.launch "cuda.tile.launch"), it is specialized and compiled just in time (JIT)
for the concrete launch arguments. It is also possible to compile a kernel ahead of time (AOT)
and export it as a CUDA binary (cubin) file, or as a TileIR bytecode file.

While just-in-time compilation provides the convenience of automatic kernel specialization,
ahead-of-time compilation requires the user to precisely describe the arguments for which the kernel
is being compiled, including their types and additional constraints (assumptions) imposed on their
values.

The main API entry point for ahead-of-time compilation
is [`cuda.tile.compilation.export_kernel()`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.export_kernel "cuda.tile.compilation.export_kernel"):

```
`cuda.tile.compilation.``export_kernel`(
```

Compile and export a kernel.

**Parameters:**
: - **kernel** ([_cuda.tile.kernel_](https://docs.nvidia.com/cuda/cutile-python/execution.html#cuda.tile.kernel "cuda.tile.kernel")) – A kernel function to export.
- **signatures** (_Sequence__[_[_cuda.tile.compilation.KernelSignature_](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.compilation.KernelSignature "cuda.tile.compilation.KernelSignature")_]_) – A non-empty list of signatures for which to compile the kernel.
- **output_file** (_IO__|__str__|__bytes__|__os.PathLike_) – Either a filename or a binary file-like object to write the output to.
To save the result in memory, you can pass an instance of the *io.BytesIO*
standard library class.
- **gpu_code** (_str_) – Name of the target GPU for which to compile the kernel (e.g., *“sm_100”*)
- **output_format** (_str_) – Set to “cubin” to export a CUDA binary file, or “tileir_bytecode”
to export a TileIR bytecode file.
- **bytecode_version** (_str__|__None_) – Set to *None* to automatically detect the latest TileIR bytecode version supported
by the compiler (default). Otherwise, it must be a string of the form “major.minor”
that specifies the version of the TileIR bytecode to use (e.g., “13.1”).
