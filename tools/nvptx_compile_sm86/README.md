# `nvptx_compile_sm86`

This small adapter invokes NVIDIA's installed `nvPTXCompiler` API to lower PTX
text to an SM86 cubin without shelling out to `ptxas`.

```bash
g++ -std=c++17 -O2 \
  -I/usr/local/cuda-13.2/targets/x86_64-linux/include \
  tools/nvptx_compile_sm86/nvptx_compile_sm86.cpp \
  /usr/local/cuda-13.2/targets/x86_64-linux/lib/libnvptxcompiler_static.a \
  -ldl -lpthread -o tools/nvptx_compile_sm86/nvptx_compile_sm86

tools/nvptx_compile_sm86/nvptx_compile_sm86 input.ptx output.cubin sm_86
```

The result is a native NVIDIA PTX-to-cubin path. It does not provide a complete
independent SASS/ABI/compiler model or scheduler visibility. Those fields stay
black-box inputs to analytical/statistical performance models.

For an official PTX linker path that emits a linked cubin rather than only the
compiler API result:

```bash
g++ -std=c++17 -O2 \
  -I/usr/local/cuda-13.2/targets/x86_64-linux/include \
  tools/nvptx_compile_sm86/nvjitlink_sm86.cpp \
  -L/usr/local/cuda-13.2/targets/x86_64-linux/lib -lnvJitLink \
  -Wl,-rpath,/usr/local/cuda-13.2/targets/x86_64-linux/lib \
  -o tools/nvptx_compile_sm86/nvjitlink_sm86

tools/nvptx_compile_sm86/nvjitlink_sm86 input.ptx linked.cubin sm_86
```

`nvJitLink` supplies NVIDIA's PTX/cubin linker and relocation path (the
installed library reports version 13.4 even though it is under the local
`cuda-13.2` toolkit path). It still
does not expose a complete SASS encoder, scheduler/control-word generator, or
CUDA ABI implementation; unsupported fields remain black-box model inputs.

The local probe also observed `nvJitLinkAddData` rejecting some valid
`ptxas`-accepted PTX address-arithmetic forms when the input was not
NUL-terminated. The adapter now appends the PTX terminator; the corrected
matrix in `results/alpha/sm86_research_restart_20260929/` links both forms.
This fixes the adapter bug but does not turn the vendor linker into an
independent SASS/ABI implementation; unsupported details remain
black-box model inputs.

`nvjitlink_launch_sm86.cpp` is a small CUDA Driver API execution adapter for
loading a linked cubin and launching its `value_kernel`; it is useful for a
native execution observation, but is deliberately not a correctness test.

`nak_cuda_bridge.py` is a bounded bridge for inserting an independently
generated NAK SM86 body into an existing CUDA ELF template. It preserves the
template's NVIDIA metadata and supports checked constant-buffer offset remaps;
it is not a general linker or ABI generator.
