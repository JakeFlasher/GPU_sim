---
name: compiling-cuda-artifacts
description: "Resolve CUDA compilation, PTX/cubin/fatbin targets, nvcc flags, device linking, JIT and build failures. Use the selected architecture rather than a stale host default."
---

# CUDA build artifacts

For this repository's producer, use `tools/nvbit_tracer/README.md` and
`tools/nvbit_tracer/pinned_build_env.sh`. The local helper checks CUDA 13.2,
GCC 14 and protobuf 34; ambient `NVCC_CCBIN` can select an incompatible compiler.
Do not provision or upgrade toolchains just to reorganize docs.

Compilation and virtual/real target semantics live under
`references/NVIDIA_Manuals/nvcc_guide/`; extraction/disassembly under
`references/NVIDIA_Manuals/cuda_binary_utility/`. Optional knowledge cards may
be absent; use these tracked references and verify version-sensitive flags
against the installed tool or current NVIDIA documentation.

For an SM86 question, start with `-gencode arch=compute_86,code=sm_86`.
PTX fallback is a deliberate additional build target, not equivalent native
code. `-Xptxas=-v` exposes allocation/spills; `cuobjdump --dump-sass` inspects
what was emitted. Select other architectures only for a question about them.
Keep static artifact identity and the traced launch owner separate from a
same-named executable guess. Capture commands are in `docs/runbook_host.md`.
