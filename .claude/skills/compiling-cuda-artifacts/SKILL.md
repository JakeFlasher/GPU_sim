---
name: compiling-cuda-artifacts
description: Resolves nvcc compilation, fatbin/cubin/PTX trajectory, and separate-compilation questions in NVIDIA_Sims_Predict_NCU. Use when prompts mention nvcc flags (-arch, -gencode, -code, -Xptxas, -keep, -lineinfo, -dc, -dlto, -rdc, -maxrregcount), virtual-vs-real architectures, fatbin layout, JIT compilation from PTX, JIT cache (CUDA_CACHE_DISABLE), -gencode=arch=compute_X,code=sm_Y recipes, sm_120 / sm_120a opt-in, sm_100a vs sm_100 architecture-conditional features, ptxas verbose output, register-spill diagnosis at compile time, separate compilation, device code linking, fatbin extraction with nvprune/cuobjdump, or "kernel does not run on my GPU" forward-compat questions.
---

# Compiling CUDA Artifacts

This skill owns nvcc / fatbin / PTX-trajectory questions. Two distilled cards.

## Cards

| Topic | Card |
|---|---|
| nvcc flags in practice (-arch, -gencode, -Xptxas, -keep, -lineinfo, -dlto, -rdc, -maxrregcount) | `.claude/knowledge/nvcc/nvcc-flags.md` |
| Compilation trajectory (.cu → PTX → cubin → fatbin; virtual vs real arch) | `.claude/knowledge/nvcc/nvcc-trajectory.md` |

Plus, for Blackwell-specific compatibility:

| Topic | Card |
|---|---|
| Cubin vs PTX survival / -gencode recipes / Pascal-scheduling JIT opt-in | `.claude/knowledge/blackwell/blackwell-compatibility.md` |

## Compilation trajectory

```
.cu source
  └── nvcc front-end (cudafe++) ─── splits host vs device
                                    ↓
                           device PTX (text)
                                    ↓
                           ptxas → cubin (binary, per-arch)
                                    ↓
                           fatbinary → fatbin (multi-arch container)
                                    ↓
                           host link (g++ / clang)
                                    ↓
                           final ELF
```

JIT happens at runtime when the fatbin contains PTX for an architecture but no matching cubin — the driver invokes `ptxas` on the embedded PTX targeted at the actual GPU. `CUDA_CACHE_DISABLE=1` disables the JIT cache (useful for repro work).

## -arch / -gencode cheat sheet

| Goal | Flag |
|---|---|
| Compile for one specific real arch (no fallback) | `-arch=sm_120` |
| Compile for one virtual arch + emit PTX (JITs everywhere) | `-arch=compute_120` |
| Multi-arch fatbin with cubins for sm_90/sm_100/sm_120 + PTX fallback | `-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120` |
| Architecture-conditional Blackwell features (FP4 NVFP4) | `-gencode=arch=compute_120a,code=sm_120a` |
| Architecture-conditional B200 features | `-gencode=arch=compute_100a,code=sm_100a` |
| Pascal-scheduling JIT opt-in on Blackwell | `-gencode=arch=compute_60,code=sm_100` (per `blackwell-compatibility.md` §1.4.3) |

`compute_NN` is **virtual** (PTX); `sm_NN` is **real** (cubin). `compute_NNa` / `sm_NNa` adds architecture-conditional features (FP4 block-scaled MMA on Blackwell, etc.).

## ptxas diagnostics

```bash
nvcc -Xptxas=-v -arch=sm_120 my_kernel.cu -o my_kernel
```

Output reports:
- Register count per kernel (`uses N registers`)
- Spill stores / loads in bytes
- Stack frame size
- Shared memory used
- Constant memory used

Watch for `register spill` — first lever is `-maxrregcount=255` or per-kernel `__launch_bounds__`. Don't immediately drop block size; Blackwell SM120's 48-warp cap restricts that axis already (see `tuning-blackwell-sm120-kernels`).

## Separate compilation (`-rdc=true`)

```bash
nvcc -dc -rdc=true a.cu -o a.o
nvcc -dc -rdc=true b.cu -o b.o
nvcc -dlink -rdc=true a.o b.o -o linked.o
g++ a.o b.o linked.o -lcudart -o app
```

Or `-dlto` for device-side LTO (CUDA 11.2+).

## fatbin inspection

```bash
cuobjdump --list-text my_app                    # show embedded PTX/cubin
cuobjdump --extract-ptx all my_app              # extract PTX
cuobjdump --extract-elf all my_app              # extract cubins
nvprune --target-gpu sm_120 in.fatbin -o out.fatbin   # strip non-target
```

## When the kernel does not run on my GPU

1. Check `nvidia-smi` (read-only, allowed) for actual CC.
2. `cuobjdump --list-text` the binary to see what archs were compiled.
3. If actual CC has no matching cubin AND no PTX fallback, the kernel will fail at launch.
4. Fix: add `-gencode=arch=compute_X,code=sm_X` for the missing target, OR add `-gencode=arch=compute_X,code=compute_X` to ship PTX-only fallback.

See `blackwell-compatibility.md` for the Blackwell-specific compat matrix.

## Cross-references

- Per-instruction PTX semantics → `interpreting-ptx-isa`.
- SASS-level realization of compiled kernels → `analyzing-sass-nvbit-traces`.
- Tuning launch configuration that drives compilation flags → `optimizing-cuda-kernels`.
- Blackwell-specific compatibility (Pascal-style JIT, sm_120a) → `tuning-blackwell-sm120-kernels`.

## Hard rules

Compilation guidance is source-level/read-only; follow root `CLAUDE.md` for
privilege, evidence, and architecture-local contracts. The toolchain is built
at `tools/nvbit_tracer/` for the NVBit tracer specifically.

## Cite consulted files

End every reply with `## Knowledge Consulted`.
