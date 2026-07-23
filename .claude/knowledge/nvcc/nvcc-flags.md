---
Use when: writing or auditing an `nvcc` invocation in a Makefile / build script; choosing `-arch` / `-gencode`; deciding sm_120 vs sm_120a (architecture-conditional opt-in suffix for FP4 / NVFP4 block-scaled MMA, tcgen05); emitting PTX or SASS for inspection; turning on `-rdc` for separate compilation; debugging "no kernel image" or register-spill issues from compiler output.
Primary raw source: references/NVIDIA_Manuals/nvcc_guide/04_nvcc-command-options.md (specifically §4.2.5 driver behavior, §4.2.7 code generation).
Supporting sources: references/NVIDIA_Manuals/nvcc_guide/05_gpu-compilation.md (virtual vs real arch); references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md (arch survival rules).
Repo touchpoints: workloads/simbench_bw/, tools/nvbit_tracer/, tools/cloud/compile_path_inspector.py.
Out of scope: full nvcc env-var list; the compilation pipeline as a whole (in `nvcc-trajectory.md`); cross-compilation host-side details.
Version family: CUDA 13.x. Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# nvcc Flags — Practical Reference

The raw `04_nvcc-command-options.md` lists every flag (≈1614 lines).
This card distills the flags this repo actually uses + the few escape
hatches you reach for during diagnosis.

## Architecture targeting (`-arch`, `-code`, `-gencode`)

### The shorthand: `-arch=sm_XX`

`-arch=sm_120 -O2` (the repo's pattern) expands to:

```
--gpu-architecture=compute_120 --gpu-code=sm_120,compute_120
```

So you get **one cubin** (`sm_120`) and **one PTX** (`compute_120`).
Forward-compat is satisfied by the embedded PTX.

For **architecture-conditional** targets (suffix `a` / `f`):

```
-arch=sm_90a   →  --gpu-architecture=compute_90a
                  --gpu-code=sm_90a,compute_90,compute_90a
```

The `a`/`f` PTX is **not** forward-compat (see `blackwell-compatibility.md`).

### The verbose form: `-gencode=arch=...,code=...`

Use when targeting **multiple** arches:

```
nvcc -gencode=arch=compute_80,code=sm_80   \
     -gencode=arch=compute_100,code=sm_100 \
     -gencode=arch=compute_120,code=sm_120 \
     -gencode=arch=compute_120,code=compute_120 \
     ...
```

`arch=` is the **front-end PTX target** (must be a `compute_X`).
`code=` is the **back-end target** (`sm_X` for cubin, `compute_X` for PTX).
Only what you put in `code=` ends up in the binary.

### Special targets

| Target | Meaning |
|--------|---------|
| `-arch=native` | nvcc autodetects visible GPUs and builds for them. **No PTX is embedded** with this option. |
| `-arch=all` | One cubin per supported `sm_*`, plus PTX for the highest major virtual arch. |
| `-arch=all-major` | One cubin per `sm_*0` major version + the earliest supported, plus PTX for the highest major virtual arch. |
| `-arch=lto_NN` | Link-time-optimized device code only (no compute_NN PTX). Pair with `-dlto`. |

**Repo policy:** active cloud workloads should use sm_120 targets when they
require Blackwell behavior, and separate sm_86 targets for local RTX 3090 Ti
smoke runs. Architecture-conditional Blackwell features such as FP4/block-scaled
paths may require `sm_120a` or `sm_120f` and should be probe-gated. Anything
shipped more broadly should use explicit `-gencode` stanzas with at least one
`code=compute_X` for forward compatibility.

## Compilation-phase steering (§4.2.2)

| Flag | Effect |
|------|--------|
| `-c` / `--compile` | Compile each `.cu` to `.o`; do not link. Whole-program mode. |
| `-dc` / `--device-c` | Compile each `.cu` to a relocatable `.o` (separate compilation mode). Implies `-rdc=true`. |
| `-dlink` / `--device-link` | Device-only link step (combine multiple `-dc` outputs). |
| `-ptx` | Stop after producing `.ptx`. Useful for inspecting generated PTX. |
| `-cubin` | Stop after producing `.cubin`. |
| `-fatbin` | Stop after producing `.fatbin`. |
| `-E` | Preprocess only (host-side `cpp`-equivalent for `.cu`). |
| `--keep` (`-keep`) | Retain intermediates (`.cpp1.ii`, `.cudafe1.cpp`, `.ptx`, `.cubin`, …) in CWD. |
| `--keep-dir DIR` | Keep intermediates under `DIR` instead of CWD. |

**Repo idiom:** `nvcc -arch=sm_120 -O2 -keep` is the diagnostic build for a tracer-unknown-opcode investigation — it preserves both the `.ptx` and `.cubin` so you can `cuobjdump -sass` and `nvdisasm -hex`.

## Optimization & debug

| Flag | Effect |
|------|--------|
| `-O0` / `-O1` / `-O2` / `-O3` | Host-side optimization level. The **device** path goes through ptxas; see `-Xptxas -O…` below. |
| `-G` / `--device-debug` | Generate device debug info (disables most optimization). Use with `cuda-gdb`. |
| `-lineinfo` / `--generate-line-info` | Per-instruction source line correlation (PC -> source line) without disabling optimization. Useful for static SASS review, NVBit trace interpretation, and CUPTI source correlation where available. |
| `-pg` | Host-side profiling instrumentation (gprof). Rarely useful for CUDA. |
| `-Werror KIND,...` (`--Werror`) | **nvcc-frontend** flag: promote specific CUDA-side warning kinds to errors. Kinds include `all-warnings`, `cross-execution-space-call`, `reorder`, `default-stream-launch`, `missing-launch-bounds`, `ext-lambda-captures-this`, `deprecated-declarations`. See raw §4.2.8.10 for the full list. For host-compiler warning control (`-Wall`, `-Wno-...`), forward via `-Xcompiler` (§4.2.4 below). |

## Passing flags through to phase tools (§4.2.4)

Raw §4.2.4 documents **exactly five** pass-through options:

| Flag | Forwards to | Common uses |
|------|-------------|-------------|
| `-Xcompiler FLAGS` (`--compiler-options FLAGS`) | Host C++ compiler / preprocessor | `-Xcompiler "-Wall -fPIC"` — the only nvcc-side path to host-compiler warnings. |
| `-Xlinker FLAGS` (`--linker-options FLAGS`) | Host linker | `-Xlinker "-rpath,/some/path"`. |
| `-Xarchive FLAGS` (`--archive-options FLAGS`) | Library manager (`ar` / `lib`) | Rare. |
| `-Xptxas FLAGS` (`--ptxas-options FLAGS`) | `ptxas` (PTX→SASS) | `-Xptxas -v` (per-kernel reg + smem report), `-Xptxas -O3` (device opt level). |
| `-Xnvlink FLAGS` (`--nvlink-options FLAGS`) | `nvlink` (device linker) | Rarely needed outside `-rdc=true` flows. |

**Repo idiom:** `-Xptxas -v` is the single most useful diagnostic flag — gives per-kernel register count, shared-mem bytes (static + dynamic), spill bytes, and stack frame size.

## Register / launch-bound control

| Flag | Effect |
|------|--------|
| `-maxrregcount N` | Hard cap on registers per thread for **all** kernels in the TU. |
| `--ptxas-options=-maxrregcount=N` | Same; just routed via ptxas. |
| `__launch_bounds__(maxThreadsPerBlock [, minBlocksPerMultiprocessor])` (in source) | Per-kernel; preferred over `-maxrregcount` when only one kernel needs the cap. |

## Separate compilation (§4.2.2.4 + §4.2.7.5)

For multi-`.cu` projects where one TU's device code calls another's:

```bash
nvcc -dc -arch=sm_120 a.cu -o a.o
nvcc -dc -arch=sm_120 b.cu -o b.o
nvcc -dlink -arch=sm_120 a.o b.o -o link.o
g++ a.o b.o link.o -lcudart -o app
```

Or in one shot:

```bash
nvcc -rdc=true -arch=sm_120 a.cu b.cu -o app
```

`-rdc=true` enables relocatable device code (separate compilation) globally for
that command. **Repo observation:** single-file measurement kernels usually do
not need `-rdc=true`. Using it can slow the build and produce different SASS due
to the linker pass.

## Link-time optimization

| Flag | Effect |
|------|--------|
| `-dlto` (`-lto`) | Device-side link-time optimization. Must be set at **both** compile and link time. Stores high-level intermediate at compile time, optimizes at link time. |
| `--gpu-code='lto_NN'` | Embed only LTO intermediate (no SASS). |
| `-arch=lto_NN` | Same as above without the regular `compute_NN` target. |

Useful for separate-compilation projects with hot cross-TU device calls.

## Common diagnostic invocations

```bash
# Per-kernel register / spill report
nvcc -arch=sm_120 -O2 -Xptxas -v -c kernel.cu

# Generate PTX only
nvcc -arch=sm_120 -O2 -ptx kernel.cu -o kernel.ptx

# Build with line info for static SASS, NVBit, or CUPTI Activity correlation
nvcc -arch=sm_120 -O2 -lineinfo kernel.cu -o kernel

# Keep all intermediates for tracer / opcode debugging
nvcc -arch=sm_120 -O2 -keep -keep-dir build/ kernel.cu -o kernel

# Architecture-conditional Blackwell FP4 build
nvcc -gencode arch=compute_120a,code=sm_120a -O2 fp4_kernel.cu -o fp4_kernel
```

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| "no kernel image is available for execution on the device" | Cubin doesn't match host CC AND no PTX (or PTX is `compute_Xa`) | Add `-gencode=arch=compute_NN,code=compute_NN` to ship PTX. |
| "too many resources requested for launch" | Block size × regs/thread > register file | Reduce regs (`-maxrregcount`), reduce block size, or add `__launch_bounds__`. |
| Build time blew up after adding `-rdc=true` | Separate compilation triggers extra device link | Use whole-program mode (`-rdc=false`, the default) when there's only one `.cu`. |
| Inconsistent intermediates between two builds | Compiled `.cu` with two different option sets in the same dir | Use `-keep-dir` to isolate; raw guide §4.2.5.11 explicitly warns about this. |
| `-arch=native` produced a binary with no PTX | That's by design | Add an explicit `-gencode=arch=compute_NN,code=compute_NN` if forward compat is needed. |

## See also

- `.claude/knowledge/nvcc/nvcc-trajectory.md` — `.cu` → host C++ + PTX → cubin → fatbin pipeline; what each phase produces.
- `.claude/knowledge/blackwell/blackwell-compatibility.md` — when to include `compute_NN` PTX vs `sm_NN` cubin.
- `.claude/knowledge/sass/sass-tooling.md` — `cuobjdump` / `nvdisasm` for inspecting build output.
- `.claude/knowledge/best-practices/perf-occupancy-launch.md` — `__launch_bounds__` and `-maxrregcount` rationale.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the full option list — raw §4.2 covers ~1500 lines of flags; this card covers the repo-relevant subset.
- You need cross-compilation / host-target setup (Tegra, QNX, Windows-from-Linux) — raw §4.2.5 has the relevant subsection.
- You need NVCC environment variables (`CUDA_HOME`, `NVCC_PREPEND_FLAGS`, etc.) — raw §4.3.
- You need detailed phase options for `cudafe1` / `cudafe2` / `nvlink` / `fatbin` etc. — raw §4.2.9.
- You hit a flag that appears in older docs but is rejected by your installed nvcc — flags rotate per CUDA version; check the manual shipped with the installed CUDA Toolkit.
