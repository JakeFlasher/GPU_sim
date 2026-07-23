---
Use when: reasoning about which compilation phase produced a given artifact (`.ptx` / `.cubin` / `.fatbin` / `.o`); deciding whether a `__CUDA_ARCH__` guard belongs in your kernel; debugging "no kernel image" errors with knowledge of where in the trajectory they came from; understanding what driver JIT will and will not do.
Primary raw source: references/NVIDIA_Manuals/nvcc_guide/05_gpu-compilation.md.
Supporting sources: references/NVIDIA_Manuals/nvcc_guide/02_compilation-phases.md (phase enumeration); references/NVIDIA_Manuals/nvcc_guide/03_the-cuda-compilation-trajectory.md (overall pipeline diagram); references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md (cubin survival rules).
Repo touchpoints: workloads/simbench_bw/, tools/nvbit_tracer/, tools/cloud/compile_path_inspector.py.
Out of scope: per-flag detail (in `nvcc-flags.md`); cross-compilation host-target setup; ptxas optimization levels.
Version family: CUDA 13.x. Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# nvcc Compilation Trajectory — `.cu` → Cubin / Fatbin / Executable

The mental model: nvcc is a **driver** that orchestrates several phase tools. Each phase has stable inputs and outputs across CUDA releases, but the intermediate phase tools may shift.

## Two-stage compilation (raw guide §5.4)

Every CUDA build is a **two-stage** flow:

```
                  Stage 1                     Stage 2
.cu  ──►  preprocess + compile to PTX  ──►  ptxas: PTX → cubin (SASS)
         (target: compute_X, "virtual")     (target: sm_X, "real")
```

| Stage | Target arg | Output | Notes |
|-------|-----------|--------|-------|
| 1 | `--gpu-architecture=compute_X` (virtual) | PTX text | Bound to the **feature set** of `compute_X`. Forward-compat. |
| 2 | `--gpu-code=sm_X` (real) | Cubin (per-arch SASS) | Bound to specific real GPU arch. **Not** forward-compat across major. |

**Raw-guide policy** (§5.4): virtual arch `compute_X` should be chosen **as low as possible** for broadest device coverage; real arch `sm_X` should be chosen **as high as possible** for performance when the target device is known.

**Repo observation:** `-arch=sm_120` expands per raw §5.7.2.2 to
`--gpu-architecture=compute_120 --gpu-code=sm_120,compute_120`. That pins the
binary to Blackwell CC 12.0+ and emits no lower-virtual-arch coverage. This is
acceptable for cloud-only sm_120 measurement kernels. For artifacts reused on
the local sm_86 smoke host or shipped more broadly, use explicit `-gencode`
stanzas.

### Why two stages exist

NVIDIA cannot guarantee SASS-level binary compat across GPU generations. PTX is the stable intermediate — once embedded in the binary, the **driver JITs** PTX to whatever real arch is present at run time. That's how forward-compat works.

## Compatibility model recap (raw guide §5.1)

| Statement | Reality |
|-----------|---------|
| Cubin compatible across major CC | **No** — `sm_80` cubin won't run on `sm_90`/`sm_100`/`sm_120`. |
| Cubin compatible across minor CC within same major | **Yes**, in the direction of newer minor: `sm_80` runs on `sm_86`, `sm_89`. |
| PTX forward-compat across major | **Yes** — `compute_80` PTX JITs to `sm_90` / `sm_100` / `sm_120`. |
| PTX with `compute_Xa` (architecture-conditional) forward-compat | **No** — `compute_90a` does NOT JIT to `sm_100`/`sm_120`. |

## Supported real arches (raw guide §5.2)

All the SASS targets the installed nvcc 13.x recognizes:

| Generation | Real archs (CC) |
|------------|-----------------|
| Turing | `sm_75` (7.5) |
| Ampere | `sm_80`, `sm_86`, `sm_87`, `sm_88` |
| Ada | `sm_89` |
| Hopper | `sm_90`, `sm_90a` |
| Blackwell | `sm_100/100f/100a`, `sm_103/103f/103a`, `sm_110/110f/110a`, **`sm_120/120f/120a`**, `sm_121/121f/121a` |

Same list applies to the virtual arches (`compute_75` … `compute_121a`) and to LTO targets (`lto_75` … `lto_121a`).

## Phase outputs (raw guide §2.3-§2.4)

| Output | nvcc flag | Purpose |
|--------|-----------|---------|
| `.cpp.ii` | `--cuda` | Host-compilable C++ after CUDA preprocessing. |
| `.ptx` | `--ptx` | Stage-1 output: textual PTX for one virtual arch. |
| `.cubin` | `--cubin` | Stage-2 output: SASS for one real arch. |
| `.fatbin` | `--fatbin` | Container holding multiple PTX + cubin entries (per-arch). |
| `.o` / `.obj` | `--compile` (`-c`) | Host object file with embedded device code (PTX/cubin/fatbin). |
| `a.out` | (no phase flag) | Linked executable; nvcc compiles + links by default. |
| `*.ltoir` | `--ltoir` | LTO intermediate for `-dlto`. |
| `*.optixir` | `--optix-ir` | OptiX-targeted IR. |
| `a_dlink.o` / `.cubin` / `.fatbin` | `-dlink` | Output of the device-only link step. |

## Just-in-time compilation (raw guide §5.6.1)

When the binary contains PTX but no matching cubin for the run-time GPU:

1. Driver loads PTX at run time.
2. Driver runs **PTXAS internally** to produce a cubin matching the current GPU.
3. Cubin is cached (CUDA driver compile cache, persistent across runs).

Cost: first-launch latency. Mitigated by the cache.

**`CUDA_FORCE_PTX_JIT=1`** forces the driver to ignore embedded cubin and JIT from PTX every time. The Blackwell compatibility guide uses this to test "does my pre-Blackwell binary still work via PTX?" (raw `blackwell_compatibility/01_blackwell-architecture-compatibility.md` §1.3.1).

## Fatbinaries (raw guide §5.6.2)

A fatbin is a container of **multiple** code instances for the **same source**:

```bash
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80,sm_86,sm_89
# Produces a fatbin holding: compute_80 PTX + sm_86 cubin + sm_89 cubin
```

At launch, the driver picks the closest entry:

- Exact `sm_X` match → use that cubin.
- Else nearest cubin within the same major → may run.
- Else PTX → JIT to the run-time arch.
- Else **launch fails** (`cudaErrorNoKernelImageForDevice`).

## Shorthand examples (raw guide §5.7.2)

| Command | Equivalent |
|---------|------------|
| `nvcc x.cu --gpu-architecture=sm_86` | `nvcc x.cu --gpu-architecture=compute_86 --gpu-code=sm_86,compute_86` |
| `nvcc x.cu --gpu-architecture=compute_80` | `nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80` (PTX only) |
| `nvcc x.cu` (no arch) | Defaults to `compute_75 --gpu-code=sm_75,compute_75` (varies per CUDA version) |

## Multiple-arch builds (raw guide §5.7.3)

For multiple arches with **different feature sets**, use repeated `--generate-code`:

```bash
nvcc x.cu \
    --generate-code arch=compute_80,code=sm_80 \
    --generate-code arch=compute_80,code=sm_86 \
    --generate-code arch=compute_120,code=sm_120 \
    --generate-code arch=compute_120,code=compute_120  # PTX for forward compat
```

Or the bracket shorthand for code lists:

```bash
nvcc x.cu \
    --generate-code arch=compute_80,code=[sm_80,sm_86,sm_89] \
    --generate-code arch=compute_120,code=sm_120
```

## Source-side macros nvcc defines (raw guide §2.1, §5.7.4)

| Macro | Meaning |
|-------|---------|
| `__NVCC__` | nvcc is compiling this TU (host or device side). |
| `__CUDACC__` | nvcc is compiling a CUDA TU specifically. |
| `__CUDACC_RDC__` | Relocatable device code mode (`-rdc=true`). |
| `__CUDACC_DEBUG__` | `-G` (device debug) is on. |
| `__CUDACC_VER_MAJOR/MINOR/BUILD__` | nvcc version components. |
| `__CUDA_ARCH__` | **In device code** during stage 1 for `compute_xy`: defined as `xy0`. Use to gate device-side feature use. **Do not** read in host code. |
| `__CUDA_ARCH_LIST__` | Comma-separated list of `__CUDA_ARCH__` values for all targeted virtual archs in the current compile. Defined for host & device code. |

**Repo idiom:** for a kernel that needs Blackwell-only PTX, gate with:

```cpp
__global__ void k() {
#if __CUDA_ARCH__ >= 1200
    // Use FP4 / tcgen05 path
#elif __CUDA_ARCH__ >= 900
    // Hopper fallback (wgmma)
#else
    // Pre-Hopper fallback
#endif
}
```

## Virtual vs real arch — choosing between

- Use `code=sm_X` → real cubin emitted; fastest at launch (no JIT) but only for that arch family.
- Use `code=compute_X` → PTX emitted; broadest forward-compat but JIT cost on first launch.
- Use **both** → fatbin with cubin for known arches + PTX for everyone else. Recommended for shipped binaries.
- Use neither (`-arch=compute_X` alone) → PTX only, no cubin. Maximum portability, slowest first launch.

## See also

- `.claude/knowledge/nvcc/nvcc-flags.md` — flag-level detail for the trajectory described here.
- `.claude/knowledge/blackwell/blackwell-compatibility.md` — cubin / PTX survival rules across CC majors.
- `.claude/knowledge/sass/sass-tooling.md` — `cuobjdump` / `nvdisasm` for inspecting trajectory output.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the full nvcc phase list (`--cuda`, `-cubin`, `-fatbin`, `-dlink`, etc., with default output names) — raw §2.4 has the complete table.
- You need cross-compilation trajectory (host on Linux, target QNX / Tegra / Windows-from-Linux) — raw §5 final subsections.
- You hit a `__CUDA_ARCH__` macro behavior that surprises you (e.g. multiple-pass interactions with `__CUDA_ARCH_LIST__`) — raw §5.7.4 is the authoritative source.
- You are interested in OptiX-IR or the LTO intermediate format — raw §5.6 / §2.4 final entries.
