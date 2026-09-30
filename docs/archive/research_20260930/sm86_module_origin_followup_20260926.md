# SM86 module-origin follow-up — 2026-09-26

The accepted-launch module resolver was extended after the earlier captures
showed `origin_unresolved` for runtime-launched kernels.  The producer now
resolves the exact accepted launch handle to its owning CUDA module or
context-independent CUDA library, avoiding same-name cross-module guesses.
This is provenance only: it does not alter SASS, addresses, predicates, or
simulator replay.

## What changed

`tools/nvbit_tracer/tracer_tool/tracer_tool.cu` now:

* keeps module origins keyed by `(CUcontext, CUmodule)`;
* keeps library origins keyed by `CUlibrary` because CUDA libraries are
  context-independent;
* resolves an accepted launch with `cuFuncGetModule`, or with
  `cuKernelGetLibrary` for the context-less `CUkernel` launch convention; and
* erases module origins at context teardown and library origins at
  `cuLibraryUnload`, preventing stale opaque handles from being reused.

Owner queries are made with callback re-entry suppressed.  A failed owner
query remains an explicit unresolved provenance result, so an unsupported or
opaque driver path cannot silently turn into a guessed module identity.

## Bounded live evidence

The rebuilt producer was exercised on the local SM86 host with the pinned
CUDA 13.2/GCC 14/protobuf environment.  Both workloads were launched with
`SKIP_ENHANCED_TRACER=1`, one active kernel selected, and no Accel-Sim or
MGPUSim process was run:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT" SKIP_ENHANCED_TRACER=1 \
ACTIVE_FROM_START=1 DYNAMIC_KERNEL_LIMIT_START=1 DYNAMIC_KERNEL_LIMIT_END=1 \
  workloads/portable_lcd/<workload>/<workload>_run \
    --timing_shape --warmup 0 --reps 1 --allow-any-device
```

| Workload | Return code | Resolution source | Module status | Basename | SHA-256 prefix |
| --- | ---: | --- | --- | --- | --- |
| `gemm_hmma_bf16_sm80_run` | 0 | `library_registry` | `stable_static_module` | `gemm_hmma_bf16_sm80_run` | `55133e328fc16238` |
| `rmsnorm_fwd_bf16_sm80_run` | 0 | `library_registry` | `stable_static_module` | `rmsnorm_fwd_bf16_sm80_run` | `2897b18afbd4eafd` |

The full sidecars were written during the run to the ignored temporary paths
`/tmp/gpusim_origin4.D6cjny/extra_info/traced_kernel_modules.json` (GEMM) and
`/tmp/gpusim_origin4.mBOih7/extra_info/traced_kernel_modules.json` (RMSNorm).
Each contains the absolute executable path, regular-file validation, and the
full SHA-256.  The output is therefore rebindable by content identity rather
than by a benchmark-name guess.

The adversarial smoke `tools/nvbit_tracer/smoketest/module_origin.sh` also
builds two same-name cubins with different outputs and checks four paths:

| Path | Result |
| --- | --- |
| two live `CUmodule` handles, unload/reload, anonymous `cuModuleLoadData` | 5 exact-owner/typed-unavailable launches |
| two context-independent `CUlibrary` handles | 4 exact-owner launches |
| library handles converted with `cuKernelGetFunction` | 4 exact-owner launches |
| library loaded before context creation | 4 typed `origin_unresolved` launches |

The smoke retained its output under
`results/captures/sm86_module_origin_20260926_QA7ttn/`; it does not run a
simulator or consume NCU labels.

The local host identity was NVIDIA GeForce RTX 3090 Ti (SM 8.6), driver
615.71.09.  Current producer SHA-256 was
`50cae3b9f936eed9c36408b18bdf446cc1810493ea369ddeaac1d46cf1a0fb62`.

## Remaining boundary

This closes the observed static-module provenance miss for the two fresh
runtime-launched kernels; it is not a universal guarantee for JIT code,
deleted temporary modules, opaque driver/library paths, or graph-node launches.
Those cases remain typed `no_stable_static_module` unless the producer can
obtain an on-disk regular file and hash it.  Provenance still has no bearing
on the separate unmodeled writeback/coherence and value-dependent CFG timing
tails.
