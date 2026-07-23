# SimBench-BW Dependency Pins

> CUPTI Activity collector build/run instructions below are retired (D8, S4-C1) and are preserved as historical record; all other pins remain active.

These pins are the frozen versions that measurement reproducibility is
declared against. Bumping any entry below requires a schema-version bump in
the affected artifact and an explicit changelog entry.

## Toolchain

| Component | Version | Source |
|---|---|---|
| NVIDIA driver | 595.58.03 (or newer) | host-installed |
| CUDA toolkit | 13.2 | host-installed |
| `nvcc` | bundled with CUDA 13.2 | `$(CUDA_HOME)/bin/nvcc` |
| `ptxas` | bundled with CUDA 13.2 | `$(CUDA_HOME)/bin/ptxas` |
| `cuobjdump` / `nvdisasm` | bundled with CUDA 13.2 | `$(CUDA_HOME)/bin/` |
| Nsight Systems | bundled with CUDA 13.2 where available | optional user-space timeline corroboration |
| CUPTI Activity API | bundled with CUDA 13.2 (`libcupti.so` + `cupti.h`) | `$(CUDA_HOME)/extras/CUPTI/{lib64,include}` (apt: `cuda-cupti-13-2`); retired by D8 in S4-C1, historical record only |
| OpenSSL libcrypto | system (any 1.1+/3.x) | host-installed (`-lcrypto`) |
| NVBit | 1.8 | vendored at `tools/nvbit_tracer/nvbit_release/` |
| Host C++ compiler | g++ 15.x | system |
| Python | 3.11+ | system |

## Library pins (CUTLASS / Triton / FlashInfer etc.)

The first-wave kernels take a small set of external pins. These are frozen at
the Round-specific commit that the kernel's `.cu` source was built against;
subsequent rounds may bump them only when the kernel source is updated in the
same commit.

| Library | Pin | Notes |
|---|---|---|
| CUTLASS | TBD (commit pinned when the first-wave kernel depending on it lands) | Used by `moe_grouped_gemm_bf16` / `moe_grouped_gemm_fp8` / `nvfp4_blockscale_gemm`. |
| CuTe DSL | bundled with CUTLASS pin | FlashAttention-4 / FlashMLA reference layouts. |
| Triton | TBD | Used only for the `tier=control` rows if they adopt Triton-to-cubin paths. |
| FlashInfer-SM120 | TBD | Required for the NVFP4 grouped-GEMM MoE row on sm_120a due to upstream CUTLASS grouped-GEMM issue #3096. |
| PyTorch | 2.x (any version that exposes FP8 / BF16 dtype tensors) | CPU-only paths for `references/`. |

Absent entries mean "not yet pinned"; they MUST be frozen before the
corresponding kernel lands. `verify_coverage.py --check-dependency-pins` will
fail the first release tag that promotes a kernel with an unfrozen pin.

## CUPTI Activity collector build entry (retired by D8 in S4-C1; historical record only)

The wall-clock telemetry collector at `cupti_activity_collector.{h,cu}`
is built as a shared library via the `cupti-collector-lib` make
target. The orchestrator loads it into the workload binary's process
via `LD_PRELOAD`. It links against `libcupti` (CUPTI Activity API
only — no profiling/range/PM surfaces; the static guard at
`tests/test_no_root_or_ncu_static_guard.py` enforces this) and
`libcrypto` (OpenSSL SHA-256 for the `kernel_signature_sha256` helper
that conforms to `schemas/kernel_signature.spec.md`):

```
make -C workloads/simbench_bw cupti-collector-lib
```

Default toolchain locations (overridable via env):
- `CUDA_HOME=/usr/local/cuda`
- `CUPTI_INCLUDE=$(CUDA_HOME)/extras/CUPTI/include`
- `CUPTI_LIB=$(CUDA_HOME)/extras/CUPTI/lib64`
- `COLLECTOR_ARCH=sm_120` (matches the cloud RTX PRO 6000 Blackwell
  target; recompile with `COLLECTOR_ARCH=sm_86` for the local RTX 3090 Ti
  smoke-test host on this branch, or `COLLECTOR_ARCH=sm_90a` for H100
  hosts).

The output library lives at
`workloads/simbench_bw/libsimbench_cupti_activity_collector.so`;
`run_suite.py`'s `--enable-cupti-activity` path checks this exact
location and exits typed `CuptiActivityCollectorBinaryMissing` (exit
12) when the library is absent. The orchestrator sets
`LD_PRELOAD=<absolute path to .so>` plus the `SIMBENCH_CUPTI_*`
metadata env vars on the per-repeat workload subprocess; the library's
constructor + destructor handle the CUPTI lifecycle in the workload
process.

## Target hardware capability references

- Blackwell workstation (sm_120 / sm_120a / sm_120f): see
  `references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md`.
- Hopper datacenter (sm_90a): see `references/NVIDIA_Manuals/ptx_isa/00_index.md`.
- Excluded targets: `sm_100*` (B200 datacenter) — no hardware available.

## Platform-identity pins

Every measurement manifest carries `compute_capability_major` +
`compute_capability_minor` captured at orchestrator startup. Known targets:

| Platform | Compute capability |
|---|---|
| NVIDIA H100 | 9.0 |
| RTX PRO 6000 Blackwell | 12.0 |
| RTX 3090 Ti | 8.6 |

Any other compute capability at startup refuses to run without
`--allow-any-device`.

## Schema-version drift policy

- Every artifact schema under `schemas/` carries `schema_version`.
- Producer tooling (`run_suite.py`, `update_goldens.py`) emits
  `schema_version` matching the current `schemas/*.json` value.
- Consumer tooling (`validate.py`, `verify_coverage.py`,
  `verify_artifacts.py`, and the dual-run/phase consumers) rejects artifacts whose
  `schema_version` differs from the committed schema.
- Bumping any `schema_version` requires updating every consumer in the
  same commit.
