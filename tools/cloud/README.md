# `tools/cloud/` — Non-Root Cloud Measurement Workflow

> **RETIRED/HISTORICAL — superseded by root CLAUDE.md (2026-07-02 re-founding);
> see `.claude/knowledge/measurement/pinned-protocol-v21.md`.** This file
> describes the old counter-free cloud workflow. Under the current charter,
> NCU is a first-class ground-truth instrument on all targets and root clock
> ops are required by pinned protocol v2.1. The body below is preserved
> as historical context for these scripts, with D8 retirement annotations.

Operator entry point for running SimBench-BW on cloud GPUs where the
container has root for package installation but no GPU-driver root:
no NCU, no privileged CUPTI profiling surfaces, no clock locking, and
no host driver or kernel-module changes.

## Contents

| Script | Purpose | Output |
|---|---|---|
| `container_bootstrap.sh` | Install container-safe build dependencies and verify `nvcc`, `cuobjdump`, and `nvidia-smi`. | Handoff commands for `run_suite.py`. |
| `preflight.sh` | Emit toolchain, GPU, and capability provenance. | `preflight_manifest.json`. |
| `preflight_manifest.py` | Validate a preflight manifest. | Non-zero on schema, architecture, or admission-gate failure; CUPTI capability fields are recorded-but-non-gating (D8). |
| `compile_path_inspector.py` | `cuobjdump`-based compile-path manifest; no GPU runtime required. | `compile_paths/<arch>.json`. |
| `trace_budgets.py` | Per-cell `max_trace_bytes`, `max_wall_time_s`, and sampling-policy enforcement. | Budget verdict plus `trace_volume_exceeded.json` sidecars. |
| `test_*.{sh,py}` | Host-independent tests for the cloud scripts. | Test results. |

## Measurement Surface

The active contract is counter-free:

1. **Historical, superseded by D8:** CUPTI Activity API produced kernel
   wall-clock timestamps and runtime sidecars. New timing truth is NCU-only;
   retained evidence validates against
   `workloads/simbench_bw/schemas/historical/cupti_activity_sidecar.v1.schema.json`.
2. `cudaEvent` timing provides the second wall-clock anchor.
3. NVBit produces dynamic instruction traces and optional `%clock64`
   phase-cycle ledgers.
4. NVML covariate polling records clock/thermal/power state without
   changing driver policy.
5. Static introspection via `cuobjdump` and retained compile artifacts
   provides ISA structure without privileged counters.

CUPTI Events, Metrics, Range Profiling, PM Sampling, NCU, and GPU clock
or mode changes are outside the contract.

## First Run

```bash
# 0. Inside nvidia/cuda:13.2.1-devel-ubuntu24.04 launched with:
#    --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
bash tools/cloud/container_bootstrap.sh

# 1. Optional but recommended provenance.
tools/cloud/preflight.sh --out results/preflight_manifest.json
```

> The commands below were removed in S4-C1 and do not exist at this revision.
> They are preserved verbatim as a historical record of the retired CUPTI
> producer; recover them only from git history. Do not run them.

```bash
# 2. Historical CUPTI producer build (retired by D8; retained evidence only).
(cd tools/nvbit_tracer && make)
make -C workloads/simbench_bw cupti-collector-lib

# 3. Validate static metadata.
python3 workloads/simbench_bw/verify_coverage.py --strict
python3 workloads/simbench_bw/validate.py --all
python3 workloads/simbench_bw/trace_size_estimator.py --validate-all

# 4. Historical CUPTI collection (retired by D8), then immutable-input replay.
python3 workloads/simbench_bw/run_suite.py --mode timing \
    --kernel rmsnorm_fwd_bf16 --shape-point joined_ac8_evidence \
    --enable-cupti-activity

python3 workloads/simbench_bw/run_suite.py --mode trace \
    --kernel rmsnorm_fwd_bf16 --shape-point joined_ac8_evidence \
    --phase-inject-ab

python3 workloads/simbench_bw/run_suite.py --mode emit-dual-run-manifest \
    --cupti-run-dir <cupti-artifact-dir> \
    --nvbit-run-dir <nvbit-artifact-dir> \
    --dual-run-manifest-output <out>/dual_run_manifest.json
```

## Offline Tests

```bash
bash tools/cloud/test_preflight_probes.sh
```

Mock fixtures under `tools/cloud/mock_inputs/` exercise the non-GPU
paths in CI.
