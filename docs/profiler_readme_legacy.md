# Non-Root GPU Measurement

This repository is now focused on measuring and predicting GPU application
behavior in environments where the process has no GPU-driver root privilege.
The hard contract is:

- no NCU / Nsight Compute profiling;
- no hardware performance counters;
- no driver-state-changing `nvidia-smi` operations, including clock locking;
- no kernel-module toggles or host-driver changes.

The research target is an NCU-replacement measurement stack for wall-clock
behavior and cache hit-ratio prediction using user-space signals: NVBit binary
instrumentation, CUPTI Activity, CUDA events, Nsight Systems where available,
static SASS inspection, and read-only environment covariates.

## Research Framing

The novelty is not a GPU simulator. The novelty is whether a non-root process
can produce precise, repeatable, and useful performance observables when the
usual hardware-counter path is unavailable.

The active questions are:

- How accurately can CUPTI Activity, CUDA events, and Nsight Systems reproduce
  application and per-kernel wall-clock timing under cloud DVFS?
- How accurately can NVBit SASS/address traces plus static binary features
  predict the **L2 cache hit ratio** without reading L2 hardware counters?
- What uncertainty bounds and reconciliation checks make those predictions
  scientifically defensible?

All claims are verified under the Two-Device Local Development Model evidence
statuses (`Offline-verified`, `Verified-live-sm_86`, `Verified-live-sm_120`,
`Cross-arch-verified`). Cross-arch promotion requires matching live evidence
on both local arches; an arch-coupled result on one arch does not extend to
the other without explicit live witness on the second arch.

### Submission-Track Scope (sm_86 + RTX 3090 Ti, L1 + L2)

The submission narrows the load-bearing prediction claim to **cache hit ratio
on sm_86 (Ampere GA102, RTX 3090 Ti)**, now covering both the L1/TEX and L2
sector layers. The submission-track overlay at
`workloads/simbench_bw/comparison_contract_sm86_submission.toml` pins:

- `scope.arch_set = ["sm_86"]` (single arch for the submission claim);
- `scope.cache_layers = ["L1", "L2"]` (both layers are load-bearing
  prediction layers; `"L1"` is the contract-side label the runner maps from
  the bundle-side `"L1TEX"`);
- `scope.device_set = ["RTX 3090 Ti"]`;
- publication-target gates (`cohort_paper_target_unique_slugs = 30`,
  `cohort_paper_target_cells = 31`,
  `aspirational_target_unique_slugs = 40`).

**L1 hit rate is promoted to a load-bearing, comparison-runner-scored metric.**
The headline runner emits per-cell L1TEX predictor bundles that the comparison
runner scores against the NCU L1 labels (`cache_hit_rate_l1tex_sector_pct`)
through the same 8-tuple-join / coverage-gate / bootstrap-CI path as L2; the
standalone L1 closure artifact is
`data/local_labels/l1_closure_comparison_sm_86.json`, and the separate
bounded-`[0,1]` L1 calibration head instance is
`data/local_labels/sm_86_l1_head.json` (the frozen L2 v5 head stays
byte-identical). On the current cohort only the cells whose L1 walker emitted a
scalar prediction enter the L1 geomean; the rest are typed
`predictor_l1_sass_provenance_missing` exclusions, so the L1 coverage gate
honestly reports `meets_floor = false` (live headline count below the smoke
floor) until the cohort grows. A separate, still-pending L1 diagnostic sidecar
(`data/sm_86_predictor_accuracy_recovery_v2/l1_diagnostics/`) remains a
typed-`l1_diagnostic_head_pending` placeholder and is never globbed by the
comparison runner's predictor-bundle discovery.

The base contract `workloads/simbench_bw/comparison_contract.toml` describes
the project's broader two-arch program (sm_86 + sm_120, two devices) and
remains the authoritative contract for non-submission flows. The overlay
narrows the submission view without modifying the base.

Cross-architecture validation on sm_120 (RTX 5060 laptop) is documented as a
follow-up after the sm_86 submission lands.

## Target Environment

### Research Target (Cloud / Server)

- Cloud GPU rentals that deny profiling permissions and clock control.
- NVIDIA Blackwell sm_120, especially RTX PRO 6000 Blackwell, as the primary
  research target.
- NVIDIA Hopper sm_90a / H100 as a secondary research platform when available.

### Local Development Model (Two-Device Cross-Arch)

Per CLAUDE.md "Two-Device Local Development Model", two local consumer devices
serve as equal-citizen first-class development arches:

- **RTX 3090 Ti (sm_86, Ampere desktop, GA102 die)** — primary architecture
  for several v3 RLCR loop sm_86 live-GPU sweeps. NCU access is permitted via
  the local-label exception (see Forbidden Surfaces below).
- **RTX 5060 laptop (sm_120, Blackwell consumer, GB206 die)** — second
  architecture for cross-arch validation. NCU access is permitted via the
  same local-label exception.

Both devices are valid live targets for the NVBit tracer's `tracer_tool.so`,
the CUPTI Activity collector, the simbench benchmark suite, and the
`SIMBENCH_REQUIRE_LIVE_GPU=1` pytest sweeps under
`workloads/simbench_bw/tests/`. The session running on one device cannot
independently verify behavior on the other; cross-arch claims require
matching live evidence from both arches.

## Layout

- `tools/cloud/` — non-root cloud bootstrap, preflight, provenance, and
  trace-budget helpers.
- `tools/nvbit_tracer/` — NVBit tracer used for dynamic SASS/address evidence.
- `tools/local_labels/` — local-only labeling driver and cross-host pivot
  operator toolchain. The labeling driver (`label_driver.sh`,
  `parse_ncu_metrics.py`, `sweep_repeatability_matrix.py`) invokes Nsight
  Compute on local devices (RTX 3090 Ti sm_86, RTX 5060 laptop sm_120) and
  emits per-launch label sidecars into the `data/local_labels/` quarantine
  path; those sidecars are consumed and validated by `import_label.py`
  against `workloads/simbench_bw/schemas/local_label_sidecar.schema.json`.
  The aggregator `repeatability_aggregator.py` reads the per-launch
  sidecars, applies the repeatability gate, and emits admitted-cell
  records into `data/local_labels/label_admitted.json` — those records
  are strict-validated against the v2 schema
  `workloads/simbench_bw/schemas/local_label_admitted.schema.json`. The
  cross-host pivot operator toolchain — `check_admission_floor.py`
  (admission-floor self-check against the contract's smoke-test floor;
  exit 0 admitted / 1 below floor / 2 schema-io error),
  `merge_admitted_cells.py` (cross-host evidence merge with strict-schema
  validation of both inputs and the merged output),
  `capture_observed_state.py` (hardware-metadata capture: device, driver,
  CUDA toolkit, observed clocks / throttle reasons; atomic-write JSON +
  CSV sidecars under
  `data/local_labels/sweeps/clock_throttle_evidence/`), and
  `validate_measurement_manifest.py` (per-cell measurement-manifest
  pre-flight validator with bidirectional `--binary`↔manifest coverage
  check) — supports the operator's cross-host transfer between the two
  local arches. The operator's reading-path is documented in
  `tools/local_labels/SWEEP_RUNBOOK.md`. Bounded by the local-label
  exception in CLAUDE.md Rule 1; this is the ONLY repository path where
  NCU invocation is permitted.
- `workloads/simbench_bw/` — active benchmark and evidence harness for CUPTI,
  CUDA-event, NVBit, and NVML-covariate collection.
- `workloads/portable_lcd/` — second active benchmark registry with
  portable-LCD-discipline kernels (sm_86 / sm_90 / sm_120 baseline; no
  architecture-conditional `sm_90a` / `sm_120a` / `sm_120f` build targets).
- `data/local_labels/` — quarantine directory for NCU-derived ground-truth
  labels emitted by `tools/local_labels/`. Excluded from the no-NCU static
  guard scan; no other path in the repository may write into this directory.
- `references/NVIDIA_Manuals/` — local NVIDIA documentation mirror. NCU docs
  are kept only for metric-definition comparison, not as an allowed profiling
  surface (except inside the local-label exception above).
- `references/papers/` — prior-art corpus for performance modeling,
  instrumentation, microbenchmarking, and related simulator literature.
- `references/Rust_refs/` — retained source-material references from the fork
  history.
- `.claude/knowledge/` — retained pre-built knowledge cards that link back to
  source material under `references/`.

Historical simulator plans, roadmap drafts, `.claude/memory/`, `.humanize`
runs, portable-LCD experiments, and generated old results were removed during
the May 3, 2026 cleanup. Reference-backed source material and knowledge cards
are preserved.

## Quick Start

```bash
# Build the NVBit tracer.
(cd tools/nvbit_tracer && make)

# Inside a fresh nvidia/cuda:13.2.1-devel-ubuntu24.04 container launched
# with --gpus all:
bash tools/cloud/container_bootstrap.sh

# Validate committed non-GPU metadata and schemas.
python3 workloads/simbench_bw/verify_coverage.py --strict
python3 workloads/simbench_bw/validate.py --all

# Build the CUPTI Activity collector.
make -C workloads/simbench_bw cupti-collector-lib

# Emit the preflight manifest (mandatory companion of --enable-cupti-activity;
# run_suite.py's EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED guard
# rejects --enable-cupti-activity if --preflight-manifest is omitted).
bash tools/cloud/preflight.sh --out /tmp/simbench_preflight.json

# CUPTI/CUDA-event timing run.
python3 workloads/simbench_bw/run_suite.py --mode timing \
    --kernel rmsnorm_fwd_bf16 --shape-point joined_ac8_evidence \
    --enable-cupti-activity \
    --preflight-manifest /tmp/simbench_preflight.json

# NVBit trace/phase run.
python3 workloads/simbench_bw/run_suite.py --mode trace \
    --kernel rmsnorm_fwd_bf16 --shape-point joined_ac8_evidence \
    --phase-inject-ab

# Pair the two producer runs.
python3 workloads/simbench_bw/run_suite.py --mode emit-dual-run-manifest \
    --cupti-run-dir <cupti-artifact-dir> \
    --nvbit-run-dir <nvbit-artifact-dir> \
    --dual-run-manifest-output <out>/dual_run_manifest.json
```

## Forbidden Surfaces

A repo-wide static guard
(`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`) rejects
active-code attempts to use privileged profiling or driver-control surfaces:

- `ncu`, `nv-nsight-cu-cli`, `.ncu-rep`, `.ncu-repz`;
- `nvidia-smi` clock, power, MIG, persistence, or compute-mode mutation flags;
- `--lock-gpu-clocks`, `--lock-memory-clocks`;
- `NVreg_RestrictProfilingToAdminUsers` or other host-driver module toggles;
- `modprobe nvidia`, DKMS, MOK, host driver install, or reboot assumptions;
- CUPTI Events, Metrics, Range Profiling, PM Sampling, and PerfWorks surfaces.

### Local-label exception (Rule 1 carve-out)

NCU and `nv-nsight-cu-cli` MAY be invoked on local devices (RTX 3090 Ti sm_86
and RTX 5060 laptop sm_120) to produce ground-truth labels for the prediction
model, but ONLY into the quarantine path `data/local_labels/`, and ONLY by
code under `tools/local_labels/`. The static guard above excludes those two
paths from the no-NCU scan and replaces the exclusion with a scoped allowlist
test that keeps the rest of the contract (the other surfaces above and Rules
2-5 in CLAUDE.md) binding inside those subtrees. This exception is bounded
by CLAUDE.md Rule 1; the cloud/server NCU prohibition above is absolute and
binding for every collection script that runs on a non-local target.
