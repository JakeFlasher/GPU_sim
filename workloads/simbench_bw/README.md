# SimBench-BW

The active S4 contract uses NCU for authoritative timing, NVBit for dynamic
trace structure, and checked-in protobuf bindings for offline decoding. Run
`python3 validate.py --all`, `python3 verify_coverage.py --strict`, and the
tracked pytest suite to validate the current tree. Retained sidecars remain
read-only historical evidence; no retired producer is rebuilt or invoked.

<!-- HISTORICAL-D8-BEGIN -->
The remainder of this bounded region records the pre-S4 workflow for evidence
interpretation only. It is non-operative and its commands must not be run.

> **D8 / S4-C1:** Live CUPTI Activity production and admission are retired.
> Collector-build commands, `--enable-cupti-activity`, live dual-source
> admission and tolerance-calibration commands, and sidecar-environment
> instructions retained below are historical records and must not be run.
> Offline replay and validators over retained evidence remain supported.

SimBench-BW is the active kernel suite and evidence harness for non-root GPU
measurement. It exists to collect wall-clock timing, dynamic instruction
structure, address-trace evidence, phase-cycle evidence, and environment
covariates without NCU, hardware counters, or driver clock control.

The current prediction target is measurement-facing: wall-clock behavior and
cache hit-ratio estimates from CUPTI Activity, CUDA events, Nsight Systems
where available, NVBit traces, static SASS features, and read-only covariates.

## Source of Truth

This README is authored against repository HEAD on the active development
branch (`5060_3090ti_cross_dev` at the time of writing). The documented
commands assume the live CLI shapes of the scripts they invoke; running
them from an earlier worktree may surface stale flags or path layouts.
For re-reads without changing the worktree, prefer
`git show <branch>:workloads/simbench_bw/README.md` over `git checkout`.
This pattern matches `BL-20260504-runbook-self-reference`: documentation
and verification target are decoupled, so the operator never re-reads the
runbook from a stale post-checkout worktree.

All evidence claims produced by this harness are bound by the Two-Device
Local Development Model evidence statuses per `CLAUDE.md`:
`Offline-verified`, `Verified-live-sm_86`, `Verified-live-sm_120`, and
`Cross-arch-verified`. Cross-arch promotion requires matching live evidence
on both local arches (RTX 3090 Ti sm_86 and RTX 5060 laptop sm_120). The
producer-side helpers `dual_workload_pairing_index.py`,
`cupti_dual_source_live_admission.py`, and `nvml_metadata_verifier.py`
underwrite AC-15 / AC-16 promotions and enforce strict sidecar identity
contracts; see `CLAUDE.md` "Two-Device Local Development Model" for the
discipline they implement.

## Quick Start

```bash
# No GPU required (host-only validators + pytest):
python3 workloads/simbench_bw/verify_coverage.py --strict
python3 workloads/simbench_bw/validate.py --all
python3 workloads/simbench_bw/trace_size_estimator.py --validate-all
python3 -m pytest workloads/simbench_bw/tests/

# HISTORICAL / DO NOT RUN: the collector target and live CUPTI timing flag were
# removed in S4-C1. These commands document the retired producer contract only.
# Build the CUPTI Activity collector shared library:
make -C workloads/simbench_bw cupti-collector-lib

# Emit a preflight manifest (mandatory companion of --enable-cupti-activity):
bash tools/cloud/preflight.sh --out /tmp/simbench_preflight.json

# CUPTI Activity + CUDA-event timing run (rmsnorm_fwd_bf16 AND
# dram_stream_stride_mix are the current dual-source CUPTI/CUDA-event
# timing producers because both implement the
# `simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook that the
# CUPTI Activity collector reads to reconcile per-launch wall-clock pairs.
# rmsnorm_fwd_bf16 was promoted at v3 R20-R23; dram_stream_stride_mix was
# promoted at v6+v7 (commits 9f5a2ca, 14e37bc, eaf5eee, 30ae840, 5ea52f3)):
python3 workloads/simbench_bw/run_suite.py --mode timing \
    --kernel rmsnorm_fwd_bf16 --shape-point timing_00 \
    --enable-cupti-activity \
    --preflight-manifest /tmp/simbench_preflight.json

# Same flags also work for the second CUPTI/CUDA-event timing producer:
python3 workloads/simbench_bw/run_suite.py --mode timing \
    --kernel dram_stream_stride_mix --shape-point timing_00 \
    --enable-cupti-activity \
    --preflight-manifest /tmp/simbench_preflight.json

# NVBit trace + A/B phase-injection run (dram_stream_stride_mix is the
# canonical NVBit trace producer; the trace and timing roles are
# orthogonal — dram_stream_stride_mix serves both):
python3 workloads/simbench_bw/run_suite.py --mode trace \
    --kernel dram_stream_stride_mix --shape-point trace_00 \
    --phase-inject-ab

# Pair the two producer runs into a dual-run manifest:
python3 workloads/simbench_bw/run_suite.py --mode emit-dual-run-manifest \
    --cupti-run-dir <cupti-artifact-dir> \
    --nvbit-run-dir <nvbit-artifact-dir> \
    --dual-run-manifest-output <out>/dual_run_manifest.json
```

Historical pre-S4-C1 behavior (do not run): the `--enable-cupti-activity`
flag was gated by `--mode timing` and required
`--preflight-manifest <PATH>`; the orchestrator exits with
`EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED` if either gate is
violated. The preflight manifest's SHA-256 (excluding its `timestamp`
field) is recorded on every collection sidecar so a measurement run can
be re-tied to the toolchain provenance that produced it.

The timing/trace producers serve overlapping roles. `dram_stream_stride_mix`
is the canonical NVBit trace producer (used for `--mode trace` capture and
the NVBit-side phase-cycle ledger). `rmsnorm_fwd_bf16` (promoted at v3
R20-R23 on sm_86) AND `dram_stream_stride_mix` (promoted at v6+v7 on sm_86;
commits `9f5a2ca`, `14e37bc`, `eaf5eee`, `30ae840`, `5ea52f3`) are the
current AC-16-eligible dual-source CUPTI/CUDA-event timing producers
because both host drivers implement the
`simbench_cupti_record_cuda_event_wall_time_ms` dlsym hook that the CUPTI
Activity collector reads to reconcile per-launch wall-clock pairs against
the CUDA-event source. The v6+v7 hook addition to `dram_stream_stride_mix`
is regression-locked by `tests/test_dram_stream_stride_mix_hook_static.py`,
which asserts the dlsym name appears in the source AND the
`printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);` literal appears
EXACTLY twice (warmup branch + timed branch). Workloads outside this
2-producer set still lack the hook and will deterministically fail
dual-source reconciliation; extending the hook to additional roster
entries remains a queued side issue. The v6+v7 sm_86 leg is
`Verified-live-sm_86`; the matching sm_120 leg on the 5060 laptop is
deferred to the v8 plan at
`plans/ac16_dram_stream_cupti_hook_sm120_plan.md`, which targets the
`Cross-arch-verified` promotion in a fresh laptop RLCR loop.

## Layout

| Path | Role |
|---|---|
| `roster.toml` | Canonical benchmark roster. |
| `platform_matrix.toml` | Per-row platform contract. |
| `splits.toml` | Calibration / validation / holdout split for model fitting and evaluation. |
| `tolerances.toml` | Per-(dtype, reduction_kind, kernel_family) correctness tolerances. |
| `trace_budgets.toml` | Per-slug NVBit trace-size and wall-time budgets. |
| `shapes/<slug>.toml` | Timing, trace, and correctness shape points. |
| `references/` | CPU/reference implementations used by correctness mode. |
| `schemas/` | JSON schemas for active sidecars and manifests. |
| `duration_repeatability_acceptance.{toml,RETIRED.md}` | Frozen historical duration-admission thresholds and their retirement note. |
| `l1/` ... `l4/` | Kernel sources and per-kernel Makefiles. |
| `golden/` | Small committed fixtures such as CUPTI tolerance and CUPTI/NVBit join fixtures. |
| `tests/` | Host-independent pytest suite plus static guards. |

## Active Scripts

| Script | Role |
|---|---|
| `run_suite.py` | Main orchestrator. Modes: `correctness`, `timing`, `trace`, `regression`, `emit-dual-run-manifest`. |
| `cupti_activity_collector.{cu,h}` | LD_PRELOAD CUPTI Activity collector for per-launch wall-clock sidecars. |
| `nvbit_trace_summary_adapter.py` | Reduces NVBit traces to per-launch join summaries. |
| `phase_cycle_ledger_aggregator.py` | Decodes NVBit phase-record binaries and emits phase-cycle ledgers. |
| `phase_cycle_ledger_consumer.py` | Admits/rejects phase ledgers using overhead and phase-boundary rules. |
| `phase_inject_orchestrator.py` | Runs the A/B no-injection vs phase-injection pair and emits per-launch ledgers. |
| `dual_run_manifest_emitter.py` | Joins CUPTI and NVBit producer directories into a schema-valid dual-run manifest. |
| `dual_run_manifest_consumer.py` | Enforces manifest presence, ambiguity, binary identity, and NVML drift checks. |
| `dual_workload_pairing_index.py` | Combined dual-workload pairing index supporting `portable_lcd` + `simbench_bw` dual-registry contribution; underwrites AC-15 producer-side evidence. |
| `cupti_dual_source_live_admission.py` | Strict admission helper for `--enable-cupti-activity` timing capture; underwrites AC-16 producer-side evidence. Enforces a full sidecar identity contract across 4 surfaces × 5 axes via 20 typed `DEFECT_TOKENS` with scenario-backed token coverage. |
| `nvml_metadata_verifier.py` | NVML pin verifier (cadence / scope / status / paths consistency) used by the strict admission helper above and by AC-16 metadata-pin tests. |
| `calibrate_dual_source_tolerance.py` | Recalibration tool for the dual-source CUPTI / CUDA-event tolerance; produces `golden/cupti_dual_source_tolerance.json` from a 1000-sample pool. |
| `merge_evidence.py` | CUPTI/NVBit offline join helper for `cupti_activity_*.json` plus `nvbit_trace_summary_*.json`. |
| `trace_size_estimator.py` | Static trace-size pre-admission predictor. |
| `validate.py` | Schema and semantic validator for committed and produced artifacts. |
| `verify_coverage.py` | Metadata consistency and coverage verifier. |
| `verify_artifacts.py` | Trace-budget sidecar verifier. |
| `verify_kernel_contracts.py` | Optional cubin/SASS contract verifier for instruction-path rows. |

## Metric Contract

- Wall-clock: CUPTI Activity duration, CUDA-event duration, and their
  reconciliation gap.
- Cache prediction inputs: NVBit memory-address traces, static SASS memory
  features, shape metadata, and workload structure.
- Structure: static SASS features and NVBit dynamic instruction summaries.
- Phase cycles: optional NVBit-injected `%clock64` ledgers at conservative
  phase boundaries.
- Environment: read-only covariates for clock, thermal, power, and throttle
  state.
- Pairing: `dual_run_manifest.json` binds CUPTI and NVBit runs by kernel
  signature, binary hash, launch id, and covariate drift.

The harness does not read privileged hardware counters. It produces
counter-free evidence for measurement and prediction under the same non-root
cloud constraints as the target deployments.
<!-- HISTORICAL-D8-END -->
