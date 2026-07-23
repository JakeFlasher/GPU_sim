# task17 sm_86 Adaptive-Sampler Cross-Arch Runbook

This runbook is the operator-facing procedure for re-running the
adaptive-sampler clause's controller path against an sm_86
(RTX 3090 Ti) trace + upstream HRD bundle. The matching sm_120
evidence is captured during the loop's normal sweeps; this runbook
covers the cross-arch follow-up that promotes task17 from
`Verified-live-sm_120` to `Cross-arch-verified` per CLAUDE.md
"Two-Device Local Development Model".

## Source of Truth

This file lives at `tools/adaptive_sampler/TASK17_SM86_RUNBOOK.md`
on the `5060_dev` branch. Read it ONCE from `5060_dev` HEAD and
treat that version as authoritative for the entire procedure.
Re-reading the runbook from a worktree after `git checkout` may
return stale text from an earlier round (per
`BL-20260504-runbook-self-reference`). To re-read the latest
version without changing the worktree:

```bash
git show 5060_dev:tools/adaptive_sampler/TASK17_SM86_RUNBOOK.md
```

The companion verification target (the convergence sidecar +
verifier admission) is content-addressed by the live evidence,
not by the runbook text — operator output should match the
documented expected values regardless of which version of this
file the operator started with.

## § 1 Invocation

The adaptive-sampler clause's controller path is integrated into
`workloads/simbench_bw/run_suite.py::handle_trace`. There is no
separate operator binary to invoke; the controller path runs
automatically as part of the simbench `--mode trace` sweep:

```bash
# Drive the simbench trace sweep on sm_86 (the 3090 Ti card).
# Run from the repo root; the controller helper writes the
# convergence sidecar after the trace bundle is captured.
python workloads/simbench_bw/run_suite.py \
    --mode trace \
    --kernel dram_stream_stride_mix \
    --shape-point trace_00
```

The above command requires GPU hardware. Operators without an
attached card can verify the runbook's argv shape parses against
the live argparse contract by running the structural variant:

```bash
# Parser-accepted structural shape; exits cleanly without GPU.
python workloads/simbench_bw/run_suite.py \
    --mode correctness \
    --kernel dram_stream_stride_mix \
    --shape-point trace_00 \
    --dry-run-cpu
```

The structural variant is also what the C-AC-B
command-verification test
(`Task17RunSuiteDocumentedCommandTest`) drives; if the runbook's
`run_suite.py` argv shape ever drifts away from the live parser,
that test fails.

Alternatively the operator can invoke the offline CLI to
recompute the convergence bundle from an existing trace + HRD
bundle without running a fresh sweep:

```bash
python tools/adaptive_sampler/run_sampler.py \
    --hrd-bundle workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<TRACE>/hrd_profile/hrd_profile.json \
    --out workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<TRACE>/convergence_confidence/convergence_confidence.json \
    --arch sm_86 \
    --slug dram_stream_stride_mix \
    --shape-point trace_00 \
    --binary-sha256 <hex> \
    --source-commit <hex> \
    --kernel-signature-sha256 <hex> \
    --device-name "NVIDIA GeForce RTX 3090 Ti" \
    --cc-major 8 --cc-minor 6 \
    --sm-count 84 --l2-cache-size-bytes 6291456
```

R41 widened the CLI's `_SUPPORTED_ARCHES` tuple to admit both
`sm_86` and `sm_120-laptop` structurally so this runbook's
documented `--arch sm_86` shape parses against the live argparse
contract per `BL-20260504-runbook-command-verification`. Server
architectures (sm_80, sm_89, sm_90a, server sm_120) remain
rejected at argparse per the C-AC-E Loop 2 scope guard
(`Task17RunSamplerDocumentedCommandTest::test_documented_invocation_rejects_server_arch`
regression-locks this with `--arch sm_80`). The corresponding
`tools/arch_capabilities.json` entries already declare
`supported_arches: ["sm_86", "sm_120-laptop"]` for
`run_sampler.py`, `sampler.py`, and
`trace_convergence_ledger.py`. Structural admission does NOT
claim live cross-arch verification: the `Cross-arch-verified`
upgrade for sm_86 still requires a future 3090 Ti session and is
tracked in `tools/BLOCKED_ON_3090TI.md` task17 Section (e),
which remains sm_120-only live evidence in Loop 2.

## § 2 Expected HRD Input

The adaptive sampler consumes the upstream
`hrd_profile.json` produced by
`tools/hrd_profiler/bundle_from_trace.py`. The bundle's path is
fixed by the controller helper:

```text
<trace_dir>/hrd_profile/hrd_profile.json
```

Required upstream HRD bundle fields:

- `manifest_pairing_index`: contains `slug`, `shape_point`,
  `binary_sha256`, `source_commit`,
  `kernel_signature_sha256`. The convergence helper inherits
  this tuple verbatim — the convergence bundle's identity binds
  to the trace, not the converger's HEAD.
- `bins[]`: at least one admitted (`result_kind == "ok"`) bin
  per requested layer (`sector` and `line`). Each admitted bin
  declares `observation_count`, `hit_count`, `hit_rate`,
  `rd_sector_units_min`, `rd_sector_units_max` plus the
  bootstrap CI fields per
  `workloads/simbench_bw/schemas/hrd_profile.schema.json`.
- `bundle_underconfigured`: false if usable per-layer
  observations are present.

If sector or line layer has 0 admitted bins, the convergence
controller emits the
`hrd_layer_unavailable` typed-unavailable arm rather than
admitting `result_kind = ok`.

## § 3 Output Path

The convergence bundle is written to:

```text
<trace_dir>/convergence_confidence/convergence_confidence.json
```

The on-disk basename is `convergence_confidence.json` (NOT
`trace_convergence_confidence.json`) because the simbench
verifier scans `trace_*.json` for run summaries; using the
shorter basename avoids the rglob collision while preserving the
schema filename and the run-summary field name
`trace_convergence_confidence_path`.

The bundle strict-validates against
`workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`
(`additionalProperties: false` on every closed branch). Top-level
required fields: `schema_version`, `manifest_pairing_index`,
`device`, `hrd_profile_path`, `mrc_quantiles`, `distance_metric`,
`k_consecutive_windows`, `distance_threshold`,
`trace_window_size_bytes`, `windows`, `admission`,
`source_commit`. The `admission` `oneOf` enforces a strict
admitted branch (`{result_kind: ok, windows_converged,
final_residual}`) versus FIVE typed-unavailable branches keyed
by `typed_reason`:
`trace_convergence_not_reached_budget_cap` (plan-enumerated),
`mrc_quantile_underconfigured`, `hrd_layer_unavailable`,
`trace_convergence_disabled`, and
`trace_convergence_window_count_underconfigured` (round-added).

The corresponding 16 run-summary fields land in
`<trace_dir>/trace_<shape>.json` so the verifier can recompute
the cached fields against the on-disk bundle.

## § 4 Verifier Command

```bash
python workloads/simbench_bw/verify_artifacts.py \
    --results workloads/simbench_bw/l1/dram_stream_stride_mix/traces
```

Expected output: `ok` (exit 0).

The verifier exercises every rejection arm in
`_verify_trace_convergence_run_summaries`:
- missing `windows[]`;
- cached-vs-recomputed `windows_converged` disagreement;
- `result_kind=ok` with missing or out-of-range
  `final_residual`;
- `result_kind=ok` with fewer than `K=3` trailing converged
  windows;
- `trace_convergence_not_reached_budget_cap` without matching
  `trace_volume_exceeded.json` sidecar;
- cached `trace_convergence_mrc_quantile_set` disagreement
  with the canonical numeric set;
- any `admission.oneOf` branch carrying extra properties;
- non-baseline window with `distance_to_previous == null`;
- baseline window 0 with `distance_to_previous != null`;
- cached `trace_convergence_windows_digest` disagreeing with
  recomputed canonical-JSON digest;
- any cached `trace_convergence_manifest_<key>` disagreeing
  with `manifest_pairing_index.<key>` inside the sidecar OR
  with the enclosing trace run-summary's identity (the
  full 5-tuple: `slug`, `shape_point`, `binary_sha256`,
  `source_commit`, `kernel_signature_sha256`).

A cross-arch sm_86 follow-up succeeds when the verifier exits
`ok` against an sm_86 trace tree where the convergence sidecar's
`manifest_pairing_index` matches the enclosing run-summary's
identity tuple verbatim.

## § 5 Budget-Cap Handling

When the simbench trace hits the trace-volume budget cap, the
NVBit producer emits
`<trace_dir>/trace_volume_exceeded.json` per the C-AC-H
discipline. The convergence controller helper detects this
sidecar and emits the dual-sidecar binding:

- `admission.result_kind = "typed_unavailable"`
- `admission.typed_reason = "trace_convergence_not_reached_budget_cap"`
- `admission.trace_volume_exceeded_sidecar_path` references the
  matching budget sidecar (relative to the trace directory).

The verifier rejects any
`trace_convergence_not_reached_budget_cap` admission whose
referenced sidecar path is missing. The operator does NOT need
to take separate action: the controller path handles the
dual-sidecar binding automatically. If the operator wants to
manually inspect the budget evidence:

```bash
cat workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<TRACE>/trace_volume_exceeded.json
cat workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<TRACE>/convergence_confidence/convergence_confidence.json
```

The two sidecars share the `slug` and `shape_point` identity
fields and bind via the convergence sidecar's
`trace_volume_exceeded_sidecar_path`.

## Commit Lineage Reference

This runbook was authored in Loop 2 R38 alongside the
correctness rework of the R37 adaptive-sampler implementation
and updated in Loop 2 R39 to close the C-AC-B
command-verification gap. Per
`BL-20260504-runbook-command-verification`, every documented
repo CLI invocation shape is exercised by a positive +
negative-control fake integration test:

- `Task17RunSamplerDocumentedCommandTest` (in
  `tools/test_runbook_command_verification.py`) drives the
  `run_sampler.py` argv shape against schema-shaped HRD
  fixtures and asserts the typed-unavailable admission +
  argparse negative control;
- `Task17VerifyArtifactsDocumentedCommandTest` builds a
  minimal trace tree and asserts `verify_artifacts.py
  --results <trace_dir>` exits 0 + stdout `ok`, plus a
  negative control that drifts a cached manifest field and
  expects `manifest_run_summary_drift` in stderr;
- `Task17RunSuiteDocumentedCommandTest` confirms the
  documented `run_suite.py` argparser still parses;
- `Task17CatInspectionDocumentedCommandTest` regression-
  locks the `cat ...trace_volume_exceeded.json` and `cat
  ...convergence_confidence.json` literals in this runbook.

The companion controller-path tests in
`workloads/simbench_bw/tests/test_trace_convergence_run_summary_controller.py`
cover the in-process `_emit_trace_convergence_confidence_for_run_summary`
helper.

| Round | Commit | Note |
|-------|--------|------|
| R37   | 6bcf1133 | Initial task17 implementation; STALLED in review (HRD field-name bug, synthetic 4-window admission, missing typed-unavailable wiring, missing runbook). |
| R38   | ad035b0  | Correctness rework: HRD bin reader uses real schema fields, 5th typed reason added, controller wires all 5 typed-unavailable arms, manifest_pairing_index inherits from upstream HRD bundle, verifier checks all 5 identity fields, this runbook landed. |
| R39   | working-branch HEAD | C-AC-B closure + operator-CLI consistency: shared `is_usable_admitted_bin` adapter in `trace_convergence_ledger.py`; `run_sampler.py` rewritten to mirror the controller's 5-arm decision tree and emit `trace_convergence_window_count_underconfigured` for usable single-bundle input; task17 runbook command-verification tests landed; Track A Section (b) and `arch_capabilities.json` task17 notes refreshed. |

The verification target (the convergence sidecar admitting via
the verifier's `_verify_trace_convergence_run_summaries`) is
content-addressed by the live evidence at the round-closing
reviewed HEAD. Do not pin a literal SHA in the runbook's
operator instructions — the abstract-HEAD pattern keeps the
runbook stable across rounds.
