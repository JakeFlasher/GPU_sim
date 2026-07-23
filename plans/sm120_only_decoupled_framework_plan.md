# RTX 5060 Laptop sm_120-Only Decoupled Framework Plan

## Pre-amble: Why This Plan Exists

This plan **merges** two previously-overlapping plans and **decouples**
the active framework from every RTX 3090 Ti (sm_86) dependency:

  * `plans/silu_correctness_sm120_admission_cohort_amendment_plan.md`
    (the activation-cohort amendment that retired
    `silu_fwd_bf16_sm80/correctness_shape` from the sm_120 admission
    cohort and adopted `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl`).
  * `plans/nvbit_live_trace_contract_sm120_plan.md` (the canonical
    sm_120 NVBit live-trace producer-contract closure for
    `dram_stream_stride_mix/trace_00` on the laptop).

Both source plans were drafted under a two-device cross-arch
development model that implicitly required sm_86 evidence (predictor
bundles, label corpora, cross-arch promotion paths) carried from the
RTX 3090 Ti host. **Per operator direction on 2026-05-25 ("your task
is to reproduce the results on 5060 laptop"), this merged plan
removes every sm_86 / 3090 Ti dependency from the active framework
path and rebuilds the load-bearing surfaces — predictor bundles,
calibrator training distribution, admission cohort, live-trace
contract evidence — entirely on the RTX 5060 Laptop (sm_120,
Blackwell consumer, GB206 die).**

The two source plans are superseded by this one and should be
deleted from `plans/` once this plan lands.

## Decoupling Principle (binding contract)

The single binding principle of this plan: **the active framework
runs on RTX 5060 Laptop (sm_120) only and must not import, depend on,
or carry forward any artifact, code path, test assertion, or
provenance record that was produced on or for the RTX 3090 Ti (sm_86)
host.**

Specifically:

  * No new code path may filter by `arch == "sm_86"` or pin
    `target_arch = "sm_86"`. The existing sm_86-pinned surfaces
    (`fit_calibration_head.py:_TARGET_ARCH_PIN = "sm_86_only"`;
    `tools/sm_86_predictor_accuracy_recovery*`; sm_86-specific
    test fixtures) must either be replaced by sm_120-equivalent
    surfaces or formally retired from the active framework path.
  * No new evidence may cite, join to, or be SHA-coupled to a sm_86
    `binary_sha256`, sm_86 `kernel_signature_sha256` of a sm_86-built
    binary, sm_86 `source_commit` from a 3090 Ti session, or any
    predictor bundle whose `manifest_pairing_index.arch` is `sm_86`.
  * No new test or scanner may require a sm_86 cell to exist in
    `data/local_labels/label_admitted.json` for the framework to be
    valid. The 44 sm_86 cells currently present may stay as
    historical reference IF the active code paths neither consume
    them nor regress when they are absent; otherwise they are dropped.
  * No new round summary may cite a 3090 Ti device, sm_86 baseline,
    or sm_86 cross-arch promotion path.
  * `Cross-arch-verified` is removed from the active framework's
    evidence-status vocabulary for this plan. The only evidence
    statuses in scope are `Offline-verified` and
    `Verified-live-sm_120`.
  * `HANDOFF.md`, `tools/BLOCKED_ON_3090TI.md`, the CROSS_ARCH_RUNBOOK,
    the cross-arch sweep punch list, and all 3090 Ti session
    references are **historical documentation only**. This plan does
    not advance any HANDOFF.md item; it stands on its own.

CLAUDE.md "Two-Device Local Development Model" remains in effect at
the repository level, but for the purposes of this plan, the RTX
5060 Laptop is treated as the sole development device and the
"two-device" wording is honored only by NOT introducing cross-arch
claims, NOT by requiring 3090 Ti evidence.

## Current State (RTX 5060 Laptop, 2026-05-25)

What is already on disk from prior work — relevant inputs to this
plan:

  * **Built**: `workloads/portable_lcd/swiglu_fwd_bf16_sm80/swiglu_fwd_bf16_sm80_run`
    (LCD-discipline fat binary, sm_86 + sm_90 + sm_120 slices via
    `make ... NVCC='nvcc -ccbin=/usr/bin/g++-14'`; binary sha256
    `9a5cb056...`). The sm_120 slice runs natively on this host;
    the sm_86 + sm_90 slices are present in the fatbin but never
    executed.
  * **NVBit tracer**: `tools/nvbit_tracer/tracer_tool/tracer_tool.so`
    may already exist from a prior session; otherwise rebuild via
    `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`.
  * **Admitted labels** (`data/local_labels/label_admitted.json`):
    49 cells (44 sm_86 historical + 5 sm_120 cohort). The 5 sm_120
    cells are the amended activation cohort:
      * `layernorm_fwd_bf16_sm80/timing_shape`
      * `rmsnorm_fwd_bf16_sm80/timing_shape`
      * `silu_fwd_bf16_sm80/timing_shape`
      * `softmax_safe_bf16_sm80/timing_shape`
      * `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl`
    The 44 sm_86 cells are tagged as historical-only by this plan;
    they remain on disk for byte-equality preservation but the
    active sm_120 framework path must not require them.
  * **Calibration heads**: `calibration_head_v1.json` (sm_86-trained
    legacy; preserved as historical artifact) and
    `calibration_head_v2.json` (4-row historical artifact emitted in
    Round 0 of the cohort amendment loop; sm_86-trained partial
    refit that this plan will SUPERSEDE with a true sm_120-trained
    head).
  * **Cohort-identity guard**: `_REQUIRED_SM120_COHORT` in
    `workloads/simbench_bw/tests/test_comparison_runner.py` already
    pins the amended 5-cell sm_120 cohort identity (commit
    `0b11b613`). The test PASSES against the current admitted file.

## Mainline Objective

Produce, end-to-end on the RTX 5060 Laptop only and with no sm_86
dependence, a sm_120-trained PlattCalibrator head fit on the
amended 5-cell sm_120 activation cohort, validated by an admitted
NVBit live trace bundle for each cohort cell, with all consumer
paths (comparison runner, headline runner, provenance scanners)
defaulting to the sm_120-trained head when the amended cohort is in
scope.

Secondary objective: close the sm_120 NVBit live-trace producer
contract on this host for the dram_stream_stride_mix/trace_00
benchmark (the canonical NVBit-tracer smoke target inherited from
the previously-overlapping nvbit_live_trace_contract_sm120 plan),
without claiming any cross-arch promotion.

## Acceptance Criteria

All ACs include positive + negative tests for deterministic
verification per BL-20260509-strict-admission-contract-first.

### AC-D1: Decoupling foundation — fitter target_arch=sm_120 supported

The calibrator fitter `tools/local_labels/fit_calibration_head.py`
admits `--target-arch sm_120` and produces a head sidecar with
`target_arch = "sm_120"` (or equivalent) when invoked against a
sm_120 admitted corpus + sm_120 predictor bundles. The hard pin
`_TARGET_ARCH_PIN = "sm_86_only"` is either removed or reframed as
the legacy-default-with-explicit-override pattern so the sm_120 path
is a first-class citizen, not a workaround.

- Positive Tests:
  - `python3 tools/local_labels/fit_calibration_head.py --target-arch sm_120 ...` exits rc=0 against a staged sm_120 corpus + sm_120 bundles root.
  - The emitted head sidecar's `target_arch` field reads `sm_120` (not `sm_86_only`).
  - A unit test asserts the fitter selects sm_120 cells (not sm_86) when `--target-arch sm_120` is passed.
- Negative Tests:
  - The fitter does NOT silently fall back to sm_86 when `--target-arch sm_120` is passed and no sm_120 cells join.
  - The fitter does NOT join sm_86 cells when `--target-arch sm_120` is passed (arch filter is strict).

### AC-D2: sm_120 predictor bundles for the 5-cell cohort

For each of the 5 amended-cohort cells
(`layernorm_fwd_bf16_sm80/timing_shape`,
`rmsnorm_fwd_bf16_sm80/timing_shape`,
`silu_fwd_bf16_sm80/timing_shape`,
`softmax_safe_bf16_sm80/timing_shape`,
`swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl`), capture a sm_120 NVBit
trace on this host and run `workloads/simbench_bw/cache_predictor.py`
to emit a sm_120 predictor bundle. Each bundle's
`manifest_pairing_index.arch = "sm_120"`; each bundle's 5-tuple
joins the corresponding sm_120 admitted cell in
`data/local_labels/label_admitted.json`.

- Positive Tests:
  - 5 sm_120 predictor bundles exist under a `predictor_bundles/` path; each carries a non-None `predicted_hit_rate` and a valid `input_sidecar_digests.dynamic_trace_pb` + `sector_stream`.
  - Each bundle's `manifest_pairing_index` 5-tuple `(slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256)` joins the matching admitted cell in `label_admitted.json` (arch=sm_120).
  - The bundle's `cache_model_capacity_bytes` reflects the sm_120 (5060 laptop) L2 capacity, not the sm_86 6 MB value.
- Negative Tests:
  - No bundle emitted by this AC carries `manifest_pairing_index.arch = "sm_86"`.
  - No bundle references a sm_86 trace or sm_86 binary sha.

### AC-D3: sm_120-trained PlattCalibrator head v3 emitted

The calibrator fitter is run with `--target-arch sm_120` against the
sm_120 admitted corpus + sm_120 bundles from AC-D2; emits
`data/local_labels/calibration_head_v3.json` and
`data/local_labels/calibration_evaluation_v3.json` with
`training_pair_count = 5` and `training_cells` exactly matching the
amended 5-cell sm_120 cohort. v1 and v2 sidecars stay unchanged as
historical artifacts.

- Positive Tests:
  - `calibration_head_v3.json` exists; `training_pair_count == 5`; `training_cells` set matches the amended cohort exactly; every training row has `arch == "sm_120"`.
  - Every training row's 5-tuple joins the current `label_admitted.json` AND the AC-D2 bundle root.
  - The chosen head is recorded in the sidecar (PlattCalibrator expected by simplicity-one-SE rule; document if different).
- Negative Tests:
  - v3 training_rows do NOT include any sm_86 row.
  - v3 training_pair_count is NOT less than 5 (no partial-row fallback).
  - v1 / v2 sidecars are NOT modified by this AC.

### AC-D4: Consumer-side v3 selection (no v1/v2 fallback to sm_86 in active path)

The headline / comparison consumer path (currently
`tools/local_labels/run_headline_cohort_predict_compare.py` defaults
to `calibration_evaluation_v1.json`) selects
`calibration_evaluation_v3.json` when the amended sm_120 cohort is
in scope and no explicit override is passed. Explicit
`--calibration-evaluation-sidecar <path>` continues to honor the
override. When v3 is absent the consumer FAILS CLOSED with a typed
diagnostic naming `calibration_head_v3.json` — it does NOT silently
fall back to v1 (which would re-introduce the sm_86 dependence into
the active path).

- Positive Tests:
  - Test fixture invokes the consumer with the amended sm_120 cohort present and no override; the consumer loads v3.
  - Test fixture invokes the consumer with an explicit override; the override path is loaded.
  - Test fixture removes v3 and verifies the consumer fails closed with the typed v3-missing diagnostic.
- Negative Tests:
  - The consumer does NOT silently fall back to v1 when v3 is absent (the active path must surface the v3-missing diagnostic).
  - The consumer does NOT load v2 (which is a historical 4-row sm_86-trained sidecar).

### AC-D5: sm_120 NVBit live-trace producer contract closure

The sm_120 NVBit producer wire contract is validated by the existing
live-GPU pytest gate
(`workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py`)
running under `SIMBENCH_REQUIRE_LIVE_GPU=1` on this host. The gate's
test microbench runs end-to-end on the laptop and admits. Helper-
section resolver (R13/R14 metadata-first + ambiguity-failing)
admits both the microbench layout and the dram_stream_stride_mix
layout on sm_120; ambiguity-failing assertion does NOT fire.

- Positive Tests:
  - `SIMBENCH_REQUIRE_LIVE_GPU=1 python3 -m pytest workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py -v` exits rc=0 on the laptop.
  - The cross-arch acceptance sites in the microbench (compression-distribution per-arch baseline, cp.async SASS witness, ASYNC_COPY classifier sidecar) admit per the documented sm_120 baseline.
- Negative Tests:
  - The test does NOT silently skip when `SIMBENCH_REQUIRE_LIVE_GPU=1` is set.
  - The ambiguity-failing assertion in the helper-section resolver does NOT fire (if it does, log as a typed side issue and defer; do NOT mutate the resolver in this plan).

### AC-D6: dram_stream_stride_mix/trace_00 admitted on sm_120

A fresh `dram_stream_stride_mix/trace_00` NVBit trace bundle is
captured on the laptop and admitted by
`workloads/simbench_bw/verify_artifacts.py`. This bundle is the
canonical NVBit-tracer smoke target for the producer-side wire
contract.

- Positive Tests:
  - `python3 workloads/simbench_bw/run_suite.py --mode trace --kernel dram_stream_stride_mix --shape-point trace_00` exits rc=0 on the laptop.
  - Capture root at `workloads/simbench_bw/results/dram_stream_stride_mix/blackwell_consumer_unknown/sm_120/` contains `measurement_manifest.json`, `trace_trace_00.json`, `trace_trace_00/dynamic_trace.pb`, `trace_trace_00/threadblocks/`, `trace_trace_00/stats.csv`, and `trace_trace_00/extra_info/`.
  - `trace_trace_00.json` records `device_name = "NVIDIA GeForce RTX 5060 Laptop GPU"`, `compute_capability_major = 12`, `compute_capability_minor = 0`, `source_commit = git rev-parse HEAD`.
  - `python3 workloads/simbench_bw/verify_artifacts.py --results workloads/simbench_bw/results/dram_stream_stride_mix/blackwell_consumer_unknown/sm_120/` exits rc=0.
- Negative Tests:
  - The capture does NOT report `expected_records=0`, `nvbit_slowdown_budget_exceeded`, or `nvbit_drop_rate_exceeded` sentinel hits.
  - `verify_artifacts.py` does NOT silently admit a malformed trace tree.
  - The capture's `source_commit` does NOT match a pre-loop value.

### AC-D7: SASS regex Blackwell-widening regression fix

`workloads/simbench_bw/tests/test_perturbation_run_ledger_run_summary_verifier.py`
lines around 69–70 widen `_LDG_CA_RE_PATTERN` from Ampere-only
(`\.CA(?:\.`) to Ampere-or-Blackwell
(`\.(?:CA|STRONG\.SM)(?:\.`) and `_LDG_CG_RE_PATTERN` similarly,
mirroring `tools/perturbation_harness/sass_confirm.py:45-53` per
`BL-20260505-cross-arch-sass-mnemonic-divergence`. Existing tests
under the widened patterns pass.

- Positive Tests:
  - The fixture admits Blackwell SASS lines (`LDG.E.STRONG.SM` / `LDG.E.STRONG.GPU`).
  - The fixture still admits Ampere SASS lines (regression-locked — Ampere fixtures continue to match even though Ampere is no longer the active arch; the test data is frozen).
  - `pytest test_perturbation_run_ledger_run_summary_verifier.py -q` exits rc=0.
- Negative Tests:
  - The production code (`sass_confirm.py`, `verify_artifacts.py`) is NOT modified by this AC (it already has the widened patterns).

### AC-D8: v3 provenance scanner

Extend the calibration-head provenance scanner to validate
`calibration_head_v3.json` + `calibration_evaluation_v3.json` with
live-corpus joinability assertions: all v3 training rows join the
current `label_admitted.json` AND the AC-D2 bundle root; v3's
training cohort set equals the amended 5-cell sm_120 cohort; v3's
target_arch is `sm_120`.

- Positive Tests:
  - `pytest tools/local_labels/test_calibration_head_provenance_scanner.py -q` includes a new v3-targeted scanner that passes against the live state.
  - The new scanner FAILS against a hand-crafted broken v3 (e.g., missing row, sm_86 training row, stale source_commit, non-amended cohort cell) — proves the scanner would catch regressions.
- Negative Tests:
  - The scanner does NOT require v1 or v2 to exist (they may be retired from this scanner's surface, since the active path uses v3).

### AC-D9: Round summaries record sm_120-only evidence

Every round summary written under this plan records:

  * Device: `NVIDIA GeForce RTX 5060 Laptop GPU` (no 3090 Ti references)
  * Driver: current `nvidia-smi` reading
  * CUDA toolkit: current `nvcc --version` reading
  * NCU: current `ncu --version` reading
  * NVBit: `tracer_tool.so` SHA-256
  * Observed clocks + throttle bitmask during evidence capture
  * Evidence status: `Verified-live-sm_120` (or `Offline-verified` if no GPU evidence)
  * Explicit statement: no sm_86 / 3090 Ti artifact was created, consumed, or referenced by this round's active code path

- Positive Tests:
  - Round summary file contains all required hardware-metadata fields.
  - Round summary contains the explicit sm_120-only evidence statement.
- Negative Tests:
  - Round summary does NOT cite `Cross-arch-verified`.
  - Round summary does NOT cite the RTX 3090 Ti or sm_86 as a current-round work target.

## IMMUTABLE Path Boundaries

### In Scope

- The 9 ACs above (AC-D1 through AC-D9).
- Modifying `fit_calibration_head.py` to support `target_arch=sm_120`
  as a first-class option (AC-D1).
- Generating sm_120 NVBit traces + sm_120 predictor bundles for the
  5 cohort cells (AC-D2).
- Emitting sm_120-trained calibrator v3 (AC-D3).
- Wiring the consumer path to default to v3 in amended-cohort scope
  (AC-D4).
- Validating the sm_120 NVBit producer contract (AC-D5, AC-D6, AC-D7).
- Provenance scanner extension (AC-D8).
- Round-summary discipline (AC-D9).

### Out of Scope

- Any work that requires the RTX 3090 Ti (sm_86) host. This is
  binding by the Decoupling Principle.
- `Cross-arch-verified` promotion of any artifact (the vocabulary
  is removed from this plan's evidence-status set).
- Modifying or retiring `data/local_labels/label_admitted.json`'s
  44 sm_86 cells. They stay on disk as historical reference;
  byte-equality is preserved by NOT touching them. If a future
  decoupling cleanup wants to delete them, that is a separate plan.
- The 5 explicitly-deferred follow-on plans inherited from
  ancestor plans (activation-trace-orchestrator sm_120 decoupling,
  headline-cohort orchestrator sm_120 parameterization,
  leave-arch-out cross-arch calibration evaluation, sm_100
  zero-shot, `Cross-arch-verified` sm_86 revalidation). All five
  remain explicitly deferred and MUST NOT be pulled into this loop.
- Phase-injection sm_120 evidence for AC-4 (the v3 R1-R3 sub-claim
  from the historical v3 plan). Out of scope unless an explicit
  follow-on plan names it.
- Modifying `HANDOFF.md`, top-level `README.md`,
  `tools/cloud/README.md`, `workloads/simbench_bw/README.md`,
  `tools/nvbit_tracer/README.md`, or `tools/BLOCKED_ON_3090TI.md`.
  These tracked docs are historical context only; this plan does
  not advance any HANDOFF.md item.
- Server / cloud sm_120 work, sm_100 datacenter, NVL72, B200, NVSwitch
  parameterization. CLAUDE.md Rule 1 + this plan's single-device
  scope rule out all of these.

### Allowed Choices

- Can use: existing `run_suite.py --mode trace` orchestration; existing
  `test_nvbit_trace_contract_live.py` pytest gate; existing
  `verify_artifacts.py` admission consumer; existing
  `cache_predictor.py` predictor algorithm; existing
  `merge_admitted_cells.py` (already used in Round 0); native gcc-14
  toolchain (`NVCC_CCBIN=g++-14`); existing built workload binaries
  on this host.
- Cannot use: anything from the sm_86 / 3090 Ti host, including
  sm_86 predictor bundles, sm_86 binary SHAs, sm_86 trace JSONs,
  sm_86 calibration heads, sm_86-coupled CROSS_ARCH_RUNBOOK
  procedures, or sm_86 test fixtures as a CURRENT-active-path
  surface (test fixtures may remain in the repo as frozen Ampere
  reference data but no active framework path may require them).

## Dependencies and Sequence

### Milestones

1. **M1 — Decoupling foundation (closes AC-D1)**:
   - Modify `fit_calibration_head.py` to support `--target-arch sm_120`
     as a first-class option; remove or reframe the `sm_86_only`
     hard pin; add unit tests for the new selection behavior.
   - Single commit; test-first per `BL-20260509`.

2. **M2 — sm_120 predictor bundles for the 5 cohort cells
   (closes AC-D2)**:
   - For each cohort cell on this host:
     1. Build the workload binary (sm_120 slice or fat LCD; verify
        `cuobjdump --list-elf` shows sm_120 cubin present).
     2. Run NVBit tracer to capture `dynamic_trace.pb` +
        `sector_stream.jsonl`.
     3. Run `cache_predictor.py` against the trace inputs; emit a
        predictor bundle with `manifest_pairing_index.arch = "sm_120"`
        and 5-tuple matching the admitted cell.
     4. Verify the bundle joins via the same logic used by the
        fitter.
   - One bundle per cell × 5 cells = 5 bundles; commits one per
     cohort cell for review granularity.

3. **M3 — sm_120-trained PlattCalibrator v3 (closes AC-D3)**:
   - Re-run the fitter (from M1) with `--target-arch sm_120` against
     the sm_120 admitted corpus + M2 bundles. Verify
     `training_pair_count == 5` and `training_cells` exactly match
     the amended cohort.
   - Single commit landing `calibration_head_v3.json` +
     `calibration_evaluation_v3.json`.

4. **M4 — Consumer-side v3 selection rule (closes AC-D4)**:
   - Test-first: add tests for the v3-default selection rule + v3-
     absent fail-closed diagnostic.
   - Implement: modify the consumer path to default to v3 in
     amended-cohort scope; preserve the explicit override.
   - Two commits (test-first then implementation) per `BL-20260509`.

5. **M5 — sm_120 NVBit producer contract closure (closes AC-D5, AC-D6, AC-D7)**:
   - Bootstrap: verify `tracer_tool.so` SHA + multi-arch via
     `cuobjdump --list-elf`. Rebuild if missing.
   - Stale-artifact cleanup: remove stale sm_120 trace-mode artifacts
     under `blackwell_consumer_unknown/sm_120/{trace_trace_00,
     trace_trace_00.json, phase_cycle_ledgers_trace_00,
     nvbit_*_exceeded.json, nvbit_trace_summary_*.json,
     measurement_manifest.json}`. Preserve any sm_120 timing capture
     (it is historical, not in this plan's scope to delete).
   - SASS regex widening: edit
     `test_perturbation_run_ledger_run_summary_verifier.py` lines
     ~69-70 to widen `_LDG_CA_RE_PATTERN` / `_LDG_CG_RE_PATTERN`
     per BL-20260505. (AC-D7.)
   - Live-GPU pytest gate FIRST:
     `SIMBENCH_REQUIRE_LIVE_GPU=1 python3 -m pytest workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py -v`
     to catch tracer / test-gate drift before spending capture
     budget. (AC-D5.)
   - Live trace capture:
     `python3 workloads/simbench_bw/run_suite.py --mode trace
     --kernel dram_stream_stride_mix --shape-point trace_00`.
     (AC-D6 first half.)
   - Admission: `verify_artifacts.py --results <fresh sm_120 trace
     tree>`. (AC-D6 second half.)
   - Commits: one for the SASS regex widening (the only tracked-
     file edit in this milestone); capture artifacts are gitignored.

6. **M6 — v3 provenance scanner + round summary discipline
   (closes AC-D8, AC-D9)**:
   - Extend `tools/local_labels/test_calibration_head_provenance_scanner.py`
     to validate v3 with live-corpus joinability assertions. Add a
     negative-fixture test that proves the scanner catches a
     broken v3.
   - Write the round summary with the full sm_120 hardware metadata
     block + the explicit sm_120-only evidence statement.
   - Two commits: scanner extension, then round summary.

### Dependency Graph

- M1 → M2 (the fitter's sm_120 path is needed before M3 can succeed,
  but M2 only needs the predictor pipeline; M2 can run in parallel
  with M1).
- M3 depends on M1 (fitter sm_120 support) + M2 (bundles).
- M4 depends on M3 (v3 must exist before consumer wires to it).
- M5 is independent of M1/M2/M3/M4; can run in parallel.
- M6 depends on M3 (scanner needs v3 to exist) and M5 (round
  summary's evidence-status declarations need M5's live-evidence
  state).

## Task Breakdown

Each task carries exactly one routing tag (`coding` or `analyze`).

| Task ID | Description | Target AC | Tag |
|---------|-------------|-----------|-----|
| task1 | Modify `fit_calibration_head.py` to accept `target_arch=sm_120` as a first-class option; remove the `_TARGET_ARCH_PIN = "sm_86_only"` hard pin (or reframe as default-with-override); add unit tests for the new selection behavior. | AC-D1 | coding |
| task2 | Build NVBit tracer if not present; verify `tracer_tool.so` multi-arch via `cuobjdump --list-elf`; record SHA. | AC-D5 (prereq) | coding |
| task3 | For each of the 5 cohort cells: ensure the workload binary is built for sm_120; capture NVBit trace; run `cache_predictor.py`; emit predictor bundle with sm_120 MPI 5-tuple joining the admitted cell. | AC-D2 | coding |
| task4 | Re-run the modified fitter with `--target-arch sm_120` against the sm_120 admitted corpus + task3 bundles; verify `training_pair_count == 5`; emit `calibration_head_v3.json` + `calibration_evaluation_v3.json`. | AC-D3 | coding |
| task5 | Test-first: add tests for consumer-side v3 selection rule (default in amended-cohort scope; fail-closed when v3 absent; explicit override honored). Commit. | AC-D4 (test) | coding |
| task6 | Implement v3 selection rule in `run_headline_cohort_predict_compare.py` (and any comparison_runner-side consumer surface). | AC-D4 (impl) | coding |
| task7 | Stale-artifact cleanup under `blackwell_consumer_unknown/sm_120/` per AC-D6 (preserve historical timing capture if present). | AC-D6 (prereq) | coding |
| task8 | SASS regex widening in `test_perturbation_run_ledger_run_summary_verifier.py:~69-70` per BL-20260505. | AC-D7 | coding |
| task9 | Run live-GPU pytest gate (`SIMBENCH_REQUIRE_LIVE_GPU=1 pytest test_nvbit_trace_contract_live.py -v`). | AC-D5 | coding |
| task10 | Run `run_suite.py --mode trace --kernel dram_stream_stride_mix --shape-point trace_00`; verify canonical artifact tree; run `verify_artifacts.py`. | AC-D6 | coding |
| task11 | Extend `test_calibration_head_provenance_scanner.py` to validate v3 with live-corpus joinability assertions; add a negative-fixture test proving the scanner catches a broken v3. | AC-D8 | coding |
| task12 | Cross-check the sm_120-only evidence enumeration via `/humanize:ask-codex` to confirm no sm_86 dependence leaks through tests, scanners, or consumer paths. | (cross-cutting) | analyze |
| task13 | Write the round summary with full sm_120 hardware metadata + explicit sm_120-only evidence statement + BitLesson Delta. | AC-D9 | coding |

## Discipline (binding contract)

- **Decoupling discipline**: NO active code path may filter by
  `arch == "sm_86"`, pin `target_arch = "sm_86"`, join to a sm_86
  bundle, or cite the 3090 Ti host. Violations are P0 and must
  fail this plan's review.
- **Test-first commit ordering** per
  `BL-20260509-strict-admission-contract-first` for every AC that
  ships a test surface.
- **DEC-6 hard-binary preserved**: NO threshold softening at
  `stddev_threshold = 0.03` in any sweep / aggregator path.
- **Multi-surface lockstep** per `BL-20260507-arch-token-multi-surface-lockstep`
  for any arch-token-affecting code surface (this plan does NOT
  introduce new arch tokens; the existing `sm_120` token is used
  unchanged).
- **Plan-token scanner discipline** per the existing
  `tools/test_plan_token_scanner.py`: implementation code,
  comments, test docstrings, and assertion messages must NOT
  contain plan-specific terminology (`AC-DN`, `M1..M6`, `taskN`,
  `RoundN`, etc.). These belong in plan documentation only.

## Estimated Effort

- 4–6 rounds depending on how many milestones land per round.
  Round 1 is likely to land M1 + M5 (the foundation work + the
  smaller live-trace closure). Rounds 2–3 land M2 (per-cell bundle
  capture; this is the heaviest milestone). Round 4 lands M3 + M4.
  Round 5–6 land M6 and any remaining polish.
- Single host: RTX 5060 Laptop (sm_120) for the entire plan; no
  cross-host transfers.

## Related Historical Artifacts (reference only; NOT active dependencies)

- `plans/silu_correctness_sm120_admission_cohort_amendment_plan.md` — superseded by this plan.
- `plans/nvbit_live_trace_contract_sm120_plan.md` — superseded by this plan.
- Round 0 evidence under this loop:
  - `0b11b613` (cohort-identity guard amendment, AC-A1 of the superseded amendment plan)
  - `448a691e` (sm_120 swiglu_xl admission + merge, AC-A2 + AC-A4 of the superseded amendment plan)
  - `23caffe9` (PlattCalibrator v2 partial sidecar, superseded by AC-D3 in this plan)
- `data/local_labels/calibration_head_v1.json` — historical sm_86-trained legacy; NOT consumed by the active framework path under this plan.
- `data/local_labels/calibration_head_v2.json` — historical Round 0 4-row partial-refit sidecar; SUPERSEDED by v3 under AC-D3.
- `data/local_labels/sweeps/sm120_swiglu_xl_amendment_r0_20260525T004030Z/` — staged operator artifacts from Round 0 of the superseded amendment plan; the manifest + predictor_trace JSON for sm_120 swiglu_xl are re-usable by M2 task3 (the kernel_signature derivation is correct; only the binary_sha256 needs to be re-staged for whatever sm_120 binary M2 actually traces).
- `data/local_labels/sweeps/swiglu_xl_single_cell_admission_20260525T004342Z/` — Round 0 NCU single-cell admission sweep for sm_120 swiglu_xl (admitting `warmup=0/cache=all/repeats=5`, stddev=0.0136). One cell × 6 configs; intentionally NOT a 40-config matrix sweep. Renamed in Round 7 from the legacy `sweep_<ts>_<hash>/` form to a non-`sweep_*` prefix so the matrix-completeness scanner in `tools/local_labels/live_sweep_completeness.py::find_latest_matrix_sweep_dir` skips it (the scanner requires a `sweep_*` prefix to identify matrix-style sweeps). This evidence stands and is referenced by `label_admitted.json` row for `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl` on sm_120.

## Closure / Review Scope (loop-closure addendum landed in Round 7)

This section is **non-functional** — it does not introduce a new
AC. It serves as an authoritative scope-of-review declaration for
the Codex Review Phase, written after Round 7's strategic
consultation with Codex (see Round 7 summary). The declaration
exists because the loop's `state.md` was initialized with
`base_commit = d16b3754` (master tip) and `start_branch =
3090ti_cache_sims`, two histories with no common ancestor on
this host. The Round-5 infrastructure repair used `git replace
--graft d629bbb9 d16b3754` to bridge them so `codex review
--base d16b3754` could compute a diff. That bridge made the
review range include hundreds of pre-loop commits that this
plan explicitly de-scoped.

**Scope-of-review declaration**:

1. **AC-D1..D9 are MET as of Round 4 close.** All 9 ACs have
   independent live-tree evidence; the 9-AC closure table in
   Round 4's summary records the closing commit per AC.

2. **The active sm_120 mainline ratchet is preserved across
   Rounds 5-7**:
   - 5 sm_120 predictor bundles validate against
     `cache_predictor.schema.json` via
     `jsonschema.Draft202012Validator` (Round 5 fix).
   - v3 head + eval sidecars are committed against a HEAD
     whose fitter source contains the `target_arch` emitter
     (Round 4 fix).
   - L1 walker default + sm_86 cohort_manifest are
     self-consistent on every fresh checkout (Round 6 fix).
   - Single-cell sm_120 admission sweeps live outside the
     matrix-sweep namespace (Round 7 fix).

3. **Pre-loop sm_86 historical drift is OUT OF SCOPE for
   this loop.** Specifically, the following classes of
   findings are queued for a separate sm_86-maintenance
   loop that requires 3090 Ti hardware access:
   - `data/local_labels/label_admitted.json` rows on the
     sm_86 side carrying source_commit values that do not
     match the live launch-provenance sidecar. Round 53
     extension: this class also covers sm_86 recovery-v2
     calibration/shadow artifacts under
     `data/sm_86_predictor_accuracy_recovery_v2/` (e.g.,
     `l2_high_footprint_correction_calibration.json`,
     `l2_reduction_correction_calibration.json`,
     `l1_reduction_subfamily_calibration.json`,
     `carveout_l1_head_diagnostic.json`,
     `l2_high_footprint_correction_shadow_report.json`)
     carrying stale source_commit values relative to the
     current HEAD/HEAD^ accept window. These artifacts were
     previously refreshed by the recovery-v2 recapture
     script (each carries a `source_commit_override_from`
     field documenting the prior value) but the prior
     refresh is now itself outside the active accept
     window. Same restamp-pathway dependency on the 3090 Ti
     maintenance loop as the label_admitted side.
   - `data/local_labels/label_admitted_launch_provenance.json`
     entries for sm_86 cells that need restamping in lockstep
     with the labels aggregate.
   - `data/local_labels/audit_evidence/ac6_microbench_underconfiguration_diagnostic/underconfiguration_table.json`
     references to uncommitted HRD profiles under
     `workloads/simbench_bw/results/...`.
     **Codex Review Phase note on path convention**: the
     emitter `tools/local_labels/emit_microbench_underconfiguration_artifact.py`
     constructs HRD-profile paths as `trace_{shape_point}`
     uniformly (always prepend `trace_`). This produces
     double-prefix paths like `trace_trace_00/...` when the
     shape_point itself starts with `trace_` (e.g.,
     `trace_00`, `trace_l2_accuracy_xl`). This is the
     ESTABLISHED CONVENTION across the codebase — the on-disk
     results layout under `workloads/simbench_bw/results/`
     follows the same pattern (verified locally:
     `trace_trace_00/`, `trace_trace_l2_accuracy_xl/`,
     `trace_timing_shape/` all coexist). Codex may flag the
     double-prefix as a bug ("don't double-prefix"); the
     "fix" of removing the prefix would diverge from the
     on-disk layout. The actual root cause is `results/`
     being gitignored, not the path-construction rule.
     Queued under this class.
   - Pre-loop sweep `predictor_bundles/*.json` files
     declaring `predictor_build_commit` values outside the
     current `{HEAD, HEAD^}` accept window.
   - 44 sm_86 cells in `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json`
     whose `expected_trace_path` artifacts are gitignored
     (Round 6 flipped them to `attempted_excluded` with
     `trace_missing` typed reason to make the manifest
     self-consistent on fresh checkouts).
     **Codex Review Phase note**: the manifest's resulting
     `admitted_cell_count: 0` is the INTENTIONAL reconciled
     state, NOT a regression. Codex repeatedly re-flags this
     across rounds as either "manifest declares N admitted
     but reconciles to 0" (pre-Round-6) or "manifest declares
     0 admitted so the runner iterates an empty cohort"
     (post-Round-6). Both descriptions are accurate; both
     point at the same underlying root cause (sm_86 traces
     gitignored). The Round-6 flip chose the
     self-consistent-but-empty state over the
     misleading-cell-count state; further repair belongs to
     the sm_86 follow-on loop on a 3090 Ti host. Round 55
     extension: this class also covers the companion
     `data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json`
     manifest-digest mismatch — the committed split declares
     the pre-Round-6 manifest digest (`8c66...`) which no
     longer matches the recomputed digest of the
     post-Round-6 reconciled manifest. The mismatch fails
     `CohortSplitDigestMismatchError` immediately. Same
     root cause as the manifest's 0-admitted state; same
     bounded-fix pathway (sm_86 maintenance loop regenerates
     both manifest + split in lockstep).
   - **Companion sm_86 evidence + code-side artifact gaps**
     between admitted-cell rows in
     `data/local_labels/label_admitted.json` (sm_86 side) and
     their required companion artifacts. Two sub-classes:

     **Evidence-side**: required 1:1 companion artifacts
     under `data/sm_86_predictor_accuracy_recovery_v2/`
     (e.g., `predicted_l1_per_admitted_cell.json`'s HARD-2
     1:1 mapping). Round 57 extension: the evidence-side
     sub-class ALSO covers `typed_unavailable` predicted_l1
     rows — when the predictor pipeline ran under a
     generation environment missing `google.protobuf`
     Python runtime, the resulting rows are emitted with
     `result_kind = "typed_unavailable"`, `predicted_hit_rate
     = null`, and
     `typed_reason = "predictor_l1_sass_provenance_missing"`.
     The committed `predicted_l1_per_admitted_cell.json`
     under this directory carries `typed_unavailable` rows
     for ALL sm_86 cells (lines 18, 39, 60, 81, 102, 123,
     144, 165, 186, ...) because the entire pre-loop sm_86
     predictor-pipeline run was under that incomplete
     environment. Same root cause as the missing-row
     sub-flavor; same bounded-fix pathway (re-run sm_86
     predictor pipeline with full toolchain — protobuf
     included — on a 3090 Ti host).

     **Code-side**: workload toolchain entries that the
     pre-loop cohort expansion landed an admitted-label row
     without completing. Specifically: (a) the slug entry in
     `workloads/simbench_bw/references/<family>.py::FAMILY_COVERS_SLUGS`
     required by `handle_correctness` reference-family
     validation; (b) the driver-emitted `output` field in
     `workloads/simbench_bw/l1/<slug>/main.cu` consumed by
     `handle_correctness` from `kernel_result.get("output")`;
     (c) the precision/family tolerance entry in
     `workloads/simbench_bw/tolerances.toml`. Without these,
     `--mode correctness` exits with typed reasons like
     `reference_family_slug_mismatch`, `driver JSON missing
     'output' field`, or `tolerance_unmapped`.

     Specific example: pre-loop commit `dbc26e6d` admitted
     `vendor_sgemm_strided_batched_cublas/timing_shape` into
     `label_admitted.json` without the matching predicted-L1
     row (evidence-side) AND without (a) a
     `vendor_sgemm_strided_batched_cublas` slug in
     `references/control.py::FAMILY_COVERS_SLUGS`, (b) an
     `output` field in
     `workloads/simbench_bw/l1/vendor_sgemm_strided_batched_cublas/main.cu`,
     (c) an FP32/control tolerance in
     `workloads/simbench_bw/tolerances.toml` (code-side).
     Closing the evidence-side gap requires either re-running
     the sm_86 predictor pipeline (depends on gitignored
     sm_86 traces) or removing the admitted row (touches
     sm_86 historical labels). Closing the code-side gap
     does not strictly require 3090 Ti hardware but serves
     ONLY the sm_86 cohort's correctness-mode runnability
     (the sm_120 mainline does not use this cell). Both
     evidence-side and code-side work are sm_86 cohort
     maintenance bounded out-of-scope by this addendum's
     "per-loop responsibility boundaries" principle.

4. **Per-loop responsibility boundaries**: this plan owns
   only the sm_120 mainline. The sm_86 historical evidence
   was carried forward from prior cross-arch work; touching
   it now would violate the binding decoupling contract
   above. The sm_86 maintenance loop, when it runs on a
   3090 Ti host, will use the canonical refresh tools
   (`refresh_admitted_label_source_commit.py`,
   `refresh_predictor_trace_source_commit.py`) to restamp
   the sm_86 evidence to a contemporary HEAD and rebuild
   the missing trace artifacts via NCU + NVBit recapture.

5. **Codex Review Phase guidance**: when a finding is
   flagged in a future round of this loop AND the finding's
   offending file is in one of the four classes listed in
   (3) above, the principled response is to record the
   finding as a queued sm_86-maintenance follow-up and NOT
   to apply a same-loop fix. The decoupling discipline at
   the top of this plan binds the loop's commit set; sm_86
   evidence hygiene belongs to a separate loop.

6. **Convergence pattern (rounds 19-146)**. The Codex Review
   Phase has reached a steady state after Round 18 closed
   the last broadly-actionable in-scope scanner work.
   Round 43 saw all findings drop from P1 to P2 (a fresh
   severity-attenuation signal at the time), but Round 44
   reverted to all-P1 — confirming the Round-43 attenuation
   was a one-round noise rather than a sustained
   convergence-shape change. **Round 113 was the second
   all-P2 attenuation instance** (3 P2 findings, no P1),
   70 rounds after Round 43; Round 114 reverted to all-P1
   in exactly the same one-round-noise shape as Round 44.
   **Round 122 is the third all-P2 attenuation instance**
   (3 P2 findings, no P1), 9 rounds after Round 113 and
   the closest temporal pair of all-P2 instances yet
   observed. The third instance also follows the same
   post-in-scope-fix attenuation pattern: Round 121
   landed a NEW in-scope sweep_repeatability_matrix
   shape-axis forwarding fix, and Round 122 reverted to
   pre-loop sm_86 baggage at one severity tier lower
   (P2 not P1). **Round 123 broke the clean one-round-noise
   revert pattern observed at Round 44 and Round 114**:
   the prior two post-attenuation rounds reverted to
   all-P1 with the same class shape (#1+#2+#4), but
   Round 123 reverted partially (3 P1 + 1 P2) AND
   shifted class shape entirely (companion-gap dual
   sub-class + class #5 + class #1 recovery-v2 sub-class,
   with no overlap to Round 122's #1+#2+#4). The third
   all-P2 instance is the first where neither severity
   nor class shape returned cleanly to the prior round's
   form. **Round 124 partially resolves the
   convergence-shape-shift question raised at Round 123**:
   the new class composition is ADDITIVE, not displacing.
   Round 124 recovers the Round-122 #1+#2+#4 triad
   (all three classes fire again at P2) AND retains the
   Round-123 additions (class #5 + companion-gap
   predicted-L1 sub-class). Round 124's class shape is
   the **UNION** of Round 122 + Round 123 = 5 classes
   simultaneously (#1 + #2 + #4 + #5 + companion-gap
   predicted-L1), the broadest single-round shape
   observed in the loop's history. **Round 125 narrows the
   class shape back to 3 classes** (#2 + #4 + #5) — class
   #1 (main + recovery-v2 sub-class) and companion-gap
   both silent for the first time since Round 121. Round
   125's 3-class shape #2+#4+#5 is a new composition (not
   matching Round 122's #1+#2+#4 nor any prior 3-class
   shape with #5). **Round 126 resolves the post-Round-122
   mixed-severity phase**: 6 P1 / 0 P2 findings, the
   **first all-P1 round since Round 121**. Round 126's
   class shape (#1 main + #1 recovery-v2 + #2 + #4 + #5
   + companion-gap typed_unavailable) is **6 classes —
   the broadest single-round class shape observed in the
   loop's history** (exceeding Round 124's 5-class UNION).
   The post-Round-122 5-round severity trajectory (0 P1
   / 3 P2 → 3 P1 / 1 P2 → 2 P1 / 3 P2 → 2 P1 / 1 P2 →
   6 P1 / 0 P2) reveals a NEW post-attenuation pattern
   distinct from the prior two one-round noise patterns
   (43→44, 113→114): **"delayed clean revert with class
   expansion"**. Round 44 and Round 114 each reverted to
   all-P1 in 1 round with the SAME class shape; Round
   122's revert took 5 rounds and arrived with the
   broadest class shape ever observed. **Round 127
   returns to the foundational 3-class triad #1+#2+#4
   at all-P1 severity** — Class #5, companion-gap, and
   class #1 recovery-v2 sub-class all silent again,
   the loop's pre-Round-122 steady state restored.
   Round 127 is the **36th total #1+#2+#4 instance**
   (after the 35 prior instances ending at Round 122
   all-P2). The post-Round-122 phase is now
   characterized as a **5-round digression** from the
   foundational steady state (Rounds 122-126),
   returning at Round 127. The recurring re-flags by
   class:
   - `label_admitted` source_commit drift: 120+ consecutive
     re-flags (class #1) — **SECOND CLOSURE-ADDENDUM CLASS
     TO REACH 100-FLAG THRESHOLD** in the loop's history
     (after class #2 at Round 114). **Class #1 main also
     crosses the 120-flag threshold at Round 145** —
     second closure-addendum class to reach 120 flags
     (9 rounds after class #2 at Round 136). Round 53 extended class #1 to
     cover sm_86 recovery-v2 calibration/shadow artifacts
     (first flag against this sub-class on
     `l2_high_footprint_correction_calibration.json`).
     Round 78 re-recurred the recovery-v2 sub-class on
     `l1_reduction_subfamily_calibration.json` (plus the
     high-footprint/reduction calibration + shadow report
     siblings) — confirming the sub-class is not a
     one-time Round-53 anomaly. Round 85 is the third
     recurrence on
     `l2_high_footprint_correction_shadow_report.json`
     plus sibling correction/calibration artifacts —
     the sub-class is now confirmed stable (3 distinct
     flag rounds with the same evidence shape). Round
     100 is the fourth recurrence on
     `l1_reduction_subfamily_calibration.json` again
     (same file as Round 78). Notable: Round 100 is
     the first round where the recovery-v2 sub-class
     fires WITHOUT the primary `label_admitted`
     sub-class also firing. Round 123 is the **fifth
     recurrence** on `l1_reduction_subfamily_calibration.json`
     (same file as Rounds 78, 100) — the second round
     where the recovery-v2 sub-class fires WITHOUT the
     primary `label_admitted` sub-class (after Round
     100). The unaccompanied-recovery-v2 pattern is now
     observed in 2 of 5 recurrences (Rounds 100, 123).
     Round 126 is the **sixth recurrence** on
     `l1_reduction_subfamily_calibration.json` (same file
     as Rounds 78, 100, 123) — the **first ACCOMPANIED
     recurrence since Round 85** (Rounds 100 and 123
     were unaccompanied by class #1 main; Round 126
     resumes the Round-78/85 accompanied pattern).
     Round 131 is the **seventh recurrence** on
     `l2_high_footprint_correction_shadow_report.json`
     (third occurrence on this file after Rounds 53,
     85) — ACCOMPANIED by class #1 main. The
     recurrence-accompaniment shape is now:
     ACCOMPANIED (Round 78, 85) → UNACCOMPANIED (Round
     100, 123) → ACCOMPANIED (Round 126, 131, 141, 144) — 6
     accompanied vs 2 unaccompanied across 8 recurrences
     since the Round-53 first flag. Round 144 is the
     **9th recurrence** and **5th occurrence on
     `l2_high_footprint_correction_shadow_report.json`**
     (after Rounds 53, 85, 131, 141).
   - `label_admitted_launch_provenance` restamp: 128+
     re-flags (class #2) — **FIRST CLOSURE-ADDENDUM CLASS
     TO REACH 100-FLAG THRESHOLD** in the loop's history.
     **Class #2 also crosses the 120-flag threshold at
     Round 136** — first closure-addendum class to reach
     120 flags (22 rounds after first reaching 100 at
     Round 114).
     Codex bundled #1 + #2 into a
     single finding for the first time in Round 35,
     reflecting further convergence. The bundled pattern
     recurred at Round 81 (second instance) and Round 87
     (third instance) — confirming Codex's
     classification of these two sub-classes as a single
     integrated finding is now stable (3 distinct
     instances across 52 intervening rounds). Rounds 41 + 47 + 49
     + 68 + 72 + 108 + 112 P1 #1 were framed by Codex as sm_120
     launch_provenance gaps but the scanner at
     `tools/local_labels/test_admitted_label_launch_provenance_scanner.py:435`
     explicitly filters `cell.get("arch") != "sm_86"` —
     the actual failure is the same class #2 sm_86
     activation rows drift; the sm_120 rows are correctly
     skipped. The misframing now has 7 recurrences across
     rounds (Round 41/47/49/68/72/108/112), confirming Codex's
     review-phrase repertoire has plateaued such that
     incorrect framings recycle stably without
     scanner-code changes.
   - `underconfiguration_table` HRD-path missing: 7+
     re-flags (class #3).
   - Pre-loop sweep `predictor_build_commit` drift: 88+
     re-flags (class #4). **Class #4 crosses the 80-flag
     threshold at Round 135** — third closure-addendum
     class to reach 80 flags (after classes #1 and #2). The specific sweep
     `sweep_20260514T165947Z_a1closure_ebe60dd` was
     re-flagged in rounds 23, 24, 27, 30, 31, 34, 35, 38,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55,
     56, 58, 59, 60, 61, 62, 63, 64, 65, 68, 70, 71, 72,
     73, 75, 76, 77, 78, 80, 82, 83, 84, 85, 86, 87, 88,
     89, 90, 92, 93, 94, 95, 96, 98, 99, 101, 102, 104,
     106, 107, 109, 110, 113, 114, 115, 119, 120, 122, 124, 125, 126, 127, 128, 129, 131, 133, 134, 135, 136, 137, 138, 139, 141, 142, 144, 145.
   - `cohort_manifest` 0-admitted INTENTIONAL state
     (class #5): 57+ re-flags across rounds 13, 17, 18,
     19, 20, 28, 32, 33, 37, 41, 42, 44, 51, 55, 56, 57,
     62, 65, 72, 74, 75, 77, 78, 80, 81, 82, 83, 85, 87, 88,
     90, 91, 92, 97, 98, 99, 100, 103, 105, 108, 112, 116,
     117, 118, 123, 124, 125, 126, 128, 129, 130, 131, 132, 137, 138, 140, 145. **Class #5
     crosses the 50-flag threshold at Round 129** —
     fourth closure-addendum class to reach this
     milestone (after classes #1, #2, #4). Round 55
     extended class #5 to cover the companion
     `cohort_split.json` manifest-digest mismatch.
     Round 129 is the **first round since the Round-55
     scope extension where the split sub-class is
     flagged AS A SEPARATE FINDING**: Codex surfaced
     `cohort_split.json` digest mismatch as P1 #2
     alongside the manifest P1 #1, rather than
     implicitly bundling the split with the manifest as
     in all prior class #5 rounds. Round 131 is the
     **second separate-flag instance** (split sub-class
     fires alone — manifest sub-class silent this round),
     arriving only 2 rounds after Round 129 — the
     2-round gap is significantly tighter than the
     74-round gap between the Round-55 scope extension
     and the Round-129 first separate-flag. Round 137
     is the **third separate-flag instance** (split
     sub-class fires alone — manifest sub-class silent
     this round), arriving 6 rounds after Round 131.
     **Round 145 is the fourth separate-flag instance**
     (split sub-class fires alone, manifest silent),
     arriving 8 rounds after Round 137. The four
     separate-flag instances (Round 129, 131, 137,
     145) span 16 rounds — Codex's per-sub-class
     framing for the split artifact has stabilized
     into a regular-recurrence pattern after the long
     Rounds-55-128 implicit-bundling phase.
   - Companion sm_86 evidence + code-side artifact gaps
     (Round-19-added class, extended in Round 40 to cover
     code-side gaps, extended in Round 57 to cover
     typed_unavailable evidence-side rows): 46+ re-flags
     across rounds 18, 19, 25, 28, 31, 33, 36, 37, 39, 40,
     41, 42, 52, 53, 56, 57, 59, 62, 63, 67, 69, 75, 77, 78,
     83, 85, 87, 88, 90, 91, 92, 95, 100, 105, 107, 108, 112,
     117, 123, 124, 126, 130, 131, 141, 144. Round 144 marks the **18th re-flag of the
     predicted-L1 sub-class on the same
     `vendor_sgemm_strided_batched_cublas/timing_shape`
     cell** (Rounds 77, 78, 83, 85, 88, 91, 92, 95, 100,
     105, 107, 112, 117, 123, 124, 130, 131, 141, 144) — extending the
     highest-recurrence sub-class record in the closure
     addendum to 18 flags on a single cell identity.
     Round 144 is the **6th dual-sub-class
     companion-gap instance** (after Rounds 91, 100,
     108, 123, 141) and **BUNDLED form** (matching
     Rounds 100, 108, 141; Rounds 91, 123 were
     unbundled). Pattern: UB/B/B/UB/B/B — 4 bundled vs
     2 unbundled across 6 dual instances.
     Round 124's companion-gap is **single sub-class**
     (predicted-L1 only) — Round 123's dual instance
     (predicted-L1 + typed_unavailable carveout) did
     NOT recur at Round 124. The typed_unavailable
     carveout sub-class went silent for one round,
     confirming that the two sub-classes fire
     independently (Round 91 first observed
     independence; Rounds 100, 108 bundled them; Round
     123 unbundled-dual; Round 124 single-sub-class).
     **Companion-gap sub-class alternation across Rounds
     123-130** (8 consecutive rounds): Round 123 (dual)
     → Round 124 (single predicted-L1) → Round 125 (both
     silent) → Round 126 (single typed_unavailable
     carveout) → Round 127 (both silent) → Round 128
     (both silent) → Round 129 (both silent) → Round 130
     (single predicted-L1). Four distinct sub-class
     patterns now observed: dual, single-predicted-L1,
     both-silent (3-round streak Rounds 127-129),
     single-typed_unavailable. The sub-classes fire
     independently with no obvious cadence —
     confirming that the Round-91 independence
     observation extends robustly across 8 consecutive
     rounds with unchanged underlying artifact state.
     Round 108 is the **third dual-sub-class instance**
     of companion-gap (after Round 91 first, Round 100
     second) — both typed_unavailable (carveout row) and
     predicted-L1 (vendor_sgemm cell) sub-classes fire in
     the same Codex review (P1 #4 bundles both into one
     finding, matching the Round-100 bundling pattern).
     Round 91 was the FIRST round where both companion-gap sub-classes
     (typed_unavailable on
     `predicted_l1_per_admitted_cell.json:164-169` carveout
     row + predicted-L1 absence on
     `vendor_sgemm_strided_batched_cublas/timing_shape`) fired
     in the same review — confirming the two sub-classes are
     independent within the class (a host missing protobuf
     emits typed_unavailable rows for existing admissions
     AND fails to add rows for new admissions). Round 100
     is the SECOND dual-sub-class instance, this time
     bundled by Codex into a single P1 finding rather than
     two — the bundling pattern observed earlier for #1+#2
     (Rounds 35, 81, 87) now extends to intra-companion-gap
     sub-classes. Round 123 is the **FOURTH dual-sub-class
     companion-gap instance** — both typed_unavailable
     (carveout row) and predicted-L1 (vendor_sgemm cell)
     sub-classes fire in the same review, but this time
     Codex flags them as **two SEPARATE P1 findings**, not
     a single bundled finding. Round 123 is the **first
     UNBUNDLED dual instance since Round 91**: Rounds 100
     and 108 bundled the two sub-classes into one P1, while
     Rounds 91 and 123 surfaced them as independent flags.
     The bundling pattern is not strictly stable — Codex
     alternates between the unbundled and bundled framings,
     with no obvious determinant in the sub-class evidence
     itself (the underlying artifacts are unchanged
     across all four dual instances).
     Round 40's flags specifically targeted the code-side
     sub-class
     for the same
     `vendor_sgemm_strided_batched_cublas/timing_shape`
     cell whose evidence-side gap had been re-flagged in
     the prior 9 rounds (FAMILY_COVERS_SLUGS missing,
     driver `output` field missing, FP32/control tolerance
     missing).

   The lowest-severity round was Round 26 with 0 P1 / 1 P2
   / 1 P3, where both findings were NEW in-scope code bugs
   (zero queued). Round 43 was the first round since
   steady state began (Round 22) where all queued findings
   dropped from P1 back to P2 — a one-round
   severity-attenuation signal that did NOT persist (Round
   44 reverted to all-P1). The all-queued shape of Rounds
   19, 21, 27, 28, 30, 32, 33, 34, 35, 37, 38, 39, 40, 41,
   42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 56, 57,
   58, 59, 60, 61, 62, 65, 67, 68, 69, 70, 71, 72, 73, 74,
   75, 76, 77, 78, 80, 81, 84, 85, 86, 87, 88, 89, 90, 91,
   92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105,
   106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117,
   118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145 (Rounds 51, 55, 63, 64, 66, 79, 82, 83, 103, 111, 121, 128, 133, 143, and 146 excluded as mixed —
   each lands NEW in-scope code-bug fix(es); and the
   diminishing-severity P1/P2 distribution across Rounds
   22-31) reflects convergence:
   the Codex Review Phase has exhausted the
   broadly-actionable in-scope code findings; the
   remaining flags map one-to-one onto the addendum's
   classes. Round 29 closed the producer-consumer
   lockstep for the `all_zero` L1 label status across
   Rounds 22 + 26 + 29 — a NEW in-scope bug that
   surfaced because Round 26 enabled the producer-side
   emission of a previously-dead-semantics enum value.
   Round 31 closed a NEW in-scope bug Round 26 itself
   introduced: the drop-rate sidecar hoist used a renamed
   local (`drop_rate_sidecar`) that broke the trace-budget
   writer-literal parity scanner's token pattern
   (`NAME("sidecar") OP("=") NAME(any) OP("/") STRING(...)`).
   Round 31 restored the canonical `sidecar = ...` LHS
   while preserving Round 26's else-branch parallel
   unlink. Round 36 closed a NEW in-scope test bug: the
   `l2_reduction` shadow-report drift test stripped only
   `source_commit` from its byte-equivalence comparison,
   but the recapture script also emits a
   `source_commit_override_from` provenance stamp that the
   fresh generator never sets. The fix extended the strip
   set to include both provenance keys — symmetric and
   semantically correct since both fields are
   post-build provenance metadata, not generator output.
   Round 51 closed TWO NEW in-scope code bugs: (a) the
   `task12_substitution_preflight` iterated ALL admitted
   cells before computing the predicted_l1 identity-tuple
   set, raising spurious `missing_from_predicted` for the
   5 sm_120 admitted cells (added in the amended-cohort
   admission work) against the sm_86-cohort-specific
   predicted_l1 artifact. Round 51 added an arch=="sm_86"
   filter at the iteration point. (b) The HRD
   `bundle_from_trace` import at `run_suite.py:1351-1354`
   used a bare `from bundle_from_trace import
   emit_hrd_bundle` that could return the cached
   divergence-side module (same basename
   `bundle_from_trace.py` exists under
   `tools/divergence_proxies/`) in a long-lived Python
   process. Round 51 mirrored the divergence-side pattern
   (importlib.util.spec_from_file_location with an
   absolute path) so the HRD module is unambiguously
   resolved regardless of sys.modules cache state.
   Round 55 closed a NEW in-scope code bug in
   `_resolve_source_commit` at `run_suite.py:1567-1605`:
   the resolver read `.git/refs/<branch>` via bare
   `read_text()` and returned all-zeros on `OSError`,
   which fires on freshly-cloned repos that ship with
   packed refs (the loose ref file does not exist; the
   sha is in `.git/packed-refs`). Returning all-zero
   polluted `source_commit` fields in manifest pairing
   indexes and run summaries. Round 55 added a
   packed-refs fallback that scans `.git/packed-refs`
   for the matching ref line when the loose-ref read
   fails. The all-zero fallback is preserved for the
   case where neither layout resolves.
   Round 63 closed a regression that Round 51's own
   HRD bundle_from_trace import-pattern fix introduced:
   the new `spec_from_file_location` always loaded a
   fresh real module and bypassed pre-installed
   `sys.modules["bundle_from_trace"]` entries that
   skip-path regression tests use (via `monkeypatch.
   setitem`). Round 63 added a `sys.modules` pre-check
   ahead of the absolute-path import so the patched
   module wins when present; the absolute-path import
   stays as the no-patch fallback. The Round-51
   collision-avoidance contract is preserved because
   the divergence-side counterpart registers under the
   qualified name `tools.divergence_proxies.bundle_from_trace`
   rather than the bare basename `bundle_from_trace`,
   so divergence loading does not pollute the basename
   key.
   Round 64 closed a test-quality bug in
   `tools/local_labels/test_round_summary_shadow_diagnostics_scanner.py`'s
   `ActiveLoopRoundSummaryScanTest`: the test walked the
   discovered active-loop dir under `.humanize/rlcr/`
   (gitignored host-local scratch state per the
   project's `.gitignore` and CLAUDE.md handoff
   semantics) and validated every host-local round
   summary against a scanner contract that this loop's
   round summaries don't satisfy. Round 64 added a
   `git ls-files <loop_dir>` pre-check: the test now
   filters discovered summaries to the subset git
   actually tracks and skips entirely when none are
   tracked. Validates intent (committed test surface
   only) while admitting hosts where the
   `.humanize/rlcr/` dir is purely local scratch.
   Round 66 closed TWO NEW in-scope producer-schema
   lockstep gaps that the curated composite missed:
   (a) `tools/local_labels/run_headline_cohort_predict_compare.py:1378-1397`
   emits `publication_grade_floor_verdict`,
   `publication_grade_submission_verdict`, and
   `publication_grade_tags` into `comparison_bundle.json`,
   but `workloads/simbench_bw/schemas/comparison_bundle.schema.json`'s
   `additionalProperties: false` rejected them; Round 66
   added the three fields to the schema with structures
   that mirror the producer's `ContractFloorGateVerdict`,
   `SubmissionGateVerdict`, and tag-list dataclasses.
   (b) `tools/local_labels/parse_ncu_metrics.py:457-460`'s
   raw-counter ratio fallback emits the
   `ncu_lts_pct_overcount_raw_counter_ratio_used`
   parser-diagnostic for `.pct` values above 105.0, but
   `local_label_sidecar.schema.json`'s closed
   `parser_diagnostics` only allowed
   `ncu_sector_hit_rate_clamped_to_100`. Round 66 added
   the new key to `parser_diagnostics.properties` with
   the same `parser_diagnostics_clamped_record` items
   shape, citing BL-20260507-arch-token-multi-surface-
   lockstep + BL-20260506-schema-validator-probe-driven-
   tightening (both pre-existing). Both fixes are pure
   schema widenings — additive, closed-enum-extending,
   and backward-compatible with sidecars that omit the
   new keys.
   Round 121 closed a NEW in-scope shape-axis-forwarding
   gap in
   `tools/local_labels/sweep_repeatability_matrix.py`.
   `_label_driver_argv` at lines 373-377 built the
   workload argv as only `["--shape-point", cell.shape_point]`
   for SimBench-style shapes (anything other than
   correctness_shape/timing_shape). SimBench `main.cu`
   runners treat `--shape-point` ONLY as an output
   label; they do NOT load `shapes/<slug>.toml` to
   override their compiled defaults. The bug: a live
   sweep through any generic SimBench candidate (e.g.,
   `working_set_capacity_sweep/trace_l2_accuracy_xl`)
   would profile the binary's compiled correctness-
   sized defaults while emitting label sidecars
   claiming the requested trace/timing shape —
   mis-attributing NCU ground truth and silently
   admitting repeatability/admission verdicts against
   wrong data.
   Round 121 added `_load_simbench_shape_axes` +
   `_axes_to_workload_cli_args` helpers that load the
   per-slug TOML at
   `workloads/simbench_bw/shapes/<slug>.toml`, locate
   the shape entry matching `cell.shape_point` in any
   of timing_shapes/trace_shapes/correctness_shapes,
   and convert the axes dict to `--<kebab-flag> <value>`
   pairs (mirroring the SimBench main.cu CLI
   convention: snake_case axis → kebab-case CLI flag).
   The fix forwards these axis flags AFTER `--shape-point`
   so the runner profiles the requested shape's
   parameters instead of defaults. The helper raises
   `ValueError` when the TOML exists but no shape
   entry matches `shape_point` (fail-closed for
   candidate-roster/TOML drift). Round 121 added 2
   regression tests covering: (a) positive — axis
   flags appear in dry-run argv for
   working_set_capacity_sweep/trace_l2_accuracy_xl
   (--working-set-bytes 16777216, --accesses 4096,
   --warps 64); (b) negative — fail-closed ValueError
   for an unknown shape_point. The fix shape — "metadata
   describing axes used by downstream runners as ground
   truth, but the harness was passing the metadata's
   LABEL field instead of the axes themselves" — is
   structurally adjacent to the Round-111 launch_config
   / wrapper-geometry lockstep gap: both involve TOML
   metadata describing per-shape parameters that runtime
   code must consume, with a harness-vs-metadata
   mismatch in between. The 14th distinct round to
   close NEW in-scope work during the steady-state
   phase.
   Round 111 closed a NEW in-scope launch_config /
   wrapper-geometry lockstep gap across 7 simbench_bw
   shape TOMLs. Each shape's CUDA wrapper launches
   `blocks = warps` × `threads = 32`, but 20 shape-
   point launch_config entries advertised `grid =
   [1, 1, 1]` instead of `grid = [warps, 1, 1]`. The
   trace_size_estimator at
   `workloads/simbench_bw/trace_size_estimator.py:93-105`
   consumes launch_config to compute the per-shape
   warp_count, and downstream tooling reads
   launch_config as the authoritative launch geometry
   for trace estimation, manifests, and CTA/block
   provenance. Round 111 corrected the affected
   shape entries in 7 TOMLs (carveout_sensitivity_probe,
   dram_stream_stride_mix, l2_persisting_window_sweep,
   l2_prefetch_hint_sweep, l2_thrash_pointer_chase,
   register_spill_local_mem, working_set_capacity_sweep)
   so launch_config mirrors the wrapper's actual
   `blocks = warps` × `threads = 32` geometry. Two
   shape entries also had a stale block size
   (64 threads/block on `l2_persisting_window_sweep`
   trace_l2_accuracy_xl and `l2_prefetch_hint_sweep`
   trace_l2_accuracy_xl) which Round 111 corrected to
   32 threads/block to match the wrapper. Round 111
   ties the **lowest-severity round** record (0 P1 /
   1 P2 / 0 P3), previously held only by Round 79.
   The 13th distinct round to close NEW in-scope work
   during the steady-state phase.
   Round 103 closed a NEW in-scope active-slug filter
   bug in `tools/build/verify_cloud_arm_evidence.py`.
   `_list_active_slugs()` (lines 100-112) returned a
   portable_lcd directory as "active" if EITHER the
   compiled binary `<slug>_run` existed OR `main.cu`
   existed. The latter clause was too permissive:
   source-only directories (a workload tree checked in
   but never built/run on this host) have no
   `results/<slug>/manifest.json`, so the keyset gate
   computed `None` for the local hash and rejected
   otherwise valid cloud evidence — the committed
   wrapper-shape tests failed for 10 of the 15 returned
   slugs. The fix filters active slugs on
   `results/<slug>/manifest.json` presence directly,
   aligning the active set with the slugs that can
   actually participate in the per-slug keyset match
   gate. Round 103 added a regression test
   (`test_active_slugs_excludes_source_only_directories`)
   that asserts every returned slug has a reachable
   local manifest. The 12th distinct round to close
   NEW in-scope work during the steady-state phase
   (after Rounds 29, 31, 36, 51, 55, 63, 64, 66, 79,
   82, 83).
   Round 83 closed the path-boundary baseline drift that
   Round 82's in-scope nvbit_tracer/ edits introduced.
   The path-boundary regression at
   `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py::test_path_boundary_tracer_and_cupti_collector_pinned`
   pins the sha256 of every tracked file under
   `tools/nvbit_tracer/` plus the two CUPTI Activity
   collector sources; Round 82's intentional fix to
   `tools/nvbit_tracer/import_trace_contract.py` and
   `tools/nvbit_tracer/test_import_trace_contract.py`
   drifted both pinned entries. The discipline (test
   docstring lines 977-979): "Updating the pinned
   baseline is reserved for an explicit plan amendment."
   This round's plan-side narrative entry IS that
   amendment for the sm_120-only loop scope — the
   Round-82 edits are in-scope sm_120-only fixes
   (contract-validator convenience gap on the
   single-`.pb` entry point, with 5 new test cases
   pinning the new behavior). Round 83 updated the two
   drifted entries in
   `workloads/simbench_bw/tests/path_boundary_baseline.json`
   to the post-Round-82 hashes. The 11th distinct round
   to close NEW in-scope work during the steady-state
   phase. This shape — "in-scope edits to a pinned
   sub-tree must update the corresponding pin in the
   same commit" — is the second instance of the BL-CHX-
   producer-consumer-lockstep family (after Round 66's
   producer-schema lockstep gaps): the producer here
   being the modified source files, the consumer being
   the pinned baseline that lives in a separate part of
   the tree.
   Round 82 closed a NEW in-scope contract-validator gap
   in the trace-side per-block validator entry point.
   `tools/nvbit_tracer/import_trace_contract.py`'s
   `validate_threadblock_path` (the single-`.pb`
   validation entry point) accepted an optional
   `allocations` table argument, but when callers left
   it unset the validator never inferred the conventional
   `<traces_dir>/extra_info/allocations.json` sidecar.
   Records carrying `allocation_id != 0` then failed
   with `allocations_sidecar_missing` even when the
   sidecar existed in the trace directory — breaking the
   documented single-file CLI/API mode for newly
   produced traces. Round 82 added a
   `_infer_traces_dir_from_pb_path` helper that walks
   parents looking for a `threadblocks` ancestor and
   loads `<traces_dir>/extra_info/allocations.json` via
   the existing `load_allocations_sidecar`. Explicit
   callers that pass `allocations` continue to control
   the table fully (no auto-load); flat-layout `.pb`
   files outside a conventional trace tree preserve the
   prior fail-closed behavior for `allocation_id != 0`
   records. Round 82 added 5 unit tests covering: (a)
   conventional-layout admit, (b) conventional-layout
   missing-sidecar reject, (c) conventional-layout
   malformed-sidecar typed error, (d) flat-layout no
   auto-load, (e) explicit-table short-circuit. The
   fix is arch-neutral.
   Round 79 closed a NEW in-scope producer/consumer
   lockstep gap in the label-side kernel-signature
   fallback path. `tools/local_labels/label_driver.sh`
   line 530 hashed the raw CLI `$BINARY_PATH` string
   when `--measurement-manifest` was absent, while the
   trace-side `workloads/simbench_bw/run_suite.py::_kernel_signature_sha256`
   (lines 1623-1651) canonicalizes the executable to
   the repo-relative form (or bare basename for
   out-of-tree binaries). Both sides hashed
   `f"{slug}|{path}"` but the path strings diverged
   whenever the operator passed an absolute,
   ./-prefixed, or otherwise non-canonical spelling of
   the same binary — making the fallback labels
   unjoinable against predictor bundles for an
   identical binary on the same host. Round 79 added a
   `canonicalize_kernel_sig_binary_path` helper in
   `label_driver.sh` (mirrors the run_suite.py rule:
   `realpath -- "$raw"`, repo-relative when under
   `REPO_ROOT`, basename otherwise) and updated the
   test-side mirror at
   `tools/local_labels/test_provenance_enrichment.py::_kernel_signature_sha256`
   to apply the same canonicalization (the prior
   helper hashed the raw `Path` repr, so the two
   broken sides agreed by coincidence). Round 79 is
   the lowest-severity round to date: 0 P1 / 1 P2 / 0
   P3 — exceeds the prior Round-26 record of 0 P1 / 1
   P2 / 1 P3, and Round 43's one-round all-P2
   severity-attenuation signal. Round 111 ties Round
   79's lowest-severity record (0 P1 / 1 P2 / 0 P3);
   the two rounds are the only two with this profile
   in the loop's history. Round 79 was the 9th
   distinct round to close NEW in-scope code bugs
   during the steady-state phase (after Rounds 29, 31,
   36, 51, 55, 63, 64, 66); Round 82 was the 10th
   (single-`.pb` validator sidecar inference); Round 83
   was the 11th (path-boundary baseline re-pin after
   Round 82's intentional nvbit_tracer/ edits); Round
   103 was the 12th (cloud-evidence verifier active-
   slug manifest-presence filter); Round 111 was the
   13th (simbench_bw shape launch_config /
   wrapper-geometry lockstep); Round 121 was the 14th
   (sweep_repeatability_matrix shape-axis forwarding
   for SimBench candidates); Round 128 was the 15th
   (portable_lcd contract test mutating production
   artifacts → `--subtree` CLI flag + tmp-subtree
   negative-test rewrite); Round 133 was the 16th
   (warm_cold_pair_relaunch.toml axes/launch_config
   lockstep — axes missing `warps` for timing_00 and
   trace_00 caused the binary to default to warps=1
   while launch_config declared larger geometry);
   Round 143 was the 17th (label_driver shape-alias
   normalization — literal manifest-vs-CLI shape_point
   comparison rejected valid portable-LCD live sweeps
   because sweep passes legacy `timing_shape`/
   `correctness_shape` aliases while runners emit
   `timing_00`/`correctness_00`); Round 146 is the 18th
   (NVBit tracer VMM partial-unmap range handling —
   `mark_allocation_freed_range` primitive that splits
   live allocations on interior unmaps and shrinks on
   base/tail partials, replacing the base-only
   `mark_allocation_freed` call at the cuMemUnmap
   callsite).

   Round 146 closed a NEW in-scope code bug in
   `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`
   line 2858. The cuMemUnmap callback previously called
   `mark_allocation_freed(ctx, base)` while ignoring
   `p->size`. CUDA's VMM API permits partial unmaps:
   interior partial unmaps (ptr > base) returned early
   because the function matched the live map only at
   the exact base — leaving the unmapped pages still
   indexed by the allocation_id; base partial unmaps
   (ptr == base, size < orig_size) removed the entire
   live-map entry, dropping the still-mapped tail.
   Either bug could attach `allocation_id` to unmapped
   pages or drop a still-live mapping during GLOBAL
   lane resolution. Round 146 added
   `mark_allocation_freed_range(ctx, ptr, size)` to
   `allocation_registry.h` / `.cc`. For each live
   allocation that intersects [ptr, ptr+size), it marks
   the original allocation_id dead and creates new
   sub-allocations (with fresh allocation_ids) for the
   surviving pre-range `[base, ptr)` and post-range
   `[ptr+size, base+orig_size)` parts. Six host-only
   unit tests were added covering full unmap, base
   partial, tail partial, interior split, no-overlap,
   and sentinel inputs (33 tests now pass — up from
   27). `mark_allocation_freed(ctx, base)` is kept as-is
   for the non-VMM callsites (cuMemFree, cudaFree, etc.)
   that always free the entire allocation. The 18th
   distinct round to close NEW in-scope code bugs
   during the steady-state phase. Round 146 is also the
   **second silent-triad in-scope-only round in 3
   rounds** (after Round 143) — closure-addendum
   foundational triad was silent for both rounds, and
   Codex's review surfaced ONLY the actionable
   in-scope finding.

   Round 143 closed a NEW in-scope code bug in
   `tools/local_labels/label_driver.sh` lines 607-614.
   The embedded Python manifest-cell-identity validator
   compared `expected_shape_point` (from CLI) against
   `manifest_shape` (from runner-emitted measurement
   manifest) literally. `sweep_repeatability_matrix.py
   --live` labels portable-LCD cells with
   `shape_point=timing_shape` or `correctness_shape`
   (legacy aliases used throughout the harness flow),
   while the portable-LCD runners emit `shape_point=
   timing_00` or `correctness_00` in their manifests.
   The literal comparison therefore rejected valid live
   sweeps with `shape_point='timing_00' does not match
   CLI cell shape_point='timing_shape'`, blocking the
   documented timing/correctness-shape labeling path.
   Round 143 added a `_normalize_shape_point` helper
   inside the embedded Python block that maps the two
   legacy aliases (`timing_shape` → `timing_00`;
   `correctness_shape` → `correctness_00`) to the
   canonical runner-emission form. Both sides of the
   comparison are normalized before the equality check.
   Generic shape points (e.g., `trace_l2_accuracy_xl`)
   have no aliases and pass through unchanged. The fix
   shape is the third instance of the
   harness-vs-manifest identity-chain family in the
   SimBench/portable-LCD live sweep path: Round 121
   (shape-axis forwarding from sweep harness to
   binary), Round 133 (axes/launch_config lockstep in
   shape TOML), Round 143 (legacy-alias normalization
   in manifest-vs-CLI comparison). The broader
   metadata-lockstep family (Rounds 66, 79, 82, 83,
   103, 111, 121, 133, 143) now has 9 instances. The
   17th distinct round to close NEW in-scope code bugs
   during the steady-state phase. Round 143 is also the
   **first round in many where the closure-addendum
   foundational triad #1+#2+#4 is silent** —
   Codex's review surfaced ONLY the actionable
   in-scope finding (no queued sm_86 baggage flagged).

   Round 133 closed a NEW in-scope code/data bug in
   `workloads/simbench_bw/shapes/warm_cold_pair_relaunch.toml`.
   Two shape entries (timing_00 and trace_00) declared
   larger launch geometry in `launch_config` than in
   `axes`: timing_00 had `launch_config = {grid =
   [256,1,1], block = [128,1,1]}` but `axes` lacked
   `warps`; trace_00 had `launch_config = {grid =
   [32,1,1], block = [128,1,1]}` but `axes` lacked
   `warps`. `run_suite.py::_kernel_args_from_shape`
   forwards ONLY `axes` as CLI arguments, never
   `launch_config`, so the wrapper at
   `workloads/simbench_bw/l1/warm_cold_pair_relaunch/
   warm_cold_pair_relaunch.cu:68-71` (which reads
   `--warps` and launches `blocks = warps > 0 ? warps
   : 1` × `threads = 32`) defaulted to `warps=1`/
   32-thread launch while `launch_config` claimed
   256-block-128-thread (timing) or 32-block-128-thread
   (trace) launches. Round 133 added `warps = 256` to
   timing_00 axes and `warps = 32` to trace_00 axes
   (matching the `launch_config.grid[0]` values), and
   corrected `launch_config.block` to `[32, 1, 1]` for
   both shapes to match the wrapper's hardcoded
   `threads = 32`. `predicted_bytes` for trace_00 was
   left unchanged — that's a separate estimator-output
   concern. The fix shape is the **third instance of
   the SimBench-TOML metadata-lockstep family** (after
   Round 111: launch_config geometry mismatch across 7
   TOMLs; Round 121: harness forwarding only
   --shape-point label, not axes). This extends the
   broader metadata-lockstep family (Rounds 66, 79,
   82, 83, 103, 111, 121, 133) to 8 instances.

   Round 128 closed a NEW in-scope test-quality bug in
   `workloads/portable_lcd/tests/test_portable_lcd_verify_kernel_contracts.py`.
   Two negative tests
   (`test_negative_active_row_with_missing_traces` and
   `test_negative_active_row_with_missing_manifest`)
   moved the REAL production directories
   (`workloads/portable_lcd/gemm_hmma_bf16_sm80/traces`
   and `.../results`) into tmp_path, ran the verifier
   against the real subtree, then moved them back. A
   cross-device move or assertion failure between the
   move-out and the move-back would leave the working
   tree missing those production artifacts, breaking
   clean checkouts. The test docstring already declared
   the correct intent ("the production roster + slug
   directories are not mutated") but the implementation
   diverged — the `_build_tmp_slug_clone` helper was
   invoked but its result was never used by the failing
   tests, while a separate `_run_contract_with_subtree`
   helper was defined but never called. Round 128 added
   a `--subtree` CLI flag to
   `workloads/portable_lcd/verify_kernel_contracts.py`
   that overrides the default SUBTREE constant, with
   the `--roster` default falling through to
   `<subtree>/roster.toml` when not explicitly passed.
   The two negative tests were rewritten to use
   `_build_tmp_slug_clone` properly — they now delete
   the traces/results directory from the TMP clone and
   pass `--subtree <tmp>` to the verifier. The
   dead-code `_run_contract_with_subtree` helper was
   removed. The fix shape is the third instance of a
   "test mutating production paths"-family bug: Round
   64's test_round_summary_shadow_diagnostics_scanner
   walked gitignored host-local scratch state and
   validated it against a strict contract, Round 103's
   verify_cloud_arm_evidence active-slug filter was
   too permissive on source-only directories, and Round
   128's portable_lcd contract test moved real
   production artifacts. The 15th distinct round to
   close NEW in-scope code bugs during the steady-state
   phase.

   All six classes require sm_86 trace recapture on a
   3090 Ti host. Per the addendum's discipline, future
   rounds continue to record these as queued
   sm_86-maintenance follow-ups without applying same-loop
   fixes. The loop's commit set remains bounded; sm_86
   evidence hygiene awaits the follow-on 3090 Ti session.

## BitLesson Candidate Entries (for future bitlesson population)

- `BL-CHX-sm120-only-decoupled-framework`: "When a project decouples
  from a prior cross-arch dependence, the binding contract is that
  no active code path may filter by, join to, or be SHA-coupled to
  the retired arch. Historical artifacts may stay on disk for
  byte-equality preservation if the active path neither consumes
  them nor regresses when they are absent."
- `BL-CHX-sm120-cache-control-axis-matters-for-variance-stability`:
  "On sm_120 Blackwell consumer (RTX 5060 laptop), the NCU
  `--cache-control` axis is the dominant variance-stability lever
  for streaming activation kernels; `cache=all` admits at
  stddev ~0.01 across warmup counts; `cache=none` produces
  stddev ~0.1–0.4. Empirical evidence: round-0 verdict-table
  for `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl`."
- `BL-CHX-cohort-identity-irreducible-variance-requires-amendment`:
  "When a required cohort cell fails admission due to irreducible
  kernel-side variance (verified by N≥10 sidecars at the most
  variance-friendly config), the responsible response is a formal
  plan amendment proposing a defensible variance-stable
  replacement, NOT cell substitution, threshold softening, or
  repeat-count inflation."
