# Track A — Cross-Arch Issue Ledger (bidirectional: sm_86-blocked + sm_120-blocked)

This ledger records every cross-arch-blocked item that is in flight
or carried over into the current cross-arch development loop. The
filename is historical (`BLOCKED_ON_3090TI.md`); per plan v2 Track
A / C-AC-A2 the schema is now **bidirectional** — entries may be
blocked on the sm_86 receiving device (a `sm_120`-host loop deferred
the work to a future 3090 Ti session) **or** blocked on the sm_120
device (a `sm_86`-host loop is deferring work to a future 5060
laptop session). Every entry must declare which device it is
blocked on via a `**Blocked on**` label so the future receiving
operator can tell whether the deferred work targets their session.

It is the operator-facing entry point for any session that resumes
the deferred work without ambiguity.

The ledger lives at the repository root under `tools/` (not under the
session-local `.humanize/` directory) because `.humanize/` is
gitignored and the future operator must read this file after pulling
the working branch. The plan (`Path Boundaries → Allowed Choices →
Track A path chosen during execution`) explicitly grants discretion
over the ledger path; choosing a committable path is the only way
to satisfy the C-AC-A / C-AC-A2 negative tests that "ledger paths
must resolve at working-branch HEAD".

## Source of Truth

Read this ledger and every runbook it references from the working
branch's HEAD. Do **not** pin a `git checkout` target SHA when
reading the procedure; per `BL-20260504-runbook-self-reference` in
`.humanize/bitlesson.md`, pinning the source-of-truth SHA produces a
self-referential staleness trap when the documentation evolves
across the commits being pinned.

You can re-read this ledger or any referenced runbook without
changing your worktree:

```bash
git show <branch>:tools/BLOCKED_ON_3090TI.md | less
git show <branch>:tools/local_labels/TASK7_RUNBOOK.md | less
git show <branch>:tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md | less
```

Substitute `<branch>` with the working branch (currently `5060_3090ti_cross_dev`).
Verification target SHAs that the underlying runbooks pin (e.g.
`CROSS_ARCH_RUNBOOK.md`'s `f6b99da` target) are explicitly distinct
from this source-of-truth pattern: the runbooks document why the
worktree text at a pinned target is stale and how to keep using the
HEAD procedure.

## Ledger Schema

Each entry uses the five fields C-AC-A enumerates as positive-test
requirements **plus** a sixth blocking-device label C-AC-A2 added
for the bidirectional ledger. The hygiene check at
`tools/test_blocked_on_3090ti_hygiene.py` rejects any entry that
omits the blocking-device label or any of the five sections.

- **`**Blocked on**`** — required label declaring which device
  the work is deferred to. Allowed values are `sm_86` (the
  historical case: the sm_120 host can't close the work and a
  future 3090 Ti session must complete it) and `sm_120` (the
  new case introduced for this loop: an sm_86 host can't close
  the work and a future 5060 laptop session must complete it).
  The label appears once per entry, immediately after the
  `## Entry: …` heading and before the (a)-(e) sections.
- **(a) Item name** — short identifier matching the plan / prior
  loop's task tracker.
- **(b) What is blocked** — the acceptance criterion, AC, or
  contractual surface that cannot close on the originating
  device.
- **(c) Precise operator-facing artifact** — runbook path,
  command shape, and expected outcome the receiving operator
  follows.
- **(d) Originating-device prerequisites completed** — work that
  the current loop must (or has) landed on the originating
  device before the future receiving session begins. The
  historical column header used the sm_120 phrasing; under the
  bidirectional schema the same column may report `sm_86`
  prerequisites for an sm_120-blocked entry.
- **(e) Most recent originating-device evidence** — pointer to
  the live or offline evidence on the originating device that
  the deferred receiving work mirrors.

---

## Entry: task7 — sm_86 Local-Label Hardware Acceptance

**Blocked on**: sm_86

### (a) Item name
`task7` (sm_86 local-label hardware acceptance).

### (b) What is blocked
Original-plan AC-2 (Local label generation pipeline) and AC-3 (Label
repeatability gate), specifically the sm_86 portion of each. These
remain `Verified-live-sm_120` only and require a `Verified-live-sm_86`
session to earn `Cross-arch-verified` status per CLAUDE.md
"Two-Device Local Development Model".

### (c) Precise operator-facing artifact

- **Runbook**: `tools/local_labels/TASK7_RUNBOOK.md`. Read from
  branch HEAD (`git show <branch>:tools/local_labels/TASK7_RUNBOOK.md`).
  The runbook is the single source of truth for command shape,
  pre-flight probes, expected sidecar paths, and verdict
  interpretation. This ledger does NOT duplicate the command shape;
  per `BL-20260504-runbook-command-verification` the documented shape
  is exercised against the live driver via
  `Task7RunbookDocumentedCommandTest` in
  `tools/local_labels/test_label_driver_integration.py`.

- **Driver entry point**: `tools/local_labels/label_driver.sh`
  (the only NCU caller in the repository under the CLAUDE.md Rule 1
  local-label exception). The driver writes per-launch sidecars
  under `data/local_labels/sm_86/<slug>/<shape_point>/`.

- **Recommended workload**: `gemm_hmma_bf16_sm80` from the
  `workloads/portable_lcd/` suite. The Makefile under
  `workloads/portable_lcd/gemm_hmma_bf16_sm80/` builds
  `gemm_hmma_bf16_sm80_run` directly in the kernel directory root.

- **Expected outcome**: either
  - an admitted cell at
    `data/local_labels/sm_86/gemm_hmma_bf16_sm80/correctness_shape/`
    with five per-launch sidecars `label_0_0.json` …
    `label_4_0.json`, all admitting via
    `tools/local_labels/import_label.py`, and a verdict in
    `data/local_labels/label_admitted.json`; OR
  - an excluded cell with a typed reason
    (`label_repeatability_failed`, `label_repeatability_missing`,
    or `label_provenance_inconsistent`) in
    `data/local_labels/label_excluded.json`. The
    `label_repeatability_failed` verdict is itself a valid AC-3
    outcome — the Loop 1 R10/R11 sm_120 precedent established that
    the gate firing correctly is the AC-3 evidence the criterion
    requires.

- **Closure semantics**: per the plan `Claude-Codex Deliberation →
  Agreements`, closure is N≥5 individually admitting sidecars whose
  aggregator verdict is either an admitted cell or
  `label_repeatability_failed`. Verdicts of
  `label_repeatability_missing` or `label_provenance_inconsistent`
  indicate a failed collection that the operator must remediate
  (re-run failed iterations or rebuild the binary against a stable
  SHA, respectively) before attempting closure again.

### (d) sm_120-side prerequisites completed in Loop 2
At Loop 2 round 0 — none. The runbook under
`tools/local_labels/TASK7_RUNBOOK.md` is fully landed (Loop 1 round
40 repaired its command path against the live driver) and the
fake-NCU integration test
(`Task7RunbookDocumentedCommandTest::test_runbook_documented_command_reaches_ncu_and_emits_sidecar`)
is green. Subsequent Loop 2 rounds that touch the local-label
driver, the schema, or the runbook update this row.

### (e) Most recent sm_120-side evidence
Loop 1 round 10 produced
`data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json`
(admitting via `import_label.py`) and Loop 1 round 11 produced five
admitting sidecars whose aggregator verdict was
`label_repeatability_failed` with L2 stddev 6.05pp on the noisy
compute-bound microbench. The repeatability-failed verdict was the
AC-3 closure on sm_120; the 3090 Ti operator should expect a
similarly typed verdict (admitted or `label_repeatability_failed`)
on Ampere unless the cell is unusually clean. Loop 1 round 13
extended sm_120 coverage to fifteen active `portable_lcd` kernels
with hardware-accurate `gpu_sku=rtx_5060_laptop` provenance.

---

## Entry: task11b–task11d — sm_86 NVBit Tracer Cross-Arch

**Blocked on**: sm_86

### (a) Item name
`task11b-task11d-cross-arch-followup` (sm_86 NVBit tracer cross-arch
verification).

### (b) What is blocked
Original-plan AC-4 (NVBit per-instruction trace contract) cross-arch
status. Loop 1 round 5 + round 6 earned `Verified-live-sm_120` for
the per-lane resolution + `cross_allocation_lane_span` typed-exclusion
taxonomy + the consumer-side `allocation_id_missing` validator gate.
Promotion to `Cross-arch-verified` requires the same surfaces to pass
on Ampere.

### (c) Precise operator-facing artifact

- **Runbook**: `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`. Read from
  branch HEAD (`git show <branch>:tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`).
  The runbook documents the ten verification steps, the verification
  target SHA (`f6b99da`), the worktree-staleness warning, and the
  three flagged arch-coupling sites that may differ between sm_86
  and sm_120 (compression-format distribution, `cp.async` SASS
  witness, ASYNC_COPY classifier sidecar).

- **Build entry point**:
  `make -C tools/nvbit_tracer/tracer_tool` (Makefile uses
  `-arch=all`, so a single build emits both sm_86 and sm_120 cubins).
  Verify with
  `cuobjdump --list-elf tools/nvbit_tracer/tracer_tool/tracer_tool.so | grep cubin`.

- **Live sweep entry point**:
  `SIMBENCH_REQUIRE_LIVE_GPU=1 pytest workloads/simbench_bw/tests`.
  Expected baseline at the verification target: `679 passed,
  2 skipped`. Persisted into a sweep log under
  `.humanize/rlcr/<session-id>/round-N-cross-arch-sm_86-sweep.log`.

- **Offline-test entry point**:
  `pytest tools/nvbit_tracer tools/cloud tools/local_labels scripts/kb`.
  Expected baseline: `398 passed, 62 subtests passed` (or higher if
  later rounds have added tests; the count must not drop below the
  baseline).

- **Host-test entry point**:
  `make -C tools/nvbit_tracer/tracer_tool unit-tests`. Expected:
  `phase_classifier_test PASS`, `memory_op_class_test PASS`,
  `allocation_registry_test 27 tests passed`,
  `exclusion_aggregator_test 13 tests passed`.

- **Expected outcome**: all four entry points pass with the per-arch
  acceptable values; the sm_86 compression-format distribution
  matches `{base_stride: 33, base_delta: 11, list_all: 0}` (Loop 1
  round 24 master baseline) within drift tolerance; the goal-tracker
  AC-4 row promotes to `Cross-arch-verified`.

### (d) sm_120-side prerequisites completed in Loop 2
At Loop 2 round 0 — none. The producer + consumer surfaces landed in
Loop 1 (rounds 5–6 for per-lane resolution +
`cross_allocation_lane_span`; rounds 5–6 + later for the
`allocation_id_missing` validator gate). Loop 2 task9 (C-AC-H budget
completion) introduces two new typed reasons
(`nvbit_slowdown_budget_exceeded`, `nvbit_drop_rate_exceeded`) on
the trace-budget taxonomy family; if those land before the future
3090 Ti session, the runbook's expected sweep output may shift
(the test count would rise from `679 passed, 2 skipped`). This row
records that follow-up dependency as it lands.

### (e) Most recent sm_120-side evidence
Loop 1 round 5 + round 6 captured live `Verified-live-sm_120`
evidence at the Loop 1 master commit `1781304` (round 6 close).
Loop 1 round 9 broke the runbook's self-reference trap (commit
`f6b99da`). The persisted sweep evidence is
`.humanize/rlcr/2026-05-04_11-41-11/round-6-live-sweep.log`
(gitignored — must be referenced directly from the prior session
worktree if a re-export is required). The runbook itself names the
Round 9 commit `f6b99da` as the runbook-documented verification
target and explains why the worktree runbook text at that SHA is
intentionally not the source of truth.

---

## Entry: Loop 2 task7 — sm_86 Sector Stream Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task7-sector-stream-cross-arch-followup` (sm_86 cross-arch
verification for the Loop 2 R5 / C-AC-F sector stream generator).

### (b) What is blocked
C-AC-F's promotion from `Verified-live-sm_120` to
`Cross-arch-verified`. The Loop 2 R5 work earned (or will earn) the
sm_120 evidence; the matching sm_86 evidence requires capturing one
or more live traces on the 3090 Ti and consuming them through
`tools/sector_stream/generator.py` with the same admissions and
typed-exclusion outcomes the laptop produced.

### (c) Precise operator-facing artifact

- **Generator entry point**:
  `python tools/sector_stream/generator.py --input <record.json>` (the
  CLI accepts a single warp memory access record; the canonical
  consumer is the in-process Python API which lives at
  `tools/sector_stream.generator.compute_sector_stream`).
- **Output schema**:
  `workloads/simbench_bw/schemas/sector_stream.schema.json`. Both
  branches of the schema's `oneOf` (admitted and typed-exclusion)
  must continue to admit on sm_86.
- **Manifest entry**: `tools/arch_capabilities.json` declares
  `tools/sector_stream/generator.py` as
  `kind=feature_extractor` with
  `supported_arches=["sm_86", "sm_120-laptop"]`. The arch check is
  arch-independent in implementation; the supported_arches list
  reflects that both local arches can produce the captured warp
  records this generator consumes.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the producer-side wire
  format is architecture-independent. The generator therefore
  must produce identical output on identical input regardless of
  which device captured the warp record. Live evidence asymmetry
  on the second arch should expose architecture-coupling in tests
  or interpretation, not in the wire contract.

- **Expected outcome**: at least one sm_86 live trace from
  `workloads/simbench_bw/` is consumed end-to-end through the
  generator and produces sector-stream output that admits via the
  schema; the per-record outputs match the sm_120 outputs for
  identical input records (same lane addresses, masks, data
  widths, memory op class). The two new typed reasons
  (`data_width_invalid`,
  `sector_stream_class_unsupported`) emit on the sm_86 path with
  the same admission semantics as sm_120.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R5:
- `tools/sector_stream/generator.py` lands the deterministic
  generator (admitted + 4 typed-exclusion paths, 5
  classifications, 8 AC-5 positive-test patterns + vector-width
  fixtures).
- `workloads/simbench_bw/schemas/sector_stream.schema.json`
  declares the schema's `oneOf` shape with both admitted and
  typed-exclusion branches.
- `tools/arch_capabilities.json` declares the generator at
  `kind=feature_extractor`.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` adds the
  `sector_stream` family to the C-AC-C inventory with two
  surfaces (producer constant + schema enum) and a
  `SectorStreamTypedReasonsParityTest` enforcer that pins the
  family in lockstep at both surfaces.
- The two new typed reasons land in `ROUND_ADDED_TYPED_REASONS`;
  the two pre-existing reasons (`no_lanes_active`,
  `lane_address_length_invalid`) remain in
  `AC19_PLAN_ENUMERATED_TYPED_REASONS`.

### (e) Most recent sm_120-side evidence

Loop 2 R7 live evidence (R6-rework strict-schema closure):
- One sm_120 simbench live trace from the
  `dram_stream_stride_mix` kernel was captured on the laptop
  (Driver 595.71.05, CUDA 13.0, NVBit v1.8, RTX 5060 Laptop
  GPU, sm_120). Trace path:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  (gitignored).
- `python tools/sector_stream/consume_trace.py --trace-dir
  workloads/simbench_bw/l1/dram_stream_stride_mix/traces`
  → `contract_ok=true`, 41 input memory address records
  consumed, 41 admitted sector-stream records, 0 typed
  exclusions.
- Strict on-disk JSONL validation via
  `jsonschema.Draft202012Validator(sector_stream.schema.json)`:
  **41/41 records admit** (zero rejections, no
  sanitization). The R6 `_origin`-on-disk bug is fixed: the
  on-disk artifact contains exactly the schema's required +
  optional fields and nothing else.
- Memory-op-class distribution: 41 GLOBAL (all admit).
  Classification distribution: 32 strided, 8 scattered, 1
  broadcast.
  `sectors_per_request` distribution: 1.0 × 1, 4.0 × 14,
  5.0 × 26 (the broadcast record returns 1.0, confirming the
  AC9 invariant on live evidence).
- JSONL output written to
  `<trace_dir>/sector_stream/sector_stream.jsonl` (one line
  per record; 41 lines total). Provenance (label,
  warp_index, inst_index, vpc) is surfaced via the
  in-process `consume_trace_dir` return dict's `provenance`
  list and is NOT written to disk, keeping the artifact
  strictly schema-conformant.

Loop 2 R5/R6 offline tests:
- `pytest tools/sector_stream -q` → passing on host (R5: 32
  generator tests + 7 subtests; R6: +13 consumer tests, +5
  generator tests for the AC-9 invariant guard, +3 strided
  positive cases).
- `pytest tools/nvbit_tracer/test_exclusion_taxonomy_parity.py
  -q` → passing on host (sector-stream family: 2 surfaces +
  `SectorStreamTypedReasonsParityTest` enforcer; matches the
  C-AC-C inventory).

---

## Entry: Loop 2 task8 — sm_86 Divergence-Proxies Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task8-divergence-proxies-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R8 / C-AC-G divergence +
MSHR-pressure proxy module).

### (b) What is blocked
C-AC-G's promotion from `Verified-live-sm_120` to
`Cross-arch-verified`. Loop 2 R8 captured the sm_120 evidence on
the RTX 5060 laptop; the matching sm_86 evidence requires
re-running the proxy bundle producer against a 3090 Ti live trace
and confirming the bundle admits under strict schema validation
with the same per-proxy admission-vs-typed-unavailable shape.

### (c) Precise operator-facing artifact

- **Proxy module**:
  `python tools/divergence_proxies/proxies.py --input <evidence.json>`
  (CLI for ad-hoc operator verification; the canonical consumer
  is the in-process Python API
  `tools/divergence_proxies.proxies.compute_divergence_proxies`).
- **Live-trace consumer entry point**:
  `python tools/divergence_proxies/bundle_from_trace.py --trace-dir
  <PATH> [--device sm_86 | sm_120-laptop]`. The script is named
  `bundle_from_trace.py` rather than `consume_trace.py` to avoid a
  module-name collision with `tools/sector_stream/consume_trace.py`
  on the shared `sys.path` during pytest discovery.
- **Output schema**:
  `workloads/simbench_bw/schemas/divergence_proxies.schema.json`.
  Both branches of `$defs/proxy_or_unavailable.oneOf` (admitted
  numeric and typed-unavailable) must continue to admit on
  sm_86. The `outstanding_load_density_by_phase` annotation
  ("APPROXIMATION ONLY ... NOT true MSHR occupancy") is part of
  the contract.
- **Manifest entries**: `tools/arch_capabilities.json` declares
  both `tools/divergence_proxies/proxies.py` and
  `tools/divergence_proxies/bundle_from_trace.py` as
  `kind=feature_extractor` with
  `supported_arches=["sm_86", "sm_120-laptop"]`.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the proxy module is
  arch-independent in implementation. The 8 proxies must produce
  identical numeric values on identical sector-stream / phase /
  PC / lane-vector / sweep evidence regardless of which device
  captured it. Live evidence asymmetry on the second arch should
  expose architecture-coupling in tests or interpretation, not in
  the proxy formulas.
- **Expected outcome**: at least one sm_86 live trace from
  `workloads/simbench_bw/` is consumed end-to-end through
  `bundle_from_trace.py` and produces a divergence-proxies
  bundle that admits via the schema under strict
  `jsonschema.Draft202012Validator` validation; the per-proxy
  admission-vs-typed-unavailable shape matches the sm_120
  bundle's pattern (4 admitted from sector-stream + total-insts
  evidence, 4 typed-unavailable for the four evidence axes the
  current trace does not capture: phase markers, PC tracking on
  disk, lane vectors on disk, resident-warp sweep). The seven
  round-added typed reasons (`stride_entropy_unavailable_no_lane_table`,
  `throughput_saturation_slope_unavailable_no_sweep`,
  `pc_repeated_miss_likely_streams_unavailable_no_pc_data`,
  `md_loads_per_kilo_inst_unavailable_no_instruction_count`,
  `unique_sectors_per_warp_inst_unavailable_no_records`,
  `unique_lines_per_warp_inst_unavailable_no_records`,
  `sectors_per_request_unavailable_no_records`) and the one
  AC-19 plan-enumerated reason
  (`outstanding_load_density_unavailable_no_phase_records`) emit
  on the sm_86 path with the same admission semantics as sm_120.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R8:
- `tools/divergence_proxies/proxies.py` lands the proxy module
  with `compute_divergence_proxies(...)` returning a
  schema-conformant dict (8 proxies; 8 typed-unavailable
  reasons in `DIVERGENCE_PROXIES_TYPED_REASONS`).
- `tools/divergence_proxies/bundle_from_trace.py` lands the
  live-trace consumer that reads `sector_stream.jsonl` +
  `stats.csv` and writes
  `<trace_dir>/divergence_proxies/divergence_proxies.json`.
- `workloads/simbench_bw/schemas/divergence_proxies.schema.json`
  declares the bundle's structure with all 8 named proxy fields,
  the per-proxy admitted shape with optional `sample_count`, the
  per-proxy typed-unavailable shape closed under
  `additionalProperties: false` (R9: the R8 schema set this branch
  to `true` to mirror the NVBit-tracer typed-exclusion shape, but
  the divergence-proxies typed-unavailable records carry no
  context fields today; Codex R8 review tightened the branch so
  the R7 exact-artifact lesson holds for both branches of the
  proxy union), the 8-member typed_reason enum, and the
  `outstanding_load_density_by_phase` approximation annotation.
- `tools/arch_capabilities.json` declares both module entries at
  `kind=feature_extractor` with both local arches.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` adds
  the `divergence_proxies` family to the C-AC-C inventory with
  two surfaces (producer constant + schema enum) and a
  `DivergenceProxiesTypedReasonsParityTest` enforcer that pins
  the family in lockstep at both surfaces.
  `_EXPECTED_SURFACE_COUNT["divergence_proxies"] = 2`. The seven
  round-added typed reasons land in `ROUND_ADDED_TYPED_REASONS`;
  the one pre-existing reason
  (`outstanding_load_density_unavailable_no_phase_records`)
  remains in `AC19_PLAN_ENUMERATED_TYPED_REASONS`.
- `tools/divergence_proxies/test_proxies.py` exercises 29 host
  tests covering each proxy's admitted path, each proxy's
  typed-unavailable path, strict schema admission of every
  bundle shape (admitted-only, typed-unavailable-only, mixed,
  on-disk JSONL round-trip), and negative controls (extra
  field at bundle top, extra field on admitted proxy record,
  unknown typed_reason, missing proxy field, invalid
  schema_version).

### (e) Most recent sm_120-side evidence

Loop 2 R8 live evidence:
- One sm_120 simbench live trace from the
  `dram_stream_stride_mix` kernel was reused from R7 (Driver
  595.71.05, CUDA 13.0, NVBit v1.8, RTX 5060 Laptop GPU,
  sm_120). Trace path:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  (gitignored).
- `python tools/divergence_proxies/bundle_from_trace.py
  --trace-dir
  workloads/simbench_bw/l1/dram_stream_stride_mix/traces
  --device sm_120-laptop` →
  `kernel_unique_id=1, device=sm_120-laptop,
   sector_stream_records_consumed=41,
   total_instruction_count=2212, schema_admits=true,
   schema_first_failure=null`. Bundle written to
  `<trace_dir>/divergence_proxies/divergence_proxies.json`.
- Strict on-disk validation via
  `jsonschema.Draft202012Validator(divergence_proxies.schema.json)`
  on the produced file: **0 errors** (the consumer re-reads the
  bytes from disk before validating, applying the R7
  strict-validation lesson to the new bundle).
- Per-proxy values observed:
  - `unique_sectors_per_warp_inst`: admitted, value
    ≈ 4.5610 (mean over 41 sample_count).
  - `unique_lines_per_warp_inst`: admitted, value
    ≈ 1.7317 (mean over 41 sample_count).
  - `sectors_per_request`: admitted, value ≈ 4.5610
    (mean over 41 sample_count).
  - `md_loads_per_kilo_inst`: admitted, value ≈ 18.0832
    (over 2212 total instructions; corresponds to 40 of
    41 sector-stream records carrying touched_32B_sectors > 1).
  - `outstanding_load_density_by_phase`: typed-unavailable,
    typed_reason
    `outstanding_load_density_unavailable_no_phase_records`
    (the trace does not extract `%clock64` / `%globaltimer`
    phase metadata onto disk).
  - `pc_repeated_miss_likely_streams`: typed-unavailable,
    typed_reason
    `pc_repeated_miss_likely_streams_unavailable_no_pc_data`
    (per the R7 `_origin`-removal lesson, PC provenance is
    in-process only and not on the on-disk
    `sector_stream.jsonl`).
  - `stride_entropy`: typed-unavailable, typed_reason
    `stride_entropy_unavailable_no_lane_table` (the
    sector-stream consumer reconstructs the 32-element lane
    vector in memory but does not persist it).
  - `throughput_saturation_slope_under_resident_warp_sweep`:
    typed-unavailable, typed_reason
    `throughput_saturation_slope_unavailable_no_sweep`
    (single-launch trace, no resident-warp sweep).

Loop 2 R8 offline tests:
- `pytest tools/divergence_proxies -q` → 29 passed.
- `pytest tools/nvbit_tracer/test_exclusion_taxonomy_parity.py
  -q` → 166 passed, 54 subtests passed (divergence_proxies
  family adds 5 new tests + 2 new family-inventory subtests).

---

## Entry: Loop 2 task9 — sm_86 Trace-Budget Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task9-trace-budget-cross-arch-followup` (sm_86 cross-arch
verification for the Loop 2 R10 / C-AC-H trace-budget family
extension).

### (b) What is blocked
C-AC-H's promotion from `Verified-live-sm_120` to
`Cross-arch-verified`. Loop 2 R10 captured the sm_120
"correctly-does-not-produce" arm of the contract (the existing
`dram_stream_stride_mix` trace ran fully within budget, so no
slowdown / drop sidecars were emitted, and the verifier accepted
the artifact). The matching sm_86 evidence requires either (1)
re-running an sm_86 simbench live trace and verifying the same
"correctly-does-not-produce" pass, or (2) deliberately running a
sm_86 trace that crosses one of the new budgets to exercise the
"correctly-produces" arm and confirm the writer / verifier
behavior matches the sm_120 path.

### (c) Precise operator-facing artifact

- **Producer constants**:
  `tools/cloud/trace_budgets.py` declares
  `NVBIT_SLOWDOWN_BUDGET_EXCEEDED_SIDECAR: str =
  "nvbit_slowdown_budget_exceeded.json"` and
  `NVBIT_DROP_RATE_EXCEEDED_SIDECAR: str =
  "nvbit_drop_rate_exceeded.json"`.
- **Writer literals**:
  `workloads/simbench_bw/run_suite.py` carries the writer code
  paths for both new sidecars in the trace-budget family's
  `sidecar = <dir> / "<reason>.json"` shape (the parity scanner
  reads this surface). The writer is guarded by a row-level
  `slowdown_budget_max_ratio` + `baseline_wall_time_s` for
  slowdown and `drop_rate_budget_max` + per-launch
  `drop_counters` for drop-rate; today no roster row carries
  those fields and the writer stays dormant.
- **Validator discovery**:
  `workloads/simbench_bw/validate.py::_run_validation` adds
  `rglob("nvbit_slowdown_budget_exceeded.json")` and
  `rglob("nvbit_drop_rate_exceeded.json")` discovery loops,
  routing matches through `validate_trace_sidecar(...)`.
  `validate_trace_sidecar` dispatches on filename to the
  matching schema literal so all three trace-budget reasons
  appear at the validator-schema surface.
- **Output schemas**:
  `workloads/simbench_bw/schemas/nvbit_slowdown_budget_exceeded.schema.json`
  records declared budget, observed slowdown, baseline +
  actual wall times, and the trace-budgets canonical hash.
  `workloads/simbench_bw/schemas/nvbit_drop_rate_exceeded.schema.json`
  records declared budget, observed drop rate, expected /
  admitted record counts, and the four named drop-counter
  source counts (channel overflow, ring-buffer loss, decoder
  gaps, expected-record mismatch). Both schemas use
  `additionalProperties: false`.
- **Verifier**:
  `workloads/simbench_bw/verify_artifacts.py` walks the
  results tree for both new sidecars and rejects sidecars
  whose recorded actuals do not exceed the declared maxes
  (a self-contradictory sidecar that should not have been
  emitted). The drop-rate verifier also checks that the
  per-source counter sum equals `expected_record_count -
  admitted_record_count`; a counter-sum mismatch is
  `BudgetViolationUntracked`-equivalent noise.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the trace-budget
  surfaces are arch-independent in implementation. The two
  new sidecars must produce identical schemas, identical
  verifier rejection logic, and identical writer behavior on
  identical input regardless of which device captured the
  trace. Live evidence asymmetry on the second arch should
  expose architecture-coupling in the producer-side
  measurement pipeline (which is not yet wired), not in the
  trace-budget contract.
- **Expected outcome**: at least one sm_86 simbench live
  trace runs end-to-end through the orchestrator and the
  resulting `results/<slug>/<...>` tree is accepted by
  `verify_artifacts.py` either as the
  "correctly-does-not-produce" arm (no slowdown / drop
  sidecars emitted because the trace ran within budget) or
  the "correctly-produces" arm (a sidecar that admits under
  strict schema validation and is internally consistent).

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R10:
- `tools/cloud/trace_budgets.py` declares the two new
  sidecar-filename constants in the parity scanner's expected
  shape; both are exported from `__all__`.
- `workloads/simbench_bw/run_suite.py` carries the writer
  code paths for both new sidecars in the
  `sidecar = artifact_dir / "<reason>.json"` shape.
- `workloads/simbench_bw/validate.py` discovers both new
  sidecars via `rglob` and routes them through
  `validate_trace_sidecar`, which dispatches on filename to
  the matching schema literal.
- `workloads/simbench_bw/schemas/nvbit_slowdown_budget_exceeded.schema.json`
  and
  `workloads/simbench_bw/schemas/nvbit_drop_rate_exceeded.schema.json`
  are committed under `additionalProperties: false`. The
  drop-rate schema's description enumerates the four named
  drop-counter sources (NVBit channel overflow + ring-buffer
  loss + decoder gaps + expected-record mismatch).
- `workloads/simbench_bw/verify_artifacts.py` walks the
  results tree for the two new sidecars and rejects
  self-contradictory ones (actual <= max,
  trace_budgets_sha256 mismatch, drop-counter-sum mismatch).
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  `TRACE_BUDGET_TYPED_REASONS_EXPECTED` widens to the
  3-member set (`trace_volume_exceeded`,
  `nvbit_slowdown_budget_exceeded`,
  `nvbit_drop_rate_exceeded`). The two new reasons remain in
  `AC19_PLAN_ENUMERATED_TYPED_REASONS` (already plan-listed;
  no `ROUND_ADDED_TYPED_REASONS` change). The four-layer
  `TraceBudgetLiveLayerParityTest` confirms all four
  surfaces report identical sets.
- `workloads/simbench_bw/tests/test_trace_budget_c_ac_h_verifier.py`
  exercises 16 host tests covering schema admission of
  well-formed and ill-formed sidecars, verifier rejection of
  contradictory sidecars, verifier acceptance of well-formed
  sidecars, the within-budget-no-sidecars
  "correctly-does-not-produce" arm, and the validator's
  filename-dispatching schema selection.

### (e) Most recent sm_120-side evidence

Loop 2 R10 live evidence (the
"correctly-does-not-produce" arm):
- The existing R7 sm_120 simbench live trace at
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  (Driver 595.71.05, CUDA 13.0, NVBit v1.8, RTX 5060 Laptop
  GPU, sm_120) ran fully within its declared budgets (no
  baseline_wall_time_s / drop_counters configured on the
  row, so the writer stayed dormant by design).
- `python workloads/simbench_bw/verify_artifacts.py
  --results <results_root>` over a synthetic results tree
  containing the live trace's directory structure: exit
  code 0 (no slowdown / drop sidecars are present, and the
  verifier accepts the absence as the
  "correctly-does-not-produce" arm of C-AC-H's positive-test
  clause).
- The 16-test offline suite at
  `workloads/simbench_bw/tests/test_trace_budget_c_ac_h_verifier.py`
  exercises every code path of the new schemas + verifier
  with synthetic fixtures; all 16 tests pass.

Loop 2 R10 offline tests:
- `pytest workloads/simbench_bw/tests/test_trace_budget_c_ac_h_verifier.py
  -q` → 16 passed, 4 subtests passed.
- `pytest tools/nvbit_tracer/test_exclusion_taxonomy_parity.py
  -q` → 166 passed, 54 subtests passed (no test count change
  vs. R9; the trace-budget family's existing parity tests
  now scan the widened expected set without modification).

---

## Entry: Loop 2 task11 — sm_86 HRD Profiler Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task11-hrd-profiler-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R17 / task11 / AC-6
conditional HRD profiler + cache-model spec stack).

### (b) What is blocked
AC-6's promotion from `Verified-live-sm_120` to
`Cross-arch-verified`. Loop 2 R17 captured the sm_120
evidence on the RTX 5060 laptop; v3 R10 captured the
matching sm_86 evidence on the RTX 3090 Ti (validated
against carried R8 capture at HEAD `0852304`; HRD bundle
admits via strict schema validation with the same
typed-unavailable taxonomy as sm_120; 3-way source-commit
equality holds). See Section (f) below for the v3 R10
sm_86 evidence record. The remaining `Cross-arch-verified`
work is a future sm_120 v3 leg replaying the same
producer against a fresh sm_120-laptop bundle on the 5060
laptop per CLAUDE.md "Two-Device Local Development Model".

### (c) Precise operator-facing artifact

- **Profiler module**:
  `python tools/hrd_profiler/profiler.py` (the canonical
  consumer is the in-process Python API
  `tools/hrd_profiler.profiler.compute_hrd_profile`).
- **Cache-model spec validator**:
  `tools/hrd_profiler/cache_model_spec.py` provides
  `resolve_cache_model_spec(...)` + `validate_declared_cache_model_spec(...)`
  (the original-plan AC-6 negative test is the validator's
  declared-spec rejection arm).
- **Live-trace consumer entry point**:
  `python tools/hrd_profiler/bundle_from_trace.py
  --trace-dir <PATH> [--device sm_86 | sm_120-laptop]`.
  The script reads `<trace_dir>/sector_stream/sector_stream.jsonl`
  and `<trace_dir>/stats.csv`, calls the profiler, and
  writes the bundle to
  `<trace_dir>/hrd_profile/hrd_profile.json`.
- **Output schema**:
  `workloads/simbench_bw/schemas/hrd_profile.schema.json`.
  Top-level `bundle_underconfigured: bool` fast-path
  field; `bins[]` admits per-bin `oneOf` between `ok` and
  typed-unavailable branches; `evidence[]` admits
  bundle-level typed-unavailable annotations and
  informational notes; every closed branch under
  `additionalProperties: false`.
- **Manifest entries**: `tools/arch_capabilities.json`
  declares the three new HRD modules
  (`profiler.py`, `cache_model_spec.py`,
  `bundle_from_trace.py`) at `kind=feature_extractor`
  with `supported_arches=["sm_86", "sm_120-laptop"]`.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the profiler is
  arch-independent in implementation. The 16-member
  typed-reason taxonomy and the bundle's structural
  shape must not depend on the device that produced the
  trace.
- **Expected outcome**: at least one sm_86 simbench live
  trace runs end-to-end through `bundle_from_trace.py`
  and produces an `hrd_profile.json` that admits via
  the schema under strict
  `jsonschema.Draft202012Validator` validation. The
  bundle's `bundle_underconfigured` flag matches the
  recomputed value (R12 fail-closed pattern). The 16
  typed reasons emit on the sm_86 path with the same
  admission semantics as sm_120.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R17:
- `tools/hrd_profiler/profiler.py` lands the profiler
  module with `compute_hrd_profile(...)` returning a
  schema-conformant bundle dict (16 typed reasons in
  `HRD_PROFILE_TYPED_REASONS`).
- `tools/hrd_profiler/cache_model_spec.py` lands the
  resolver (with corroboration evidence) + the
  declared-spec validator (with rejection on
  contradiction).
- `tools/hrd_profiler/bundle_from_trace.py` lands the
  live-trace consumer that writes
  `<trace_dir>/hrd_profile/hrd_profile.json` and
  strict-validates the on-disk bytes via
  `Draft202012Validator`.
- `workloads/simbench_bw/schemas/hrd_profile.schema.json`
  declares the bundle's structure with all 9 reuse-state +
  6 cache-model spec + 1 bin-quality typed reasons in the
  shared `$defs.typed_reason.enum`.
- `workloads/simbench_bw/run_suite.py:116` `class
  DeviceProps` carries `l2_cache_size_bytes`;
  `probe_device_identity()` reads
  `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`.
- `tools/arch_capabilities.json` declares all three new
  HRD modules at `kind=feature_extractor`.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  adds the `hrd_profile` family with two surfaces and the
  `HrdProfileTypedReasonsParityTest` enforcer; the 13
  round-added reasons land in
  `ROUND_ADDED_TYPED_REASONS`; the 3 plan-enumerated
  reasons (`cache_model_capacity_missing`,
  `cache_model_capacity_inconsistent`,
  `cache_model_underconfigured`) remain in
  `AC19_PLAN_ENUMERATED_TYPED_REASONS`.
- `tools/hrd_profiler/test_*.py` exercises the offline
  test suite (hierarchy descent paths, every typed reason,
  both underconfigured trigger arms, scalar-consumer
  refusal, verifier negatives, strict schema round-trip).

### (e) Most recent sm_120-side evidence

Loop 2 R17 live evidence: the live consumer was run
against the existing `dram_stream_stride_mix` sm_120
trace (Driver 595.71.05, CUDA 13.0, NVBit v1.8, RTX 5060
Laptop GPU, sm_120). The exact command + observed bundle
shape are recorded in `round-17-summary.md`.

Loop 2 R18 mainline rework (offline-verified, sm_120 live
re-capture queued):

- `compute_hrd_profile` now resamples whole **kernel
  launches** (the
  `unit=kernel_launch` resampling unit declared in the
  schema). The R16 task10 artifact §3 / 5 mandate. The
  per-access flat resample was an R17 oversight; the R18
  refactor brings the bootstrap CI back in line with the
  spec. Test: `LaunchUnitBootstrapTest` exercises a
  bimodal-launch fixture and asserts a wide CI half-width
  the per-access resample would have under-reported.
- `tools/hrd_profiler/observation_builder.py` lands a
  reuse-distance observation builder
  (`build_observations_from_sector_stream_records`,
  `build_launches_from_sector_stream_jsonl`) that walks
  the on-disk `<trace_dir>/sector_stream/sector_stream.jsonl`
  and emits per-access observations with finite per-layer
  reuse distances via LRU-stack distance. Same-launch
  last-touched-position semantics; cold/infinite labels
  per the R16 task10 artifact §1 hierarchy.
- `tools/hrd_profiler/bundle_from_trace.py::emit_hrd_bundle`
  is the new production-path API: it accepts real
  `manifest_pairing_index`, `device`, `cuda_l2_cache_size_bytes`,
  and `static_card_l2_cache_size_bytes` from the caller
  (`run_suite.py::handle_trace`) and returns a status dict.
  The CLI `consume_trace_dir` keeps ad-hoc operator
  inspection convenience defaults; the controller path no
  longer fabricates placeholders.
- `workloads/simbench_bw/run_suite.py::handle_trace`
  invokes `_emit_hrd_bundle_for_run_summary(...)` after
  the trace is captured. The helper pulls the real device
  block, sha256 of the simbench executable, source-commit
  ref-resolution from `.git/HEAD`, and kernel-signature
  placeholder, then calls `emit_hrd_bundle` and persists
  the R16 task10 artifact §10 evidence fields into the
  per-shape `trace_<shape>.json` summary. Skip path: when
  the sector-stream JSONL is missing or `emit_hrd_bundle`
  raises, all Section 10 fields land None and a typed
  `hrd_skip_reason` is recorded.
- `workloads/simbench_bw/verify_artifacts.py::_verify_hrd_run_summaries`
  is the new R12-style fail-closed verifier. For every
  `trace_*.json` carrying `hrd_profile_path`, it loads the
  referenced bundle, strict-validates it under
  `Draft202012Validator`, recomputes
  `typed_unavailable_count`, `underconfigured_bin_count`,
  `low_confidence_bin_count`, `bundle_underconfigured`, and
  `cache_model_underconfigured_typed_reason` from the
  bundle's authoritative `bins[]` and `evidence[]`, and
  rejects any cached-vs-recomputed disagreement as
  `BudgetViolationUntracked`-equivalent.
- `workloads/simbench_bw/tests/test_hrd_run_summary_verifier.py`
  lands the verifier-rejection negative-control tests:
  schema-invalid bundle rejected, cached-vs-recomputed
  disagreement rejected (4 separate fields), missing-bundle
  rejected, typed-unavailable arm without
  `hrd_profile_path` admits, well-formed positive control
  admits.

The R18 work is **Offline-verified** at this writing
(host-only `python -m pytest workloads/simbench_bw/tests/`
+ `python -m unittest tools.hrd_profiler.test_hrd_profiler`
+ `python -m unittest tools.nvbit_tracer.test_exclusion_taxonomy_parity`
all pass). The next sm_120 session must re-run the
simbench `dram_stream_stride_mix` trace through the
updated controller (real device block, no placeholders)
and record the bundle's `bundle_underconfigured` /
`underconfigured_bin_count` shape. The matching sm_86
cross-arch promotion landed in v3 R10 — see Section (f)
below for the recorded admission record, source-commit
equality, and bundle structural shape verbatim.

Loop 2 R19 mainline rework (Verified-live-sm_120;
sm_86 cross-arch follow-up still queued):

- `tools/hrd_profiler/observation_builder.py` is now
  **proto-backed**. The synthetic-record-index identity
  builder from R18 is replaced by a builder that walks
  every `<trace_dir>/threadblocks/**/*.pb` proto via
  `import_trace_contract.validate_trace_dir` +
  `_reconstruct_lane_vector`, emits one HRD observation
  per active+predicated lane × per touched 32B sector,
  computes byte-address-derived identities at all four
  hierarchy layers (`sector_addr = byte_addr // 32`,
  `line_addr = byte_addr // 128`,
  `allocation_id` from
  `extra_info/allocations.json` when the address range
  matches, otherwise the line-address fallback per R16
  task10 §1, `region_id = byte_addr // 4096`), and groups
  observations by kernel-launch directory
  (`device_<id>/stream_<id>/kernel_<id>/`). Live sm_120
  trace produces 24,577 observations (R18 was 41 records).
- `tools/hrd_profiler/bundle_from_trace.py::emit_hrd_bundle`
  now returns a `contract_ok` field; on wire-contract
  failure the bundle is not written and the controller
  surfaces a typed `hrd_skip_reason` derived from
  `contract_typed_reason`.
- `workloads/simbench_bw/run_suite.py::_emit_hrd_bundle_for_run_summary`
  is wired to **generate the sector-stream JSONL from a
  fresh `dynamic_trace.pb` via
  `tools/sector_stream/consume_trace.consume_trace_dir`
  in-process when the JSONL is missing**, requires
  `contract_ok` and `schema_admits_all`, and surfaces
  typed `hrd_skip_reason` strings on contract / schema
  failures (closure of Codex's R18 directive #5: "do not
  let the normal fresh-trace path silently skip HRD
  because a sidecar was not precomputed"). The helper
  persists the **full 41-field R16 task10 §10 evidence
  list** (R18 wrote 9 fields). New helpers
  `_HRD_SECTION10_FIELDS`, `_hrd_skip_run_summary`,
  `_ensure_sector_stream_jsonl`,
  `_bundle_capacity_typed_reason`, `_stats_csv_kernel_name`.
- `workloads/simbench_bw/verify_artifacts.py::_recompute_hrd_bundle_section10`
  now enforces the **three-way invariant**:
  `bundle_underconfigured` requires three-way agreement
  between (a) at least one
  `cache_model_underconfigured` bin in `bins[]`,
  (b) the bundle-level
  `cache_model_underconfigured` annotation in
  `evidence[]`, and (c) the top-level
  `bundle_underconfigured: true` cached flag. The verifier
  surfaces the specific mismatch class
  (bin-without-annotation, annotation-without-bin,
  top-level-disagreement) as separate
  `BudgetViolationUntracked`-equivalent errors. The
  evidence record's `underconfigured_bin_count` and
  `scalar_hit_rate_refusal` are also checked. The new
  per-summary-field comparison loop walks the 39
  Section 10 fields with a bundle counterpart and rejects
  any disagreement.
- `workloads/simbench_bw/tests/test_hrd_run_summary_verifier.py`
  adds the R19 negative-control class
  `HrdVerifierThreeWayInvariantTest` with 3 tests
  covering all three mismatch classes
  (bin-without-evidence-with-truthful-counts;
  evidence-without-bin; top-level-disagrees-with-bin-and-evidence).
  The R18 9-test suite is preserved with field-name
  updates.

R19 fresh sm_120 live evidence (Driver 595.71.05, CUDA
13.2, RTX 5060 Laptop GPU, sm_120):

- Live consumer reads
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  via the proto-backed builder.
- Bundle's `device.l2_cache_size_bytes = 33554432`
  (32 MiB, CUDA-derived).
- Bundle's
  `manifest_pairing_index.binary_sha256 =
  329dabf226f49f2c76e460ee403811fc4b17a7633c3eb611a262f89267c9e97e`
  (sha256 of `dram_stream_stride_mix_run`),
  `source_commit = e211bfbf0d016d48c5710479c0e48591d21afbf1`
  (R18 commit; R19 evidence will rebase on the R19 commit
  during the round commit step),
  `kernel_signature_sha256 = be7d275827bf27b3a235976d0c15d9448128976edf88d88e21cf708ae7baacc0`,
  `kernel_name = "dram_stream_stride_mix_kernel"`.
- Bundle's bin distribution: 5 underconfigured bins
  (sample counts below the 50-record floor at
  sector / line / allocation / region layers) +
  1 region-cold bin. `bundle_underconfigured: true`.
  `evidence[]` carries the bundle-level annotation with
  `underconfigured_bin_count: 5,
  scalar_hit_rate_refusal: true`.
- Run summary persists all 41 R16 task10 §10 fields:
  `trace_access_count = 24577`, `trace_drop_count = 0`,
  `trace_drop_rate = 0.0`, `bootstrap_resamples = 2000`,
  `ci_half_width_target = 0.05`,
  `minimum_bin_sample_count = 50`,
  `cuda_l2_cache_size_bytes = 33554432`,
  `static_card_l2_cache_size_bytes = 33554432`,
  `cache_model_capacity_source = "cuda_device_prop"`,
  `cache_model_capacity_typed_reason = null`,
  `geometry_source = "constant"`,
  `geometry_typed_reason = null`,
  `associativity_policy = "fully_associative_lru_approximation"`,
  `fill_policy = "write_allocate_on_global_load"`,
  `write_policy = "write_back_approximation"`,
  `l1_bypass_cases = [".cg", "atomic"]`,
  `ncu_cache_hit_rate_l2_sector = null`,
  `ncu_cache_hit_rate_l1tex_sector = null` (the local-
  labels pipeline reconciles these in a future round),
  `bundle_underconfigured = true`,
  `cache_model_underconfigured_typed_reason = "cache_model_underconfigured"`.
- `python workloads/simbench_bw/verify_artifacts.py
  --results workloads/simbench_bw/l1/dram_stream_stride_mix/traces`
  → `ok` (the verifier walks the new three-way invariant
  + per-Section-10-field comparison without errors).
- Offline sweep clean: 732 simbench tests +
  827 tools tests, 4 + 458 subtests.

Loop 2 R20 closure (Verified-live-sm_120 at reviewed HEAD):

- New `tools/hrd_profiler/test_observation_builder.py` lands
  13 focused regression tests for the proto-backed builder
  (Codex's R19 review action item). The tests use synthetic
  threadblock proto fixtures (the same generated bindings the
  production builder consumes) to assert: one HRD observation
  per active+predicated lane × per touched 32B sector;
  vector-width sector spanning (4-byte / 32B-aligned / 128-byte
  / misaligned 64-byte cases); per-kernel-launch grouping (no
  cross-launch LRU pollution); allocation-sidecar identity vs.
  line-address fallback; same-launch LRU reuse-distance
  semantics across all four hierarchy layers; modeled-hit
  derivation from `sector_rd < capacity_bytes // 32`.
- Live sm_120 evidence re-emitted at reviewed HEAD
  `f78c4c83b0d1a8783a85da22999533402ee4defe`. Both
  `hrd_profile.json` and `trace_trace_00.json` now record
  `manifest_pairing_index.source_commit` /
  `source_commit = f78c4c83b0d1a8783a85da22999533402ee4defe`,
  closing the R19 evidence-freshness blocker. Bundle still
  carries 24,577 observations with bundle_underconfigured=true
  (5 underconfigured bins + 1 region-cold bin); verifier
  exits 0.
- Offline sweep clean at R20 HEAD: 732 simbench tests +
  **840 tools tests** (R19 baseline 827 + R20's 13 new
  builder tests).

Loop 2 R21 closure (Verified-live-sm_120 at the **R20 reviewed
HEAD** `6ec3ceba60358f2d52948f9b4e8ad3d40693a4f8`):

- Live sm_120 evidence re-emitted via the in-process probe
  through the R19 controller path at the R20 reviewed HEAD.
  Both
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/hrd_profile/hrd_profile.json`
  (`manifest_pairing_index.source_commit`) and
  `trace_trace_00.json` (`source_commit`) now record
  `6ec3ceba60358f2d52948f9b4e8ad3d40693a4f8`. Bundle shape
  preserved: 24,577 observations, 14 bins (5 underconfigured +
  1 region-cold + admitted bins), `bundle_underconfigured: true`,
  matching evidence record. `verify_artifacts.py` exits 0.
- `tools/hrd_profiler/test_observation_builder.py::AllocationIdentityTest`
  rewritten to lock the sidecar branch:
  - `test_two_lines_in_same_allocation_share_identity` uses a
    `MEMORY_OP_CLASS_GLOBAL` instruction with
    `address.allocation_id = 1`, two distinct lane addresses
    (`base + 0x40` then `base + 0xC0`) inside one
    allocations.json entry of size `0x200`. The sidecar arm
    asserts the second observation has
    `allocation_state="finite"` and `allocation_rd=0` AND
    `line_state` is `cold` for both observations (because the
    two addresses live in distinct 128B lines). The latter is
    the load-bearing distinguisher: a line-fallback
    implementation would also produce two cold line identities,
    which would force the allocation layer to be cold too.
    Asserting allocation finite under that line-cold shape locks
    the sidecar lookup path.
  - `test_two_lines_no_sidecar_remain_cold_at_allocation_layer`
    is the paired control: same two addresses, no
    `allocations.json`, SHARED memory class. Asserts the second
    observation's `allocation_state` is `cold` (line-fallback
    identities differ, so no LRU reuse at the allocation layer).
  - `test_same_line_no_sidecar_still_re_touches_at_allocation_layer`
    is the sanity baseline: two addresses in the same 128B line
    with no sidecar still produce a finite allocation reuse via
    the line-address fallback (guards the fallback path itself).
- Offline sweep clean at R21 HEAD: 732 simbench tests +
  **841 tools tests** (R20 baseline 840 + R21's strengthened
  allocation arm net +1 test). The local 732/2 simbench shape
  reflects this session's live-GPU host; the SIMBENCH_REQUIRE_LIVE_GPU
  test set admits, so the count is higher than a sandbox
  reviewer's 709/25 shape. Both shapes admit on a clean tree.

Loop 2 R22 evidence-refresh discipline (process documentation):

- The HRD bundle's `manifest_pairing_index.source_commit` and
  the run summary's `source_commit` are derived from the
  working-tree HEAD at the time the in-process probe runs.
  Because each round's commit advances HEAD, the operator
  rebases the live evidence after every commit by re-running
  the probe (`run_suite.py::_emit_hrd_bundle_for_run_summary`
  via `importlib.util.spec_from_file_location` against the
  existing `dram_stream_stride_mix` trace dir).
- The verifier (`workloads/simbench_bw/verify_artifacts.py::_verify_hrd_run_summaries`)
  enforces internal consistency between bundle, run summary,
  and Section 10 fields; it does NOT enforce
  source_commit-equals-HEAD because traces may legitimately
  outlive their original commit (e.g., when re-consumed by
  a future round-added analysis without re-tracing). The
  source_commit pinning is a Codex-review-time freshness
  check, not a verifier contract. The post-commit rebase is
  the operator's hook to keep evidence in lockstep with HEAD
  for each Codex review cycle.

### (f) Most recent sm_86-side evidence (v3 R10 capture, validated through R18)

Captured during the v3 RLCR loop's Round 10 closure on the RTX 3090 Ti
(NVIDIA driver 595.71.05, CUDA toolkit 13.2, NVBit 1.8, native
`g++-14` 14.3.1, compute capability 8.6, GPU UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`). The capture was
validated against the carried R8 trace at the R8/R9-aligned HEAD
`085230431ed08c47f4ae28b0e10caa86db4b9b8f`; subsequent rounds
(R14, R15, R16, R17) re-pinned the dram_stream_stride_mix sm_86
trace to advancing HEADs by post-commit recapture per
`BL-20260508-post-commit-recapture-discipline`. Codex's R10 review
accepted the evidence as `Verified-live-sm_86` for AC-6.

- **Trace directory**:
  `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00/`.
- **HRD bundle path**:
  `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00/hrd_profile/hrd_profile.json`.
- **Strict schema validation**: admits without errors via
  `jsonschema.Draft202012Validator` against
  `workloads/simbench_bw/schemas/hrd_profile.schema.json`.
- **Source-commit equality** (R10 capture record at HEAD
  `0852304`; rebases on every implementation commit per the
  abstract-HEAD pattern; R16's recapture re-pinned to
  `fc34bf2`; subsequent in-process probes update the field
  to current HEAD): 3-way `git rev-parse HEAD ==
  trace_trace_00.json.source_commit ==
  bundle.manifest_pairing_index.source_commit`.
- **Bundle structural shape**: `schema_version=1`,
  `result_kind="ok"`, `manifest_pairing_index` with all 5
  real fields populated; device sm_86 (CC 8.6, RTX 3090 Ti,
  84 SMs, 6 MiB L2); `cache_model_spec` with capacity
  `6291456` / sector 32B / line 128B / fully-associative-
  LRU-approximation / write-allocate-on-global-load /
  write-back-approximation / `l1_bypass=[".cg", "atomic"]`;
  binning unit=`sector_32b`, `bootstrap_resamples=2000`,
  `ci_confidence=0.95`, `ci_half_width_target=0.05`,
  `minimum_bin_sample_count=50`.
- **sm_86 HRD bin distribution baseline (R10 capture, 13 bins)**:
  - allocation: 2 ok / 0 unavailable
  - line: 2 ok / 2 unavailable (`cache_model_underconfigured`)
  - region: 0 ok / 1 unavailable (`region_reuse_cold`)
  - sector: 4 ok / 2 unavailable (`cache_model_underconfigured`)
  - 8 ok bins total + 5 typed-unavailable bins
- **Admitted-bin hit-rate distribution**: 4 bins at 0.0 +
  4 bins at 1.0; total observation_count across bins = 24577.
- **evidence[]**: 1 typed-unavailable record at scope=bundle
  with `typed_reason=cache_model_underconfigured`,
  `underconfigured_bin_count=4`, `scalar_hit_rate_refusal=true`.
- **Bundle-underconfigured three-way invariant** (from the
  verifier's `_recompute_hrd_bundle_section10`):
  `bin_implies_underconfigured=true ∧
  evidence_implies_underconfigured=true ∧
  cached_top_level_underconfigured=true ⇒
  three_way_agreement=true` (✓).
- **R12-style fail-closed verifier admission**:
  `verify_artifacts.py --results
  workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86`
  exits 0 (`ok`).
- **AC-6 v2 positive test**: HRD bundle admits via strict
  schema; bundle-underconfigured arm fires; three-way
  invariant holds.
- **AC-6 v2 negative-test arm coverage**:
  `pytest workloads/simbench_bw/tests/test_hrd_run_summary_verifier.py`
  -> 12 passed (covers cached-vs-recomputed digest mismatch
  + cached source_commit mismatch + schema rejection paths);
  `pytest workloads/simbench_bw/tests/test_verify_hrd_budget_exceeded.py`
  -> 3 passed.
- **HrdProfileTypedReasonsParity**:
  `pytest -k HrdProfileTypedReasonsParity tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  validates the 16-member typed-reason taxonomy lockstep
  across producer constants and schema enum.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the HRD profiler is
  arch-independent in implementation. The producer/schema/
  verifier surfaces remain in lockstep across both arches;
  the empirical bin distribution differs by L1/L2 sizing
  (RTX 3090 Ti L2=6 MiB vs RTX 5060 laptop L2 differs),
  which is the per-arch behavior CLAUDE.md predicts. The
  bundle-underconfigured arm fires on sm_86 (this Section)
  because of the 3090 Ti's L1/L2 sizing producing
  `cache_model_underconfigured` typed-unavailable bins;
  the same arm fires on sm_120 with a different empirical
  bin distribution per Section (e). Formal v3
  `Cross-arch-verified` status remains pending a future
  5060 sm_120 v3 session leg replaying the same producer
  against a fresh sm_120-laptop bundle.

---

## Entry: Loop 2 task12 — sm_86 Sharing-Locality Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task12-sharing-locality-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R22 / task12 / the
sharing-locality clause GPU-Shalom sharing-locality
classifier stack).

### (b) What is blocked
The sharing-locality clause's promotion from
`Verified-live-sm_120` to `Cross-arch-verified`. Loop 2 R22
captured the sm_120 evidence on the RTX 5060 laptop; the
matching sm_86 evidence requires re-running the live
classifier
`tools/sharing_locality/classifier.py::compute_sharing_locality`
against a 3090 Ti simbench trace and verifying the bundle
admits under strict schema validation with the same
typed-reason taxonomy.

### (c) Precise operator-facing artifact

- **Classifier module**:
  `python -c "from tools.sharing_locality.classifier import compute_sharing_locality"`
  (the canonical consumer is the in-process Python API).
- **Output schema**:
  `workloads/simbench_bw/schemas/sharing_locality.schema.json`.
  Top-level required fields:
  `schema_version`, `result_kind`, `manifest_pairing_index`,
  `device`, `grounded` (3 records), `projected` (4 records),
  `evidence`. Per-class records use `oneOf` between
  admitted-grounded / typed-unavailable-grounded /
  admitted-projected / typed-unavailable-projected branches;
  every closed branch under `additionalProperties: false`.
- **Manifest entries**: `tools/arch_capabilities.json`
  declares the new classifier module at
  `kind=feature_extractor` with
  `supported_arches=["sm_86", "sm_120-laptop"]`.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the classifier is
  arch-independent in implementation. The 7-member typed-reason
  taxonomy and the bundle's structural shape must not depend
  on the device that produced the trace.
- **Expected outcome**: at least one sm_86 simbench live
  trace runs end-to-end through the classifier and produces a
  sharing-locality bundle that admits via the schema under
  strict `jsonschema.Draft202012Validator` validation. The 7
  typed reasons emit on the sm_86 path with the same
  admission semantics as sm_120.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R22:
- `tools/sharing_locality/classifier.py` lands the producer
  module with `compute_sharing_locality(...)` returning a
  schema-conformant bundle dict (7 typed reasons in
  `SHARING_LOCALITY_TYPED_REASONS`).
- `workloads/simbench_bw/schemas/sharing_locality.schema.json`
  declares the bundle's structure with all 7 typed reasons in
  the shared `$defs.typed_reason.enum`, the disjoint
  `grounded_class_name` / `projected_class_name` enums, and
  the per-record `oneOf` invariants that prevent
  `provenance="grounded"` from appearing on a projected class.
- `tools/arch_capabilities.json` declares the new module at
  `kind=feature_extractor`.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` adds
  the `sharing_locality` family with two surfaces and the
  `SharingLocalityTypedReasonsParityTest` enforcer; all 7
  reasons land in `ROUND_ADDED_TYPED_REASONS`.
- `tools/sharing_locality/test_classifier.py` exercises the
  offline test suite: 3 grounded-class fixture tests + 8
  schema negative controls + 1 producer/schema parity test +
  1 strict round-trip admission test + 1
  projected-classes-typed-unavailable test + 1
  projected-classes-admitted-with-projection-inputs test.

### (e) Most recent sm_120-side evidence

Loop 2 R24 live evidence under the corrected R24 classifier
(R24 added CTA-keyed identity to `intra_lane` and
`intra_warp`; post-commit probe rebases
`manifest_pairing_index.source_commit` to the working-branch
HEAD; the operator probe re-derives the numbers below from
the in-process classifier):

- **Trace directory**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  (Driver 595.71.05, CUDA 13.2, RTX 5060 Laptop GPU, sm_120).
- **Artifact path**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/sharing_locality/sharing_locality.json`.
  Strict schema validation:
  `jsonschema.Draft202012Validator` admits without errors.
- **Grounded counts/fractions** (R23 corrected semantics):
  - `intra_lane`:
    `reuse_count=2640, distinct_pair_count=19841,
    fraction=0.133058`. About 13.3% of `(warp, lane, sector)`
    triples are revisited by the same lane within the launch.
  - `intra_warp`:
    `reuse_count=2048, distinct_pair_count=2049,
    fraction=0.999512`. Near-complete intra-warp coalescing —
    nearly every `(warp, sector)` has at least 2 distinct
    lanes within the warp touching the sector. Under the R23
    corrected semantics the count tracks `|distinct lanes per
    (warp_id, sector)| >= 2`, not access count, so this number
    is the genuine warp-level sharing rate.
  - `inter_warp_same_cta`:
    `reuse_count=0, distinct_pair_count=2049,
    fraction=0.000000`. No inter-warp same-CTA sharing in the
    streaming-workload kernel. Under the R23 corrected
    semantics the accumulator is per-CTA (per `.pb` file), so
    cross-CTA sectors are not conflated with same-CTA
    sharing; the workload genuinely has no warp-overlap
    within any threadblock at this allocation.
- **Projected typed-unavailable reasons** (no projection
  inputs supplied):
  - `inter_cta_same_sm`:
    `inter_cta_same_sm_projection_required`.
  - `inter_cta_cross_sm`:
    `inter_cta_cross_sm_projection_required`.
  - `sequential_kernel`:
    `sequential_kernel_projection_required`.
  - `private_cold`:
    `private_cold_projection_required`.
- **Evidence note**:
  `total_pairs = 23939`,
  `addresses_observed = 24577` (the
  `sharing_locality_pair_count_summary` evidence record).
- **Low-confidence flags**: none of the three grounded
  records has `low_confidence: true` — every grounded record
  has `distinct_pair_count > 0`.
- **Threadblock count**: the live
  `dram_stream_stride_mix` sm_120 trace contains exactly
  ONE threadblock under
  `threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb`.
  (R24 named `0_0_0.pb` as the filename; that was a
  documentation typo Codex's R24 review caught — the actual
  on-disk filename is `d_0_s_0_k_1_0,0,0.pb`.) The
  R24 CTA-keyed accumulators (`intra_lane`, `intra_warp`)
  produce IDENTICAL numbers to the R23 globally-keyed
  accumulators on this trace because there is no second
  CTA whose local warp/lane IDs could collide. The R24
  corrections lock the boundaries under negative-control
  tests
  (`test_two_threadblocks_same_local_lane_no_intra_lane`,
  `test_two_threadblocks_same_local_warp_different_lanes_no_intra_warp`)
  so multi-CTA workloads cannot drift into false-positive
  semantics.

---

## Entry: Loop 2 task13 — sm_86 Order-Projections Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task13-order-projections-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R25 / task13 / the
order-projections clause schedule-marginalized order
projections producer).

### (b) What is blocked
The order-projections clause's promotion from
`Verified-live-sm_120` to `Cross-arch-verified`. Loop 2 R25
captured the sm_120 evidence on the RTX 5060 laptop; the
matching sm_86 evidence requires re-running the live
projector
`tools/order_projections/projector.py::compute_order_projections`
against a 3090 Ti simbench trace and verifying the bundle
admits under strict schema validation with the same
typed-reason taxonomy.

### (c) Precise operator-facing artifact

- **Projector module**:
  `python -c "from tools.order_projections.projector import compute_order_projections"`
  (canonical consumer; no CLI wrapper today).
- **Output schema**:
  `workloads/simbench_bw/schemas/order_projections.schema.json`.
  Top-level required fields:
  `schema_version`, `result_kind`, `manifest_pairing_index`,
  `device`, `projections` (6 records), `disagreement_variance`
  (numeric or typed-unavailable), `evidence`. Per-projection
  records use `oneOf` between admitted / typed-unavailable
  branches; every closed branch under
  `additionalProperties: false`. The original-plan
  order-projections-clause negative invariant (a bundle
  missing `disagreement_variance` fails the schema) is
  enforced by the top-level `required`.
- **Manifest entries**: `tools/arch_capabilities.json`
  declares the new projector module at
  `kind=feature_extractor` with
  `supported_arches=["sm_86", "sm_120-laptop"]`.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the projector is
  arch-independent in implementation. The 7-member
  typed-reason taxonomy and the bundle's structural shape
  must not depend on the device that produced the trace.
- **Expected outcome**: at least one sm_86 simbench live
  trace runs end-to-end through the projector and produces
  an order-projections bundle that admits via the schema
  under strict `jsonschema.Draft202012Validator`
  validation. The 7 typed reasons emit on the sm_86 path
  with the same admission semantics as sm_120.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R25 (initial landing) + R26 (typed-reason +
controller/verifier wiring rework) + R27 (controller-path
regression test + ledger refresh):
- `tools/order_projections/projector.py` lands the producer
  module with `compute_order_projections(...)` returning a
  schema-conformant bundle dict.
- Producer-side typed-reason set
  `ORDER_PROJECTIONS_TYPED_REASONS` declares 7 reasons. R26
  restored the SM-local unavailable arm to the
  original-plan exclusion-taxonomy enumerated reason
  `projection_unavailable_sm_local` (it lives in the
  parity scanner's `AC19_PLAN_ENUMERATED_TYPED_REASONS`).
  The remaining 6 reasons
  (`lane_projection_unobservable`,
  `warp_local_projection_unobservable`,
  `cta_local_projection_unobservable`,
  `kernel_l2_projection_unobservable`,
  `sequential_allocation_projection_unobservable_no_allocations_sidecar`,
  `insufficient_admitted_projections`) are round-added.
- `workloads/simbench_bw/schemas/order_projections.schema.json`
  declares the bundle's structure with all 7 typed reasons
  in the shared `$defs.typed_reason.enum`, the 6-element
  `projection_name` enum, and the per-projection +
  disagreement-variance `oneOf` invariants. Top-level
  `required` includes `disagreement_variance` (the
  original-plan negative-test invariant).
- `tools/arch_capabilities.json` declares the new module
  at `kind=feature_extractor`.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  carries the `order_projections` family with two surfaces
  and the `OrderProjectionsTypedReasonsParityTest`
  enforcer. The R26 parity test
  `test_overlap_with_ac19_plan_enumeration` asserts overlap
  equals exactly `{"projection_unavailable_sm_local"}`.
- `tools/order_projections/test_projector.py` exercises
  the offline test suite: 6 per-projection fixtures, 1
  deterministic-single-CTA agreement test, 6 schema
  negative controls, 1 producer/schema parity test, 1
  strict round-trip admission test, and a typed-unavailable
  variance arm probe — totaling 19 tests.
- **R26 controller + verifier wiring**:
  - `workloads/simbench_bw/run_suite.py::_emit_order_projections_for_run_summary`
    is the controller helper, called from `handle_trace`
    immediately after the HRD bundle is emitted. Reuses
    the same manifest-pairing index, device block, and L2
    capacity the HRD path uses; writes the bundle to
    `<trace_dir>/order_projections/order_projections.json`
    during the normal trace flow; persists 9 Section-
    equivalent fields into the run summary
    (`order_projections_path`, `_schema_version`,
    `_source_commit`, `_admitted_count`,
    `_typed_unavailable_count`,
    `_disagreement_variance_result_kind`, `_value`,
    `_typed_reason`, `_total_access_events`). Skip path
    with typed `order_projections_skip_reason` on
    producer/capacity failure.
  - `workloads/simbench_bw/verify_artifacts.py::_verify_order_projections_run_summaries`
    (with `_recompute_order_projections_section`) is
    wired into `verify_trace_dir`. Strict-validates the
    bundle, checks
    `manifest_pairing_index.{slug,shape_point,source_commit}`
    equality with the run summary, recomputes Section
    fields from `projections[]` + `disagreement_variance`,
    rejects cached-vs-bundle disagreement.
- **R26 + R27 verifier and controller tests**:
  - `workloads/simbench_bw/tests/test_order_projections_run_summary_verifier.py`
    lands 9 verifier tests (positive control + missing
    bundle file + schema-invalid + source-commit mismatch
    + slug mismatch + cached disagreement-variance value
    mismatch + cached admitted-count mismatch +
    typed-unavailable variance arm + typed-unavailable
    arm with no `order_projections_path`).
  - `workloads/simbench_bw/tests/test_order_projections_run_summary_controller.py`
    (R27) lands 3 controller-path tests: import
    `run_suite.py` in-process via importlib, monkeypatch
    `sys.modules['projector']` with a deterministic
    schema-valid bundle, call
    `_emit_order_projections_for_run_summary` directly,
    assert (a) the bundle file is written, (b) the
    on-disk JSON strict-validates, (c) the run-summary
    fields match the bundle, (d) the projector-raises
    skip path records a typed
    `order_projections_skip_reason`, and (e) the
    end-to-end helper-then-verifier subprocess sweep
    accepts the artifact tree.

### (e) Most recent sm_120-side evidence

Loop 2 R26 + R27 live evidence (the post-commit probe
re-emits both the HRD bundle and the order-projections
bundle so `manifest_pairing_index.source_commit` rebases
to the new HEAD on every commit):

- **Trace directory**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  (Driver 595.71.05, CUDA 13.2, RTX 5060 Laptop GPU,
  sm_120; the same single-threadblock trace task11/task12
  consume).
- **Artifact path**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/order_projections/order_projections.json`.
  Strict schema validation:
  `jsonschema.Draft202012Validator` admits without errors.
- **Run-summary `order_projections_*` fields** persist
  alongside the HRD Section-10 fields in
  `trace_trace_00.json`:
  - `order_projections_path = order_projections/order_projections.json`
  - `order_projections_schema_version = 1`
  - `order_projections_source_commit` = working-branch HEAD
    (the post-commit probe rebases on every commit)
  - `order_projections_admitted_count = 6`
  - `order_projections_typed_unavailable_count = 0`
  - `order_projections_disagreement_variance_result_kind = "ok"`
  - `order_projections_disagreement_variance_value = 0.0`
    (the deterministic-single-CTA agreement arm of the
    original-plan order-projections clause's positive test)
  - `order_projections_disagreement_variance_typed_reason = null`
  - `order_projections_total_access_events = 24577`
- **Verifier**:
  `python workloads/simbench_bw/verify_artifacts.py
  --results workloads/simbench_bw/l1/dram_stream_stride_mix/traces`
  exits 0; the new
  `_verify_order_projections_run_summaries` path is
  exercised in addition to the existing C-AC-H + HRD
  surfaces.

---

## Entry: Loop 2 task15 — sm_86 Perturbation-Harness Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task15-perturbation-harness-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R31 / task15 / the
perturbation-harness clause cache-tomography perturbation
harness with SASS modifier confirmation + correctness
equivalence).

### (b) What is blocked
The perturbation-harness clause's promotion from
`Verified-live-sm_120` to `Cross-arch-verified`. Loop 2 R31
captures the sm_120 evidence on the RTX 5060 laptop; the
matching sm_86 evidence requires re-running the harness
against a 3090 Ti `dram_stream_stride_mix_run` baseline
binary with the sm_86 codegen flag set, confirming the
SASS modifier audit on Ampere SASS (the cuobjdump dump
format is shared, but the cache-modifier mnemonic differs
across architectures: Ampere emits literal `LDG.E.CA` /
`LDG.E.CG` while Blackwell with CUDA 13.x emits
`LDG.E.STRONG.SM` / `LDG.E.STRONG.GPU` for the same PTX
hints — the producer-side regex tolerates both encodings
per `BL-20260505-cross-arch-sass-mnemonic-divergence`),
and verifying the bundle admits under strict schema
validation.

### (c) Precise operator-facing artifact

- **Operator entry point**:
  `python tools/perturbation_harness/build.py --slug
  dram_stream_stride_mix --arch sm_86 --out
  workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<trace>/perturbation_run_ledger/perturbation_run_ledger.json`
  (the sm_86 follow-up requires extending the harness's
  `--arch` enum to admit `sm_86` and emitting the matching
  `-gencode arch=compute_86,code=sm_86` build flags; the
  Loop 2 R31 implementation pins `sm_120-laptop` only).
- **Output schema**:
  `workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`.
  Top-level required: `schema_version`, `result_kind`,
  `manifest_pairing_index`, `device`, `perturbations`,
  `evidence`. Per-record `oneOf` between an admitted branch
  and four strict typed-unavailable branches keyed by
  `typed_reason` (`perturbation_compile_failed`,
  `modifier_optimized_away`, `correctness_equivalence_failed`,
  `arch_feature_unavailable_*`). Cache-modifier admission
  requires `sass_match_count >= sass_expected_match_count = 2`
  inside the target kernel SASS section. Per-mode
  `correctness_delta` `oneOf` enforces required fields per
  `comparison_mode` (`abs_and_rel`, `exact_match`,
  `sha256_f32le`).
- **Manifest entries**: `tools/arch_capabilities.json`
  declares the new modules at
  `kind=operator_entry_point` (`build.py`) /
  `kind=feature_extractor`
  (`sass_confirm.py`, `correctness_check.py`, `ledger.py`)
  with `supported_arches=["sm_120-laptop"]`. The sm_86
  follow-up extends the supported_arches to `["sm_86",
  "sm_120-laptop"]` once the live Ampere evidence has been
  captured.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the harness's
  Python orchestration layer is arch-independent; the
  sm_86 path differs only in the nvcc `-gencode` flag set
  and the cuobjdump SASS opcode coverage (Ampere does not
  have Blackwell's `UTMA*` paths but does carry the same
  `LDG.E.{CA,CG}` cache-modifier semantics on the target
  global-memory loads). The 9-record loop-scope set
  (`cache_modifier_ca`, `cache_modifier_cg`, 7
  `working_set_sweep` MiB values) MUST emit on sm_86 with
  the same admission semantics.
- **Expected outcome**: at least one sm_86 simbench live
  trace runs end-to-end through the harness and produces
  a `perturbation_run_ledger.json` admitting via the strict
  schema. Either all 9 records admit (the positive control)
  or any binary that fails admission lands in the
  appropriate typed-unavailable arm with full evidence per
  the schema's per-branch `required` array. The verifier's
  `_verify_perturbation_run_ledger_run_summaries` recomputes
  the SASS confirmation hash from the on-disk dump and
  rejects any cached-vs-recomputed disagreement.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R31 (initial landing):
- `tools/perturbation_harness/build.py` lands the
  orchestrator that generates perturbation `.cu` files
  (cache-modifier variants inject the `simbench_ld_global_{ca,cg}_f32`
  `asm volatile` helper and route both target source loads
  `in[idx]` and `in[i]` through it) and compiles each via
  nvcc against `-gencode arch=compute_120,code=sm_120`.
- `tools/perturbation_harness/sass_confirm.py` lands the
  kernel-scoped SASS modifier audit
  (`cuobjdump --dump-sass` + `LDG.E.{CA,CG}` regex
  inside the target kernel's `Function : <mangled>`
  section + sha256 over normalized matching lines).
  Cache-modifier admission requires
  `sass_match_count >= sass_expected_match_count = 2`;
  the synthetic one-of-two negative-control fixture in
  `test_sass_confirm.py` regression-locks the boundary.
- `tools/perturbation_harness/correctness_check.py`
  lands the per-mode delta computation
  (`abs_and_rel` / `exact_match` / `sha256_f32le`) with
  pinned per-dtype tolerances (fp32 abs=1e-5/rel=1e-4;
  fp16 abs=1e-3/rel=1e-2; bf16 abs=1e-2/rel=5e-2;
  int8/int32 exact-match).
- `tools/perturbation_harness/ledger.py` declares the
  producer typed-reason constant
  `PERTURBATION_RUN_LEDGER_TYPED_REASONS` (the C-AC-C
  parity inventory's `producer_python_constant` surface)
  plus `build_bundle(...)` which strict-validates against
  the schema before returning.
- `workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`
  declares the bundle's structure with the 9-element
  typed-reason enum, the per-variant `family` enum, the
  per-mode `correctness_delta` `oneOf`, and the
  `additionalProperties: false` invariant on every closed
  branch.
- `workloads/simbench_bw/run_suite.py` adds
  `_emit_perturbation_run_ledger_for_run_summary` invoked
  from `handle_trace` immediately after the order-projections
  emit; persists 8 run-summary fields plus a typed
  `perturbation_run_ledger_skip_reason` on producer
  failure.
- `workloads/simbench_bw/verify_artifacts.py` adds
  `_verify_perturbation_run_ledger_run_summaries` +
  `_recompute_perturbation_run_ledger_section` +
  `_recompute_sass_confirmation_hash` +
  `_correctness_delta_exceeds_tolerance` +
  `_perturbation_typed_reason_present`. The verifier
  recomputes the cached fields from the bundle's
  `perturbations[]`, recomputes each admitted record's
  SASS hash from the on-disk dump at `sass_artifact_path`,
  and rejects any cached-vs-recomputed disagreement.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  adds the `perturbation_run_ledger` family with two
  surfaces (producer constant + JSON schema enum); the
  family's expected typed-reason set names
  `modifier_optimized_away` as the AC-19 plan-enumerated
  reason and the other 8 (2 pure round-added + 6
  wildcard-family `arch_feature_unavailable_*`) as
  round-added.
- `tools/arch_capabilities.json` declares the 4 new
  module entries.

### (e) Most recent sm_120-side evidence

Captured during R33 closure (and refreshed at R34) on the
RTX 5060 Laptop GPU (NVIDIA driver 595.71.05, CUDA toolkit
13.2, compute capability 12.0). R33 closed Codex's R32
review action items (producer-side `working_set_sweep`
element_count invariant for `abs_and_rel` mode; verifier-
side working-set parameter metadata + comparison_mode +
element_count invariants; refreshed test fixtures with
real per-mib element counts; three negative-control tests
that prove the invariants reject the R31 wrong-shape
class). R34 closed Codex's R33 review action item by
porting this Section (e) to the abstract-HEAD pattern
task13 Section (e) uses (line 1380-1381 of this file):
the post-commit probe regenerates the
perturbation_run_ledger bundle so
`manifest_pairing_index.source_commit` rebases to the new
HEAD on every commit. The reviewer's
`source_commit` value is the working-branch HEAD they
inspect; no literal SHA is pinned in this section because
the documentation source-of-truth is the rebase
discipline, not a frozen commit identifier.

- **Trace directory**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`.
- **Artifact path**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/perturbation_run_ledger/perturbation_run_ledger.json`.
- **Strict schema validation**: admits without errors via
  `jsonschema.Draft202012Validator` against
  `workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`
  (R32 schema raises `perturbations.minItems` from 1 to 9;
  the verifier additionally enforces the exact 9-parameter
  default loop set).
- **Bundle counts** (the loop-scope positive control):
  - `perturbations.length = 9`
  - `cache_modifier_ca` admitted: 1 (sass_match_count=12 inside
    the target kernel SASS section, expected=2;
    `correctness_delta.element_count = 128`,
    `comparison_mode = "abs_and_rel"`)
  - `cache_modifier_cg` admitted: 1 (sass_match_count=12 inside
    the target kernel SASS section, expected=2;
    `correctness_delta.element_count = 128`,
    `comparison_mode = "abs_and_rel"`)
  - `working_set_sweep` admitted: 7 (one per MiB ladder
    value `{1, 2, 4, 8, 16, 32, 64}`; sass_match_count
    matches the per-shape LDG count produced by the int64
    indexing kernel; per-record `correctness_delta`
    uses the canonical `element_count = working_set_mib *
    262144` matching the parameter table:
    `{262144, 524288, 1048576, 2097152, 4194304,
    8388608, 16777216}`. The 1/2/4/8 MiB rows use
    `comparison_mode = "abs_and_rel"`; the 16/32/64 MiB
    rows use `comparison_mode = "sha256_f32le"` per the
    task14 large-output digest contract).
  - typed-unavailable: 0
  - sass_confirmation_pass_count: 9
  - correctness_equivalence_pass_count: 9
- **Per-record `correctness_delta`**: cache_modifier
  variants and the four small-MiB working_set_sweep rows
  use `abs_and_rel` mode with `dtype = "fp32"`, `abs_tol
  = 1e-5`, `rel_tol = 1e-4`, `max_abs_delta = 0.0`,
  `max_rel_delta = 0.0` against the per-shape
  deterministic baseline reference cached under
  `<perturbation_build>/reference/`. The three large-MiB
  rows use `sha256_f32le` digest mode with
  `actual_sha256 == reference_sha256` (digest match per
  the task14 spec).
- **Run-summary `perturbation_*` fields** persist in
  `trace_trace_00.json` alongside the HRD Section-10
  and order-projections fields:
  - `perturbation_run_ledger_path = perturbation_run_ledger/perturbation_run_ledger.json`
  - `perturbation_run_ledger_schema_version = 1`
  - `perturbation_run_ledger_source_commit` = working-branch HEAD
    (the post-commit probe rebases on every commit)
  - `perturbation_admitted_count = 9`
  - `perturbation_typed_unavailable_count = 0`
  - `perturbation_per_family_admitted_count = {cache_modifier_ca: 1, cache_modifier_cg: 1, working_set_sweep: 7}`
  - `perturbation_typed_unavailable_breakdown = {modifier_optimized_away: 0, correctness_equivalence_failed: 0, arch_feature_unavailable_l2_prefetch_64b: 0, arch_feature_unavailable_l2_prefetch_128b: 0, arch_feature_unavailable_l2_prefetch_256b: 0, arch_feature_unavailable_l2_persisting: 0, arch_feature_unavailable_eviction_buffer: 0, arch_feature_unavailable_carveout: 0, perturbation_compile_failed: 0}`
  - `sass_confirmation_pass_count = 9`
  - `correctness_equivalence_pass_count = 9`
- **Verifier**:
  `python workloads/simbench_bw/verify_artifacts.py
  --results workloads/simbench_bw/l1/dram_stream_stride_mix/traces`
  exits 0; the
  `_verify_perturbation_run_ledger_run_summaries` path
  exercises strict schema admission, the per-mode
  `_correctness_delta_exceeds_tolerance` check, the
  per-record SASS hash recomputation, the R32
  default-loop 9-parameter set invariant, and the new R33
  working_set_sweep metadata invariants
  (`parameter.{allocation_bytes, warps, accesses,
  stride_classes}` per the per-mib table;
  `correctness_delta.element_count = mib * 262144`;
  `comparison_mode = sha256_f32le` for `mib >= 16` and
  `abs_and_rel` otherwise). Three negative-control tests
  in
  `test_perturbation_run_ledger_run_summary_verifier.py`
  prove the new invariants reject the R31 wrong-element-
  count, wrong-mode, and drifted-`accesses` failure
  classes.
- **Cross-arch finding (Plan Evolution per R31, retained
  in R32)**: CUDA 13.2 emits Blackwell-specific
  cache-modifier mnemonics (`LDG.E.STRONG.SM` for `.ca`
  PTX; `LDG.E.STRONG.GPU` for `.cg` PTX) rather than the
  Ampere-style `LDG.E.{CA,CG}` literal mnemonics. The
  perturbation-harness clause's SASS regex accepts BOTH
  encodings so the producer-side wire format stays
  architecture-independent per CLAUDE.md
  "Two-Device Local Development Model" and per
  `BL-20260505-cross-arch-sass-mnemonic-divergence`.

---

## Entry: Loop 2 task17 — sm_86 Adaptive-Sampler Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task17-adaptive-sampler-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 R37 / task17 / AC-11
Condens-style adaptive NVBit sampler clause: cumulative-
window MRC at sector + line layers, `K=3` consecutive-window
stationary stopping rule with `distance_threshold = 0.02`
hit-rate-fraction).

### (b) What is blocked
The adaptive-sampler clause's promotion from
`Verified-live-sm_120` to `Cross-arch-verified`. Loop 2 R37
captures the sm_120 evidence on the RTX 5060 laptop; the
matching sm_86 evidence requires re-running the controller
helper `_emit_trace_convergence_confidence_for_run_summary`
against a 3090 Ti `dram_stream_stride_mix` trace +
upstream `hrd_profile.json` produced under sm_86, and
verifying the bundle admits under strict schema validation.
The producer is pure Python and architecturally
arch-independent in implementation, so the cross-arch test
exercises HRD-bundle parity (sm_86 producer emits the same
`bins[]` shape per
`workloads/simbench_bw/schemas/hrd_profile.schema.json`
admitted-bin branch: `result_kind == "ok"`,
`hierarchy_layer ∈ {sector, line}`, integer
`rd_sector_units_min/max`, integer `observation_count > 0`,
integer `hit_count >= 0`, numeric `hit_rate`; at least one
admitted record per layer) and confirms the bundle's
`manifest_pairing_index.binary_sha256` rebases to the sm_86
binary. The shared `is_usable_admitted_bin` /
`usable_admitted_bins` adapter in
`tools/adaptive_sampler/trace_convergence_ledger.py` is the
single source of truth for the admitted-bin contract that
the producer module, the operator CLI, and the controller
helper all consume.

### (c) Precise operator-facing artifact

- **Operator entry point**:
  `python tools/adaptive_sampler/run_sampler.py
  --hrd-bundle workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<trace>/hrd_profile/hrd_profile.json
  --out workloads/simbench_bw/l1/dram_stream_stride_mix/traces/<trace>/convergence_confidence/convergence_confidence.json
  --arch sm_86 --slug dram_stream_stride_mix
  --shape-point trace_00 --binary-sha256 <hex>
  --source-commit <hex> --kernel-signature-sha256 <hex>
  --device-name <name> --cc-major 8 --cc-minor 6
  --sm-count 84 --l2-cache-size-bytes 6291456`. R41 widened
  the CLI's `_SUPPORTED_ARCHES` tuple to admit both
  `sm_86` and `sm_120-laptop` structurally, and updated
  `tools/arch_capabilities.json` `supported_arches` for
  the three `tools/adaptive_sampler/*` entries to
  `["sm_86", "sm_120-laptop"]` so the documented
  `--arch sm_86` shape parses against the live argparse
  contract per BL-20260504-runbook-command-verification.
  Live v3 sm_86 evidence captured in v3 R15 (validated
  against current reviewed HEAD; live bundle admits via
  the `hrd_layer_unavailable` typed-unavailable arm with
  `admission.bundle_underconfigured=true`,
  `admission.unavailable_layer="line"`,
  `admission.usable_bin_count=2`; cached
  `trace_convergence_windows_digest` round-trips through
  `verify_artifacts.py`). Section (f) below records the
  v3 sm_86 evidence; Section (e) preserves the sm_120
  baseline. The remaining `Cross-arch-verified` work is
  a future sm_120 v3 leg replaying the same producer
  against a fresh sm_120-laptop HRD bundle on the 5060
  laptop. Server architectures (sm_80, sm_89, sm_90a,
  server sm_120) remain rejected at argparse per the
  C-AC-E Loop 2 scope guard. The controller path
  (`workloads/simbench_bw/run_suite.py::_emit_trace_convergence_confidence_for_run_summary`)
  is arch-independent in implementation and runs on
  either device.
- **Output schema**:
  `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`.
  Top-level required: `schema_version`,
  `manifest_pairing_index`, `device`, `hrd_profile_path`,
  `mrc_quantiles`, `distance_metric`,
  `k_consecutive_windows`, `distance_threshold`,
  `trace_window_size_bytes`, `windows`, `admission`,
  `source_commit`. Per-window record requires
  `window_index`, `observation_count_cumulative`,
  `trace_bytes_cumulative`, `mrc_at_quantile`,
  `distance_to_previous`, `converged`. The `admission`
  `oneOf` enforces a strict admitted branch
  (`{result_kind: ok, windows_converged, final_residual}`)
  versus FIVE typed-unavailable branches keyed by
  `typed_reason`
  (`trace_convergence_not_reached_budget_cap` —
  plan-enumerated; `mrc_quantile_underconfigured`,
  `hrd_layer_unavailable`, `trace_convergence_disabled`,
  `trace_convergence_window_count_underconfigured` —
  round-added). The 5th reason
  `trace_convergence_window_count_underconfigured`
  was added in R38 to honestly name missing
  iterative cumulative-window evidence rather than
  fabricating convergence by duplicating one HRD bundle.
  Every closed branch under `additionalProperties: false`.
- **Manifest entries**: `tools/arch_capabilities.json`
  declares the 3 new modules at
  `kind=operator_entry_point` (`run_sampler.py`) /
  `kind=feature_extractor`
  (`sampler.py`, `trace_convergence_ledger.py`) with
  `supported_arches=["sm_86", "sm_120-laptop"]` (R41
  widened from `["sm_120-laptop"]` so the runbook's
  documented `--arch sm_86` offline command parses against
  the live argparse contract per
  BL-20260504-runbook-command-verification). Live v3
  sm_86 evidence captured in v3 R15 (see Section (f)
  below); the remaining `Cross-arch-verified` work is a
  future sm_120 v3 leg.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the adaptive
  sampler's Python producer layer is arch-independent;
  the sm_86 path differs only in the upstream
  `hrd_profile.json` payload (Ampere-side reuse-distance
  bins vs. Blackwell-side bins) and the
  `device.compute_capability_*` / `device.l2_cache_size_bytes`
  values inside the bundle. The five typed-unavailable
  reasons are emitted by the same admission helpers on
  both architectures. An sm_86 trace with a fully-
  populated HRD bundle that supplies usable per-layer
  admitted bins (per the shared
  `is_usable_admitted_bin` adapter) AND iterative
  cumulative-window evidence covering at least
  `(K_CONSECUTIVE_WINDOWS + 1) * TRACE_WINDOW_SIZE_BYTES`
  raw bytes will admit via the `result_kind=ok` arm
  with the same cumulative-window MRC reduction at
  sector + line layers. Until iterative cumulative-
  window evidence is wired into the producer, the
  same fully-populated HRD bundle admits via the
  `trace_convergence_window_count_underconfigured`
  arm — the typed-unavailable arm honestly names the
  missing evidence rather than fabricating
  convergence from a single window.

### (d) sm_120-side prerequisites completed in Loop 2

At Loop 2 R37 (initial landing) and refreshed in R38 / R39 / R40:
- `tools/adaptive_sampler/sampler.py` lands the
  cumulative-window MRC reduction
  (`compute_mrc_at_quantile`, `iterate_windows`,
  `trailing_converged_count`,
  `convergence_admission_for`) plus the five typed-
  unavailable admission helpers
  (`budget_cap_admission`,
  `mrc_quantile_underconfigured_admission`,
  `hrd_unavailable_admission`,
  `disabled_admission`,
  `window_count_underconfigured_admission`). The 5th
  helper (`window_count_underconfigured_admission`) was
  added in R38 to emit the
  `trace_convergence_window_count_underconfigured`
  typed-unavailable arm when usable per-layer admitted
  bins exist but iterative cumulative-window evidence
  does not. Each helper returns the `typed_reason`
  string literal pinned by
  `TRACE_CONVERGENCE_TYPED_REASONS`.
  R39 added a shared `is_usable_admitted_bin` /
  `usable_admitted_bins` adapter in
  `trace_convergence_ledger.py`; R40 routed the
  controller helper
  (`workloads/simbench_bw/run_suite.py::_emit_trace_convergence_confidence_for_run_summary`)
  through the same adapter so all three call sites
  (controller helper, `compute_mrc_at_quantile`,
  `run_sampler.py`) share one filter.
- `tools/adaptive_sampler/trace_convergence_ledger.py`
  declares the producer typed-reason constant
  `TRACE_CONVERGENCE_TYPED_REASONS: frozenset[str]`
  (the C-AC-C parity inventory's `producer_constants`
  surface) plus the canonical `MRC_QUANTILES` tuple
  `(0.10, 0.25, 0.50, 0.75, 0.90)`,
  `DISTANCE_THRESHOLD = 0.02`,
  `K_CONSECUTIVE_WINDOWS = 3`,
  `TRACE_WINDOW_SIZE_BYTES = 16777216`,
  `canonicalize_windows`, the canonical-JSON sha256
  `windows_digest` helper, and `build_bundle(...)` which
  strict-validates against the schema before returning.
- `tools/adaptive_sampler/run_sampler.py` lands the
  operator CLI for offline/regression-test runs.
  R41 widened `_SUPPORTED_ARCHES` from
  `(TARGET_ARCH,)` to `("sm_86", TARGET_ARCH)` so both
  local-development arches are admitted structurally
  by the live argparse contract; this matches the
  documented `--arch sm_86` shape in
  `tools/adaptive_sampler/TASK17_SM86_RUNBOOK.md` § 1
  per `BL-20260504-runbook-command-verification`.
  Section (e) records the sm_120 Loop 2 baseline;
  Section (f) records the v3 sm_86 evidence captured
  in v3 R15. Server architectures remain rejected at
  argparse per the C-AC-E Loop 2 scope guard.
- `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`
  declares the bundle's structure with the per-mode
  `admission` `oneOf` (1 admitted branch + 5 typed-
  unavailable branches: budget_cap +
  mrc_quantile_underconfigured + hrd_layer_unavailable +
  trace_convergence_disabled +
  trace_convergence_window_count_underconfigured),
  the per-window-record `oneOf` shape, and the
  `additionalProperties: false` invariant on every
  closed branch.
- `workloads/simbench_bw/run_suite.py` adds
  `_emit_trace_convergence_confidence_for_run_summary`
  invoked from `handle_trace` immediately after the
  perturbation-harness emit; persists 16 run-summary
  fields (16 with the cached `windows_digest` and the 5
  cached manifest-tuple fields) plus a typed
  `trace_convergence_skip_reason` on producer failure.
- `workloads/simbench_bw/validate.py` adds
  `validate_trace_convergence_confidence` and the
  discovery `rglob("convergence_confidence.json")`
  routing into the new validator.
- `workloads/simbench_bw/verify_artifacts.py` adds
  `_verify_trace_convergence_run_summaries` plus the
  `_trace_convergence_canonical_windows_digest` and
  `_trace_convergence_trailing_converged_count`
  recomputation helpers. The verifier recomputes the
  cached fields from the bundle's `windows[]` rather
  than trusting the cached field per the
  R7/R9/R12/R13 strict-schema lesson family;
  cached-vs-recomputed disagreement is
  `BudgetViolationUntracked`-equivalent.
- `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`
  adds the `trace_convergence` family with 4 surfaces
  (producer constant + run_suite writer helper +
  validate.py dispatch + JSON schema oneOf consts);
  the family's expected typed-reason set names
  `trace_convergence_not_reached_budget_cap` as the
  AC-19 plan-enumerated reason and the other 4
  (`mrc_quantile_underconfigured`,
  `hrd_layer_unavailable`,
  `trace_convergence_disabled`,
  `trace_convergence_window_count_underconfigured`) as
  round-added. R38 added the 5th typed reason
  `trace_convergence_window_count_underconfigured` to
  honestly name the missing evidence when the trace is
  admittedly sized but does not have enough cumulative
  bytes for K consecutive windows; this replaced the
  R37-era synthetic `[cumulative] * 4` admission path
  Codex's R37 review rejected.
- `tools/adaptive_sampler/TASK17_SM86_RUNBOOK.md` lands
  the operator-facing 5-section procedure for the sm_86
  cross-arch follow-up (invocation, expected HRD input,
  output path, verifier command, budget-cap handling) per
  `BL-20260504-runbook-self-reference` +
  `BL-20260504-runbook-command-verification`.
- `tools/arch_capabilities.json` declares the 3 new
  module entries.

### (e) Most recent sm_120-side evidence

Captured during R38 closure on the RTX 5060 Laptop GPU
(NVIDIA driver 595.71.05, CUDA toolkit 13.2, compute
capability 12.0). R37 ported this Section (e) to the
abstract-HEAD pattern up-front per
`BL-20260504-runbook-self-reference`; R38 reworked the
recorded evidence to match the actual on-disk artifact
after the R37 review identified that the prior section
claimed a fabricated `result_kind=ok` admission. The
post-commit probe regenerates the
trace_convergence_confidence bundle so
`manifest_pairing_index.source_commit` rebases to the
new HEAD on every commit. The reviewer's `source_commit`
value is the working-branch HEAD they inspect; no
literal SHA is pinned in this section because the
documentation source-of-truth is the rebase discipline,
not a frozen commit identifier.

- **Trace directory**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`.
- **Artifact path**:
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/convergence_confidence/convergence_confidence.json`.
- **Strict schema validation**: admits without errors via
  `jsonschema.Draft202012Validator` against
  `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`.
- **Bundle shape** (the typed-unavailable controller-path
  evidence; the live HRD bundle reports
  `bundle_underconfigured = True` because the bootstrap
  CI half-width exceeds the configured threshold, so the
  controller emits the honest `hrd_layer_unavailable`
  arm rather than admitting `result_kind=ok` from
  insufficient HRD evidence):
  - `mrc_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]`
    (the canonical 5-quantile pin)
  - `distance_metric = "max_abs_mrc_quantile_delta"`
  - `k_consecutive_windows = 3`
  - `distance_threshold = 0.02`
  - `trace_window_size_bytes = 16777216` (16 MiB)
  - `windows.length = 1` (the single baseline window;
    the producer no longer duplicates HRD bundles)
  - `admission.result_kind = "typed_unavailable"`
  - `admission.typed_reason = "hrd_layer_unavailable"`
  - `admission.bundle_underconfigured = true` (the
    upstream HRD bundle's bootstrap CI half-width
    exceeded the configured threshold)
  - `admission.unavailable_layer ∈ {"sector", "line"}`
    (the controller picks the first underconfigured
    layer)
  - The controller path's decision tree fires the
    honest typed-unavailable arm. Once the upstream
    HRD bundle's CI half-width drops below threshold
    (more observations or wider bins), the controller
    re-routes to the next typed-unavailable arm
    (`trace_convergence_window_count_underconfigured`
    until iterative cumulative-window evidence is
    implemented; then `result_kind=ok` once K trailing
    converged windows are observed).
- **Run-summary `trace_convergence_*` fields** persist in
  `trace_trace_00.json` alongside the HRD,
  order-projections, and perturbation-harness fields.
  The R38 controller helper inherits the
  manifest_pairing_index from the upstream HRD bundle's
  `manifest_pairing_index`, so the convergence sidecar's
  identity tuple matches the trace's identity verbatim
  (the verifier compares all 5 identity fields against
  the enclosing run summary):
  - `trace_convergence_confidence_path = convergence_confidence/convergence_confidence.json`
  - `trace_convergence_schema_version = 1`
  - `trace_convergence_admitted = false`
  - `trace_convergence_typed_reason = "hrd_layer_unavailable"`
  - `trace_convergence_windows_observed = 1`
  - `trace_convergence_windows_converged = 0`
  - `trace_convergence_final_residual = null`
  - `trace_convergence_mrc_quantile_set = [0.10, 0.25, 0.50, 0.75, 0.90]`
  - `trace_convergence_source_commit` = the trace's
    `source_commit` (inherited from HRD manifest;
    matches the enclosing run-summary's
    `source_commit`)
  - `trace_convergence_windows_digest` = sha256 over
    canonical-JSON `windows[]` (the verifier recomputes
    this from raw evidence per the strict-schema lesson
    family; cached-vs-recomputed disagreement rejects)
  - `trace_convergence_manifest_slug = "dram_stream_stride_mix"`
  - `trace_convergence_manifest_shape_point = "trace_00"`
  - `trace_convergence_manifest_binary_sha256` =
    inherited from HRD manifest (matches the trace's
    `binary_sha256` in the enclosing run summary)
  - `trace_convergence_manifest_source_commit` =
    inherited from HRD manifest (matches the trace's
    `source_commit` in the enclosing run summary)
  - `trace_convergence_manifest_kernel_signature_sha256` =
    inherited from HRD manifest (matches the trace's
    `kernel_signature_sha256` in the enclosing run
    summary)
- **Verifier**:
  `python workloads/simbench_bw/verify_artifacts.py
  --results workloads/simbench_bw/l1/dram_stream_stride_mix/traces`
  exits 0; the
  `_verify_trace_convergence_run_summaries` path is
  exercised in addition to the existing C-AC-H, HRD,
  order-projections, and perturbation-harness surfaces.
  All 12 rejection arms are negative-controlled by
  `workloads/simbench_bw/tests/test_trace_convergence_run_summary_verifier.py`
  (17 tests total: 12 rejection arms + positive control +
  manifest_run_summary_drift + 3 R38 negative controls
  for the binary_sha256/source_commit/kernel_signature
  identity arms).
- **R38 controller-path test counts**:
  `workloads/simbench_bw/tests/test_trace_convergence_run_summary_controller.py`
  carries 7 tests: positive control with
  manifest-inheritance + window-count-underconfigured
  admission, helper-skip path when HRD is missing,
  hrd_layer_unavailable arm, budget-cap arm with
  `trace_volume_exceeded.json` binding,
  trace_convergence_disabled arm with config flag,
  duplicate-window negative control regression-locking
  Codex's R37 review directive #4, and helper-to-
  verifier end-to-end subprocess test.

### (f) Most recent sm_86-side evidence (v3 R15 capture)

Captured during the v3 RLCR loop's Round 15 closure on the RTX 3090 Ti
(NVIDIA driver 595.71.05, CUDA toolkit 13.2, compute capability 8.6).
The post-commit probe regenerates the trace_convergence_confidence
bundle so `manifest_pairing_index.source_commit` rebases to the new
HEAD on every commit; the reviewer's `source_commit` value is the
working-branch HEAD they inspect.

- **Trace directory**:
  `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00/`.
- **Artifact path**:
  `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00/convergence_confidence/convergence_confidence.json`.
- **Strict schema validation**: admits without errors via
  `jsonschema.Draft202012Validator` against
  `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`.
- **Source-commit equality**: 3-way: `git rev-parse HEAD ==
  trace_trace_00.json.trace_convergence_source_commit ==
  convergence_confidence.json.manifest_pairing_index.source_commit`.
- **Bundle structural shape**: `schema_version=1`,
  `manifest_pairing_index` populated, device sm_86
  (CC 8.6, RTX 3090 Ti, 84 SMs), `mrc_quantiles=[0.1, 0.25,
  0.5, 0.75, 0.9]`, `distance_metric="max_abs_mrc_quantile_delta"`,
  `k_consecutive_windows=3`, `distance_threshold=0.02`,
  `trace_window_size_bytes=16777216`, 1 window in `windows[]`.
- **Admission record (typed-unavailable arm)**:
  - `admission.result_kind = "typed_unavailable"`
  - `admission.typed_reason = "hrd_layer_unavailable"`
  - `admission.bundle_underconfigured = true`
  - `admission.requested_layers = ["sector", "line"]`
  - `admission.unavailable_layer = "line"` (the line layer
    has 0 usable admitted bins)
  - `admission.usable_bin_count = 2` (the sector layer has
    2 usable admitted bins per the shared
    `is_usable_admitted_bin` adapter)
- **Run-summary cached fields**:
  `trace_convergence_admitted=false`,
  `trace_convergence_typed_reason="hrd_layer_unavailable"`,
  `trace_convergence_windows_observed=1`,
  `trace_convergence_windows_converged=0`,
  `trace_convergence_final_residual=null`,
  `trace_convergence_mrc_quantile_set=[0.1, 0.25, 0.5,
  0.75, 0.9]`,
  `trace_convergence_windows_digest` is a real 64-char
  hex string that round-trips through `verify_artifacts.py
  --results` (the verifier recomputes from `windows[]`
  and rejects any cached-vs-recomputed disagreement).
- **AC-11 v2 positive test (1 of 2)**: `_SUPPORTED_ARCHES`
  in `tools/adaptive_sampler/run_sampler.py` includes
  `sm_86`.
- **AC-11 v2 positive test (2 of 2)**: live sweep produced
  the windows_digest that round-trips through the
  verifier.
- **AC-11 v2 negative-test arm**: stale cached digest /
  stale cached manifest tuple rejection covered by
  host-only synthetic-fixture tests in
  `tools/adaptive_sampler/test_*.py` and
  `workloads/simbench_bw/tests/test_trace_convergence_run_summary_verifier.py`;
  the live capture itself exercises the
  `hrd_layer_unavailable` typed-unavailable arm via the
  upstream HRD's `bundle_underconfigured=true` plus the
  line layer's 0 usable admitted bins.
- **Cross-arch invariance contract**: per CLAUDE.md
  "Two-Device Local Development Model" the adaptive
  sampler is arch-independent in implementation; the
  empirical `trace_convergence_admitted` flag and
  `typed_reason` value depend on the upstream HRD
  admission state, which IS arch-coupled by L1/L2 sizing
  differences (per AC-6's note). Formal v3
  `Cross-arch-verified` status remains pending a future
  5060 sm_120 v3 session leg replaying the same producer
  against a fresh sm_120-laptop HRD bundle.

---

## Entry: Loop 2 task18 — sm_86 Microbenchmark Suite Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task18-microbench-suite-cross-arch-followup` (sm_86
cross-arch verification for the Loop 2 task18 microbenchmark-
suite-coverage criterion: ten plan-enumerated families
enumerated in `tools/microbench_inventory.py`, scored against a
closed 4-state `Ac14FamilyState` enum
{`live_verified_sm_120`, `structural_only`, `arch_unsupported`,
`not_implemented`} where `not_implemented` causes the validator
to FAIL).

### (b) What is blocked
The sm_86 cross-arch SASS-slice verification of every
`live_verified_sm_120` and `structural_only` slug declared in
`tools/microbench_inventory.py`. Loop 2 sm_120 evidence is
captured on the RTX 5060 laptop (Blackwell); the matching
sm_86 evidence requires running the simbench `Makefile sass:`
target on the 3090 Ti to produce
`<slug>/sass_archive/<slug>.sass` for each `live_verified` /
`structural_only` slug, then re-running the suite's feature
extractors against sm_86 NVBit traces. The validator's RED
not_implemented set is closed across follow-up Loop 2 rounds
(one family per round); each new slug joins this same sm_86
cross-arch follow-up scope.

### (c) Precise operator-facing artifact

- **Inventory module**:
  `tools/microbench_inventory.py` declares the ten plan-
  enumerated families as a typed `Ac14Family` tuple. R44
  replaced the prior covered/pending model with a closed 4-
  state enum (`Ac14FamilyState`); each entry carries a `state`
  field plus a state-specific evidence/rationale field. The
  validator FAILS when any family is `not_implemented`, closing
  the prior deferral loophole.
- **Offline validator**:
  `tools/test_microbench_ac14_coverage.py` regression-locks the
  contract: family count is exactly ten; every family is in
  exactly one state; every `live_verified_sm_120` or
  `structural_only` slug is in
  `workloads/simbench_bw/roster.toml` OR
  `workloads/portable_lcd/roster.toml`; every covered simbench
  slug has a shape file at
  `workloads/simbench_bw/shapes/<slug>.toml` with
  `timing_shapes`, `trace_shapes`, and `correctness_shapes`
  blocks; six negative controls cover each rule. The
  `test_no_family_is_not_implemented` arm FAILS until every
  family closes.

- **`live_verified_sm_120` families at R45**:
  - `stride_misalignment_sweep` → `dram_stream_stride_mix`
    (existing simbench trace under
    `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`).
  - `cache_modifier_variants` → `dram_stream_stride_mix` (via
    the perturbation-harness clause's three cache-modifier
    families: `cache_modifier_ca`, `cache_modifier_cg`,
    `cache_modifier_cs`; R45 added the `.cs` (cache-streaming)
    variant to close this family). Live evidence in the
    perturbation_run_ledger.json artifact at the reviewed HEAD.
  - `bandwidth_saturation_sweep` → `dram_stream_stride_mix`
    (saturation regime exposed by the timing-shape axes
    `warps=1024, accesses=4096, stride_classes=4`).

- **`structural_only` families at R50**: NONE. R50 captured
  live sm_120 evidence for the 7 slugs that were
  `structural_only` at R49 reviewed HEAD; every family is
  now `live_verified_sm_120` (see live evidence list below).

- **`not_implemented` families at R50**: NONE. After R48
  every microbench-suite-coverage family is at
  `live_verified_sm_120`; after R50 every family also has
  live sm_120 trace + SASS evidence. The validator is GREEN.

### (d) sm_120-side prerequisites completed in Loop 2

Across R43, R44, and R45:

- **R43**: `tools/microbench_inventory.py` and
  `tools/test_microbench_ac14_coverage.py` initial landing
  with the early covered/pending model (later rejected by the
  R43 review).
- **R44**: state-model rework — closed 4-state enum
  `Ac14FamilyState`; validator FAILS on any `not_implemented`
  family. Two new structural slugs landed
  (`working_set_capacity_sweep`,
  `warm_cold_pair_relaunch`) with full simbench-infrastructure
  integration: `splits.toml` (40 → 42 calibration), schema
  count consts, `verify_coverage.py::BENCHMARK_ROW_COUNT`
  (72 → 74), `platform_matrix.toml`. Validator extended to
  query both rosters (simbench + portable_lcd). C-AC-D plan-
  token regression resolved (re-prosed both files).
- **R45**: `cache_modifier_variants` family closed by
  extending the perturbation-harness clause to compile the
  `.cs` (cache-streaming) variant. `tools/perturbation_harness/build.py`
  `generate_cache_modifier_source` admits `cs`;
  `_build_flags_for_family` / `_binary_name` /
  `_shape_args_for` route the new variant; the main loop
  family iteration adds `("cache_modifier_cs", {"modifier":
  "cs"})`.
  `tools/perturbation_harness/ledger.py` `PERTURBATION_FAMILIES`
  / `CACHE_MODIFIER_FAMILIES` / `SASS_EXPECTED_MATCH_COUNT_BY_FAMILY`
  extended. Schema enums admit `cache_modifier_cs`. Family
  count for the perturbation harness grew from 9 to 10
  records (3 cache-modifier + 7 working-set ladder).
- **R46**: `l2_prefetch_hint_variants` family closed
  structurally by adding the `l2_prefetch_hint_sweep`
  microbench slug (CUDA kernel using `prefetch.global.L2`
  PTX inline asm + Makefile + main.cu + shape file). Roster
  + splits.toml + schema + verify_coverage.py +
  platform_matrix.toml updated to admit the new row (74 →
  75 rows; calibration 42 → 43). The R45 producer-side
  schema/test 9→10 gap also closed: `perturbations.minItems`
  now 10; `_default_loop_records()` builds 10 records;
  negative-control test rejects 9-record bundle missing
  cache_modifier_cs.
- **R47**: `l2_persisting_window_sweep` family closed
  structurally by adding the `l2_persisting_window_sweep`
  microbench slug (CUDA kernel + host driver invoking
  `cudaStreamSetAttribute` /
  `cudaStreamAttributeAccessPolicyWindow` over the input
  allocation; `persisting_window_bytes` shape axis sweeps
  the requested window size). The host driver emits a
  typed-unavailable diagnostic when
  `cudaDevAttrMaxAccessPolicyWindowSize` reports the
  attribute is unsupported. Roster + splits.toml + schema
  + verify_coverage.py + platform_matrix.toml updated for
  the new row (75 → 76 rows; calibration 43 → 44). R46
  blocking C-AC-D plan-token regression closed (re-prosed
  the two `AC-<digit>` literals in
  `l2_prefetch_hint_sweep.cu`); R46 reference-family
  mismatch closed for 4 microbench-suite-coverage slugs
  (`mma_microbench.FAMILY_COVERS_SLUGS` widened;
  `build_reference` adds explicit shape branches matching
  each host driver's printed JSON output); R46 shape /
  launch metadata mismatch on `l2_prefetch_hint_sweep`
  fixed (`launch_config` now matches the actual 1×32
  kernel launch). New focused dry-run regression
  `test_microbench_correctness_dry_run.py` parametrizes
  every structural / live-verified slug through
  `--mode correctness --dry-run-cpu --component-strict`.
- **R48**: `carveout_sensitivity_probe` family closed
  structurally by adding the `carveout_sensitivity_probe`
  microbench slug (CUDA kernel + host driver invoking
  `cudaFuncSetAttribute` with
  `cudaFuncAttributePreferredSharedMemoryCarveout` ahead of
  launch; `carveout_percent` shape axis sweeps the
  requested L1/shared partition split). After R48 every
  microbench-suite-coverage family is at `structural_only`
  or `live_verified_sm_120`; the validator's
  `test_no_family_is_not_implemented` arm is GREEN. R47
  blocking persisting-window typed-unavailable gap closed:
  the host driver now emits a structured typed-unavailable
  JSON document on stdout (`result_kind ==
  "typed_unavailable"`, `typed_reason ==
  "arch_feature_unavailable_l2_persisting"`) instead of a
  successful exit with stderr-only diagnostic;
  `run_suite.py`'s correctness path detects the result_kind
  and persists the typed-unavailable record as an
  unsupported typed outcome rather than comparing to the
  reference. New negative-control test
  `test_microbench_correctness_typed_unavailable.py` proves
  both the typed-unavailable persistence path and the
  unaltered admitted-output reference-compare path. R47
  blocking C-AC-D plan-token regression closed (re-prosed
  the `AC-14` literal in
  `test_microbench_correctness_dry_run.py`). Roster +
  splits.toml + schema + verify_coverage.py +
  platform_matrix.toml updated for the new row (76 → 77
  rows; calibration 44 → 45). The `carveout` typed-reason
  carries the same canonical name already enumerated in
  `tools/perturbation_harness/ledger.py::PERTURBATION_RUN_LEDGER_TYPED_REASONS`.
- **R49**: trace-size pre-admission unblocked + carveout
  sweep made real + typed-unavailable extended to trace +
  timing modes. (1) `trace_size_estimator.py::_shape_axis_product`
  no longer multiplies byte-suffixed axes (`*_bytes`) or
  per-launch configuration knobs (`carveout_percent`,
  `prefetch_distance`) into the iteration-count product;
  the static estimator now matches what NVBit actually
  records on the wire instead of conflating buffer-size
  axes with dynamic instruction count.
  `warm_cold_pair_relaunch/trace_00.predicted_bytes`
  updated 1310720 → 40960. `--validate-all` exits 0.
  (2) `carveout_sensitivity_probe` reworked into a real
  sweep: shape file now declares 3 trace shapes (25 / 50 /
  75 percent) + 3 timing shapes; kernel rewritten to read
  from a multi-KiB carveout-bounded shared tile AND the
  global allocation; host driver opts the kernel into the
  requested dynamic-shared budget via
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`.
  (3) typed-unavailable wiring extended from correctness
  to trace and timing: `handle_trace` detects
  `result_kind == "typed_unavailable"` on the baseline
  (uninstrumented) workload run and persists
  `trace_<shape>.json` + manifest without dispatching
  NVBit; `handle_timing` detects it on the first warm-up
  and persists `timing_<shape>.json` + manifest without
  entering the repeat loop.
  `test_microbench_correctness_typed_unavailable.py` grew
  from 2 to 4 tests (added timing + trace negative-control
  arms).
- **R50**: live sm_120 trace + SASS evidence captured for
  all 7 R49-structural-only slugs on the RTX 5060 Laptop
  GPU (driver 595.71.05, CUDA 13.2). For each slug:
  - `make clean all` builds the binary
    (sm_90a + sm_120 -gencode flags; ptxas reports clean
    register usage).
  - The binary was invoked with the trace_00 shape args
    (`_kernel_args_from_shape` derived) under
    `CUDA_INJECTION64_PATH=tools/nvbit_tracer/tracer_tool/tracer_tool.so`,
    producing
    `workloads/simbench_bw/l1/<slug>/traces/dynamic_trace.pb`
    (NVBit dynamic trace).
  - `make sass` produces
    `workloads/simbench_bw/l1/<slug>/sass_archive/<slug>.sass`
    (cuobjdump dump containing sm_120 instruction
    sequences; 1362-4450 lines per slug).
  - `python workloads/simbench_bw/verify_artifacts.py
    --results workloads/simbench_bw/l1/<slug>/traces`
    exits `ok` for all 7 slugs.
  Inventory transitions: 7 families flip from
  `structural_only` to `live_verified_sm_120`
  (`pointer_chase`, `working_set_sweep`,
  `warm_cold_pair`, `l2_prefetch_hint_variants`,
  `l2_persisting_window_sweep`,
  `resident_warp_pressure_sweep`,
  `carveout_sensitivity_probe`). After R50 every family
  in the AC-14 inventory is `live_verified_sm_120`;
  no families remain in any other state.

  The `traces/` and `sass_archive/` directories under
  `l1/<slug>/` are gitignored (per the repo's
  `.gitignore` `traces/` rule), so the live evidence
  lives in the local working tree at the round-closing
  reviewed HEAD per the abstract-HEAD pattern. The
  inventory's per-slug rationale records the live
  capture; future verification rebases on the next
  capture session.
- The simbench harness's feature extractors already produce
  per-trace sidecars for the `live_verified_sm_120` slugs (HRD
  profile + sector stream + divergence proxies + sharing
  locality + order projections + perturbation run ledger +
  trace convergence confidence). Per-run sidecar consumption
  by the suite is implicit through this trace-time emission.

### (e) Most recent sm_120-side evidence

Captured during R50 closure on the RTX 5060 Laptop GPU
(NVIDIA GeForce RTX 5060 Laptop GPU, CC 12.0, driver
595.71.05, CUDA 13.2).

- **Live sm_120 evidence captured for all 7 R49-structural-
  only slugs**: each slug's `traces/dynamic_trace.pb` was
  produced by direct NVBit injection on the RTX 5060
  Laptop GPU; each slug's `sass_archive/<slug>.sass` was
  produced by `cuobjdump --dump-sass` on the per-arch
  build artifact:
  - `l2_thrash_pointer_chase`: 189 B trace, 1362-line SASS.
  - `working_set_capacity_sweep`: 197 B trace, 2178-line SASS.
  - `warm_cold_pair_relaunch`: 368 B trace (3 traced
    kernel launches: cold + eviction + warm — confirmed
    via `traces/stats.csv` which lists kernel-1.trace,
    kernel-2.trace, kernel-3.trace under their distinct
    mangled names `warm_cold_pair_kernel`,
    `warm_cold_pair_eviction_kernel`,
    `warm_cold_pair_kernel`; the HRD-profiler criterion
    pairs the cold and warm `warm_cold_pair_kernel`
    launches), 2140-line SASS.
  - `l2_prefetch_hint_sweep`: 187 B trace, 1890-line SASS.
  - `l2_persisting_window_sweep`: 197 B trace, 2178-line
    SASS (the device advertised
    `cudaDevAttrMaxAccessPolicyWindowSize > 0` and the
    persistence path ran natively, not the typed-
    unavailable path).
  - `register_spill_local_mem`: 191 B trace, 4450-line
    SASS (80 registers used per ptxas, exposing the
    register-pressure / local-memory spill envelope).
  - `carveout_sensitivity_probe`: 201 B trace (the device
    advertised `cudaDevAttrMaxSharedMemoryPerBlockOptin > 0`
    and the carveout path ran natively), 2738-line SASS.
  Each tree admits via `python
  workloads/simbench_bw/verify_artifacts.py --results
  workloads/simbench_bw/l1/<slug>/traces` (`ok`).
- **Validator state**:
  `python -m pytest tools/test_microbench_ac14_coverage.py -v`
  reports `14 passed` after R50 with 0 not_implemented and
  0 structural_only; the
  `test_no_family_is_not_implemented` arm continues to
  pass. AC-14 microbench-suite-coverage validator is GREEN
  with all 10 plan-enumerated families at
  `live_verified_sm_120`.
- **Trace-size pre-admission state**:
  `python workloads/simbench_bw/trace_size_estimator.py
  --validate-all` exits 0 (was exit 1 with 5
  `trace_prediction_stale` errors at R48 reviewed HEAD). R49
  updated `_shape_axis_product` to exclude byte-suffixed
  axis names (`*_bytes`) and per-launch configuration knobs
  (`carveout_percent`, `prefetch_distance`) from the
  iteration-count multiplier; these describe buffer sizes /
  configuration knobs, not work amounts. `predicted_bytes`
  on `warm_cold_pair_relaunch` updated 1310720 → 40960 to
  match the new estimator output. The `handle_trace` pre-
  admission no longer rejects any structural-only slug at
  reviewed HEAD.
- **Dry-run correctness coverage**:
  `python -m pytest workloads/simbench_bw/tests/test_microbench_correctness_dry_run.py -v`
  reports `1 passed, 8 subtests passed`. The 8 subtests are
  the deduplicated microbench-suite-coverage slugs in
  `live_verified_sm_120` or `structural_only` state at R49
  (`dram_stream_stride_mix`, `l2_thrash_pointer_chase`,
  `working_set_capacity_sweep`, `warm_cold_pair_relaunch`,
  `l2_prefetch_hint_sweep`, `l2_persisting_window_sweep`,
  `register_spill_local_mem`, `carveout_sensitivity_probe`).
- **Typed-unavailable harness regression**:
  `python -m pytest workloads/simbench_bw/tests/test_microbench_correctness_typed_unavailable.py -v`
  reports `4 passed` after R49 (was 2 passed at R48). The
  4 tests pin (a) the correctness-mode typed-unavailable
  persistence path, (b) the unaltered correctness admitted-
  output reference-compare path, (c) the timing-mode
  typed-unavailable persistence path (`handle_timing` skips
  the repeat loop when the first warm-up reports
  typed-unavailable; persists `timing_<shape>.json` +
  manifest), and (d) the trace-mode typed-unavailable
  persistence path (`handle_trace` skips the NVBit-injected
  run when the baseline workload reports typed-unavailable;
  persists `trace_<shape>.json` + manifest).
- **`carveout_sensitivity_probe` multi-shape sweep**: R49
  reworked the slug from a single fixed 50-percent point to
  a real sweep: 3 trace shapes (25 / 50 / 75 percent) and
  3 timing shapes matching, plus a 50-percent correctness
  shape kept deterministic. The kernel now reads from BOTH
  a multi-KiB carveout-bounded shared tile and the global
  allocation, so the partition split is observable in L1
  traffic. The host driver opts the kernel into the
  requested dynamic-shared budget via
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`.
- **Perturbation harness live evidence**: the post-commit
  in-process probe re-runs `tools/perturbation_harness/build.py`
  to compile 3 cache-modifier variants + 7 working-set ladder
  records (10 total). The
  `perturbation_run_ledger.json` artifact contains an admitted
  `cache_modifier_cs` record at the reviewed HEAD; the
  `manifest_pairing_index.source_commit` rebases on every
  commit per the abstract-HEAD pattern.
- **Live evidence cross-references**: the existing sm_120
  evidence under
  `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`
  carries forward; `verify_artifacts.py` exits `ok`; the
  convergence sidecar's `manifest_pairing_index` continues to
  inherit from the upstream HRD bundle's identity tuple per
  the abstract-HEAD pattern.
- **Pending live evidence** for `structural_only` slugs:
  NONE on sm_120. R50 captured all 7 R49-pending slugs.
  Live v3 sm_86 cross-arch evidence captured on the RTX
  3090 Ti for the same 7 slugs across 9 R51-contract shape
  points: `l2_thrash_pointer_chase trace_00`,
  `working_set_capacity_sweep trace_00`,
  `warm_cold_pair_relaunch trace_00`,
  `l2_prefetch_hint_sweep trace_00`,
  `l2_persisting_window_sweep trace_00`,
  `register_spill_local_mem trace_00`, and
  `carveout_sensitivity_probe trace_{00,50,75}`. Each
  canonical artifact root at
  `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/`
  admits via `verify_artifacts.py --results <root>` and
  carries the per-shape feature sidecars (sector_stream,
  hrd_profile, divergence_proxies, sharing_locality,
  order_projections); the prerequisite `arch_gate`
  widening landed in the same commit (each slug's
  `roster.toml` row, `Makefile`, and `.cu` source comment
  now admit sm_86). See Section (f) below for the full
  per-shape sm_86 evidence record. The remaining
  `Cross-arch-verified` work is a future sm_120 v3 leg
  replaying the same captures against fresh sm_120-laptop
  bundles on the 5060 laptop per CLAUDE.md "Two-Device
  Local Development Model".
- **R51 path-discipline scaffolding (sm_120 side)**:
  Codex's R50 review flagged that R50's seven trace
  trees were captured via a direct
  `CUDA_INJECTION64_PATH=tracer_tool.so <binary>` invocation
  that bypassed `run_suite.py::handle_trace`; as a result
  the trees contain only raw `dynamic_trace.pb` and
  `kernel-N.trace` files — they do NOT contain the
  per-shape `trace_<shape>.json` run summaries, the
  `measurement_manifest.json` per-shape manifest, or any
  of the AC-5..AC-9 feature sidecars
  (`sector_stream/`, `hrd_profile/`, `divergence_proxies/`,
  `sharing_locality/`, `order_projections/`,
  `perturbation_run_ledger/`, `convergence_confidence/`).
  R51 lands the orchestrator-side path discipline so a
  future canonical re-capture produces the full sidecar
  set:
    1. `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`:
       the `USER_DEFINED_FOLDERS == 1` branch now updates
       every output path (`traces_path`, `extrainfo_path`,
       `cubin_path`, `sass_path`, `register_usage_path`,
       `threadblock_trace_path`,
       `threadblock_register_values_path`,
       `traces_location`, `stats_location`) BEFORE
       `create_folder(traces_path.c_str())`. The pre-R51
       branch only redirected `traces_location` and
       `stats_location` and only after the default
       `./traces/` folder had already been created, which
       left `dynamic_trace.pb`, the extra_info subtree,
       and the threadblocks subtree at the hardcoded
       path even when `TRACES_FOLDER` was set.
    2. `workloads/simbench_bw/run_suite.py::handle_trace`:
       sets `env["USER_DEFINED_FOLDERS"] = "1"` (the gate
       that unlocks the tracer's `TRACES_FOLDER` lookup)
       and pre-cleans the per-shape `traces_folder` with
       `shutil.rmtree` before the NVBit run, so a stale
       `dynamic_trace.pb` from a previous capture cannot
       silently pass the post-capture existence check.
    3. `workloads/simbench_bw/verify_artifacts.py`: a new
       `_verify_canonical_trace_pb_admission` helper flags
       any canonical-shape `trace_<shape>/dynamic_trace.pb`
       lacking a binding `trace_<shape>.json` summary or
       `measurement_manifest.json`. The pre-R51 verifier's
       `*.trace*` glob did not match `dynamic_trace.pb` and
       the per-shape summary verifier returned a vacuous
       pass when no summary existed; the combination
       silently admitted bypass-orchestrator captures.
       The new helper is scoped to canonical
       `trace_<shape>/` dirs only so legacy `traces/`
       layouts (the R50 captures referenced above) are
       not flagged.
    4. Two new fail-closed test files:
       `tests/test_run_suite_handle_trace_user_defined_folders.py`
       (4 AST-based tests asserting the env discipline +
       pre-clean) and
       `tests/test_verify_artifacts_canonical_trace_pb_admission.py`
       (5 end-to-end verifier tests covering raw-only,
       missing manifest, unparseable summary, mixed
       canonical+broken, and the canonical-pass path).
- **R51 toolchain blocker (sm_120 side)**:
  The tracer source patch landed clean but the rebuild
  is blocked: this host's `g++ 16.1.1` libstdc++ headers
  use C++23 explicit object parameter syntax
  (`operator()(this _Self&& __self, ...)`) that nvcc
  13.2's cudafe frontend cannot parse, so any nvcc
  invocation against `tracer_tool.cu` fails at the STL
  parse stage — independent of my patch. The errors
  surface in `<functional>`, `<memory>`, and
  protobuf/Abseil headers (~100 errors per file).
  `--allow-unsupported-compiler` does not help (the
  headers are simply unparseable, not just from an
  unsupported compiler version), and using `clang++` as
  ccbin keeps the same libstdc++ headers. The previous
  `tracer_tool.so` was a binary built before this host
  upgraded to `g++ 16.1.1`, so the R50 capture worked
  with that older binary; `make clean` discarded it.
  Resolution requires either `sudo pacman -S gcc14`
  + `nvcc -ccbin=g++-14`, or `sudo pacman -S libc++` +
  clang+libc++ Makefile change. Both require sudo and
  were deferred from R51. Once the rebuild lands, the
  next round captures the 9 shape points (7 R49-pending
  slugs at `trace_00` + carveout extras at `trace_50`,
  `trace_75`) via `python workloads/simbench_bw/run_suite.py
  --mode trace --kernel <slug> --shape-point <shape>`.
- **R51 deferred mainline work**: the canonical capture
  (success criteria 3 and 4 of round-51-contract.md) and
  the canonical-path inventory refresh (criterion 6,
  bullet 1) are deferred to a follow-up round once
  toolchain is unblocked. The path-discipline
  scaffolding (criteria 1, 2, 5, plus the warm/cold
  prose fix in criterion 6 bullet 2) and the no-
  regressions gate (criterion 7 partial: tests +
  legacy-tree admission) all closed in R51.
- **R52 canonical capture (sm_120 side, 8 of 9 shape
  points)**: gcc14 install (`sudo pacman -S gcc14`)
  unblocked the toolchain. The Makefile gained an
  env-driven `NVCC_CCBIN?=$(CXX)` selector so
  `NVCC_CCBIN=g++-14 make` produces a clean
  `tracer_tool.so` against gcc14's libstdc++. R52 then
  ran `python workloads/simbench_bw/run_suite.py --mode
  trace --kernel <slug> --shape-point <shape>` for the
  9 R51-contract shape points; 8 succeeded canonically
  with full sidecar emission:
    - `l2_thrash_pointer_chase trace_00`
    - `working_set_capacity_sweep trace_00`
    - `l2_prefetch_hint_sweep trace_00`
    - `l2_persisting_window_sweep trace_00`
    - `register_spill_local_mem trace_00`
    - `carveout_sensitivity_probe trace_00`
    - `carveout_sensitivity_probe trace_50`
    - `carveout_sensitivity_probe trace_75`
  Each canonical artifact root at
  `workloads/simbench_bw/results/<slug>/blackwell_consumer_unknown/sm_120/`
  contains the per-shape `trace_<shape>.json` run
  summary, the per-shape `measurement_manifest.json`,
  the per-shape `trace_<shape>/dynamic_trace.pb` raw
  NVBit trace (187-201 bytes, matching R50 raw sizes),
  the typed-exclusion sidecars under
  `trace_<shape>/extra_info/` (allocations.json,
  unsupported_memory_path.json,
  unmapped_memory_range.json,
  cross_allocation_lane_span.json,
  enhanced_execution_info.json), and the AC-5..AC-9
  feature sidecars: `sector_stream/sector_stream.jsonl`,
  `hrd_profile/hrd_profile.json`,
  `order_projections/order_projections.json`,
  `perturbation_run_ledger/perturbation_run_ledger.json`,
  `convergence_confidence/convergence_confidence.json`,
  plus the per-CTA threadblocks subtree. `python
  workloads/simbench_bw/verify_artifacts.py --results
  <root>` exits `ok` for each of the 6 artifact roots
  (carveout's 3 shapes share one root). The R50 raw
  trees under `workloads/simbench_bw/l1/<slug>/traces/`
  remain in place as historical evidence.
- **R52 canonical capture queued for warm_cold_pair_relaunch**:
  RESOLVED in R53. The `warm_cold_pair_relaunch trace_00`
  capture's pre-R53 hang was caused by an O(N^2) LRU
  reuse-distance loop in `tools/hrd_profiler/observation
  _builder.py` (and a duplicate in
  `tools/order_projections/projector.py::
  _hit_rate_proxy_under_order`) — `lru_list.index/remove
  /append` per access. R53 replaced both with a
  Fenwick-tree-backed `ReuseDistanceTracker`
  (O(log N) per access, O(N log N) per launch).
  warm_cold canonical capture now lands in ~110s
  including all 7 sidecars; verify_artifacts.py
  exits `ok`.
- **R52 absolute-path resolver landed**: the R51
  review's blocking issue #2 (absolute-`TRACES_FOLDER`
  bug at `tracer_tool.cu:858`, `:1004`, `:3505`) is
  closed. New header
  `tools/nvbit_tracer/tracer_tool/path_resolver.h`
  declares `inline std::string
  resolve_output_path(const std::string &cwd, const
  std::string &p)` that returns `p` verbatim when
  absolute, else `cwd / p` via
  `std::filesystem::path::operator/`. The three
  flagged sites in `tracer_tool.cu` now route through
  the helper. Host-only unit test
  `tools/nvbit_tracer/tracer_tool/tests/path_resolver_test.cc`
  (8 assertions) and Python source-level lint
  `workloads/simbench_bw/tests/test_tracer_tool_resolve_output_path_invariant.py`
  (7 assertions) regression-lock the invariants.
- **R52 inventory canonical-path refresh**: the 6
  inventory rationales for slugs whose canonical
  capture succeeded
  (`l2_thrash_pointer_chase`,
  `working_set_capacity_sweep`,
  `l2_prefetch_hint_sweep`,
  `l2_persisting_window_sweep`,
  `register_spill_local_mem`,
  `carveout_sensitivity_probe`) point to
  `workloads/simbench_bw/results/<slug>/blackwell_consumer_unknown/sm_120/`
  and admit via verify_artifacts.py.
- **R53 closure of all 9 R51-contract canonical
  shape points + 5 AC-5..AC-9 feature sidecars per
  shape**:
  - **HRD complexity fix**: replaced the
    O(N) `lru_list.index/remove/append` per access
    in `tools/hrd_profiler/observation_builder.py`
    and in `tools/order_projections/projector.py::
    _hit_rate_proxy_under_order` with a Fenwick-
    tree-backed `tools/hrd_profiler/
    reuse_distance_tracker.py::ReuseDistanceTracker`
    (O(log N) per access, total O(N log N) per
    launch). Algorithm chosen per Codex's R53
    design review (BIT over a per-launch monotonic
    timestamp axis, 4 independent trackers per
    layer matching the existing
    sector/line/allocation/region partitioning;
    0-indexed-from-top semantics preserved to
    match the existing observation-builder tests).
    16 host-only unit tests pin the algorithm
    semantics and lock a 100k+100k re-touch
    benchmark below 5s (the legacy O(N^2) would
    take 10-30s on the same workload).
  - **Two missing canonical sidecars wired**:
    new `_emit_divergence_proxies_for_run_summary`
    and `_emit_sharing_locality_for_run_summary`
    helpers in `workloads/simbench_bw/run_suite.py`
    (placed between the HRD bundle emission and
    `_emit_order_projections_for_run_summary`).
    Each calls the matching producer
    (`tools.divergence_proxies.bundle_from_trace.
    consume_trace_dir` and
    `tools.sharing_locality.classifier.
    compute_sharing_locality`), strict-validates
    the bundle against the committed JSON schema,
    writes the JSON to
    `trace_<shape>/<sidecar>/<sidecar>.json`, and
    caches per-bundle metadata + source_commit in
    the run summary. Producer failure → typed
    `*_skip_reason`. Per BL-20260505 (tools-subdir-
    module-name-collision), the divergence_proxies
    importer uses `importlib.util.
    spec_from_file_location` to bypass the
    sys.modules cache that would otherwise return
    `tools.hrd_profiler.bundle_from_trace`'s
    `consume_trace_dir`.
  - **Verifier sidecar-completeness check**:
    `_verify_canonical_trace_pb_admission` in
    `workloads/simbench_bw/verify_artifacts.py`
    extended so a canonical
    `trace_<shape>/dynamic_trace.pb` whose run
    summary admits MUST also have all 5 sidecars
    (sector_stream, hrd_profile,
    divergence_proxies, sharing_locality,
    order_projections). Typed-unavailable
    bundles are admissible; typed-skip reasons in
    the run summary admit sidecar absence. 4 new
    fail-closed tests in
    `workloads/simbench_bw/tests/
    test_verify_artifacts_canonical_trace_pb_admission.py`
    (missing divergence_proxies, missing
    sharing_locality, typed-skip-reason admits,
    full-canonical-tree admits).
  - **9 R51-contract shape points captured
    canonically** under
    `workloads/simbench_bw/results/<slug>/
    blackwell_consumer_unknown/sm_120/` for:
    `l2_thrash_pointer_chase trace_00`,
    `working_set_capacity_sweep trace_00`,
    `warm_cold_pair_relaunch trace_00`,
    `l2_prefetch_hint_sweep trace_00`,
    `l2_persisting_window_sweep trace_00`,
    `register_spill_local_mem trace_00`,
    `carveout_sensitivity_probe trace_00`,
    `carveout_sensitivity_probe trace_50`,
    `carveout_sensitivity_probe trace_75`. Each
    canonical artifact root admits via
    `verify_artifacts.py` (exit `ok`) and contains
    the 5 AC-5..AC-9 feature sidecars + the
    perturbation_run_ledger and
    convergence_confidence bundles + per-CTA
    threadblocks subtree.
  - **gcc14 rebuild path documented in
    `tools/nvbit_tracer/README.md`** (Building
    section / `Host compiler selection
    (NVCC_CCBIN)`). Closes the R52 contract gap
    where the env-driven selection landed but
    was not visible to the next operator.
  - **Inventory rationales refreshed**: 7 R49-
    pending slug rationales in
    `tools/microbench_inventory.py` cite the
    canonical artifact root + all 7 sidecars
    (sector_stream, hrd_profile,
    divergence_proxies, sharing_locality,
    order_projections, perturbation_run_ledger,
    convergence_confidence).

### (f) Most recent sm_86-side evidence (v3 R17 capture)

Captured during the v3 RLCR loop's Round 17 closure on the RTX 3090 Ti
(NVIDIA driver 595.71.05, CUDA toolkit 13.2, NVBit 1.8, native
`g++-14` 14.3.1, compute capability 8.6, GPU UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`). The post-arch_gate-
widening commit landed first (per BL-20260508 ordering); the 9
captures then ran POST-commit so each on-disk artifact's
`source_commit` field pins to the new HEAD.

- **Prerequisite arch_gate widening**: each of the 7 R49-pending
  microbench slugs had its `arch_gate` widened from
  `["sm_90a", "sm_120"]` to `["sm_86", "sm_90a", "sm_120"]` in
  three mirror surfaces per the multi-surface-lockstep discipline
  (`BL-20260507`):
  - `workloads/simbench_bw/roster.toml` (the slug's row's
    `arch_gate` array).
  - `workloads/simbench_bw/l1/<slug>/Makefile` (the row's
    `ARCH_GATE` token list).
  - `workloads/simbench_bw/l1/<slug>/<slug>.cu` (the row's
    `// arch_gate:` source comment).
- **Pre-capture build**: each binary built clean with
  `NVCC_CCBIN=g++-14 make -C workloads/simbench_bw/l1/<slug>`;
  `cuobjdump --list-elf` confirmed sm_86 cubin presence in each
  `*_run` binary alongside the existing sm_90a + sm_120 cubins.
- **9 R51-contract canonical sm_86 captures** under
  `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/` for:
  `l2_thrash_pointer_chase trace_00`,
  `working_set_capacity_sweep trace_00`,
  `warm_cold_pair_relaunch trace_00`,
  `l2_prefetch_hint_sweep trace_00`,
  `l2_persisting_window_sweep trace_00`,
  `register_spill_local_mem trace_00`, and
  `carveout_sensitivity_probe trace_{00,50,75}` (3 shapes share
  one root). Each canonical artifact root admits via
  `python workloads/simbench_bw/verify_artifacts.py --results
  workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/` (exit
  `ok`) and contains the per-shape `trace_<shape>.json` run
  summary, `measurement_manifest.json`, the per-shape
  `trace_<shape>/dynamic_trace.pb` raw NVBit trace, the typed-
  exclusion sidecars under `trace_<shape>/extra_info/`, and the
  AC-5..AC-9 feature sidecars (`sector_stream/`, `hrd_profile/`,
  `divergence_proxies/`, `sharing_locality/`,
  `order_projections/`).
- **Per-shape source-commit equality** at the post-commit HEAD:
  for each of the 9 captures,
  `git rev-parse HEAD == trace_<shape>.json.source_commit ==
  bundle.manifest_pairing_index.source_commit` for every per-
  shape feature sidecar present.
- **AC-14 coverage validator GREEN** at the post-capture HEAD:
  `python -m pytest tools/test_microbench_ac14_coverage.py -v`
  reports 14 passed (the
  `test_no_family_is_not_implemented` arm continues to admit;
  `Ac14FamilyState` enum stays closed at exactly 10 families).
- **Inventory rationales refreshed (Loop 3 R17 supplement)**: 7
  R49-pending slug rationales in
  `tools/microbench_inventory.py` now cite the v3 sm_86
  canonical artifact root in addition to the carried R52/R53
  sm_120 evidence references; the per-rationale prose names the
  RTX 3090 Ti host, the CUDA + NVBit + g++-14 toolchain, the
  `arch_gate` widening prerequisite, and the typed-fail-closed
  arms that fire under L1/L2-sizing-induced HRD
  bundle_underconfigured states.
- **Cross-arch invariance contract**: the producer / schema /
  verifier surfaces are arch-independent in implementation;
  per-arch differences land in the empirical
  bundle_underconfigured arm, the `unavailable_layer` /
  `usable_bin_count` distribution, and the typed-unavailable
  arm fire frequency. Formal v3 `Cross-arch-verified` status
  for AC-14 remains pending a future 5060 sm_120 v3 session leg
  replaying the 9 captures against fresh sm_120-laptop bundles.

---

## Update Discipline

Every Loop 2 round that touches an sm_86-blocked surface (or
discovers a new one) updates this ledger before the round summary is
written. The C-AC-A negative test fails the goal-tracker hygiene
check at round end if a new sm_86 issue is discovered but not added
here, or if any runbook path referenced above does not resolve at
working-branch HEAD.

The ledger is intentionally narrow: it tracks **only** sm_86-blocked
work that is deferred to a future 3090 Ti session. Active Loop 2
work on the laptop (Track B, Track C, Track D, Track E) is tracked
in the session goal tracker under `.humanize/rlcr/<session-id>/`
rather than here.

## See Also

- `CLAUDE.md` "Two-Device Local Development Model" — the
  authoritative source for `Verified-live-<arch>` /
  `Cross-arch-verified` semantics and the four explicit evidence
  statuses every round summary must use.
- `tools/local_labels/TASK7_RUNBOOK.md` — the operator-facing
  procedure for the `task7` entry above.
- `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` — the operator-facing
  procedure for the `task11b-task11d-cross-arch-followup` entry
  above.
- `.humanize/bitlesson.md` — `BL-20260504-runbook-self-reference`
  governs the no-pinned-source-of-truth-SHA pattern this ledger and
  both runbooks follow.

## Entry: Loop 2 task19 — sm_86 Combined Dual-Workload Pairing Index Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task19-combined-pairing-index-cross-arch-followup`
(sm_86 cross-arch verification for the combined dual-workload
manifest-pairing index helper at
`workloads/simbench_bw/dual_workload_pairing_index.py`).

The Loop 2 sm_120 closure of AC-15 across R56-R59 was
re-classified at handoff time as `Offline-verified`, NOT
`Verified-live-sm_120`, because the 26 focused tests that
landed are entirely host-only: in-process `tempfile`
fixtures, schema validation against synthetic manifests,
content-derived `kernel_signature_sha256` against
synthetic binary content, missing-roster fail-closed
arms, and a `LiveRosterSmokeTest` that admits typed-
exception paths (`MissingActiveBinaryError`,
`MalformedManifestError`) as success. None of the tests
required a real GPU. Per CLAUDE.md's tightened evidence-
status discipline, `Verified-live-*` is reserved for
real GPU execution.

### (b) What is blocked
The producer-side live evidence of the combined helper
on EITHER architecture. Specifically:

- **A real run that produces an admissible 5-tuple
  identity** for at least one slug from each workload
  registry: a simbench_bw run summary at
  `workloads/simbench_bw/results/<slug>/<sku>/<arch>/trace_<shape>.json`
  carrying the full 5-tuple `(slug, shape_point,
  binary_sha256, source_commit, kernel_signature_sha256)`,
  AND a portable_lcd active row whose
  `<slug>/<slug>_run` binary actually exists on disk
  with `binary_sha256` matching the manifest's declared
  digest.

- The `LiveRosterSmokeTest` admits typed-exception
  paths when the committed portable_lcd binaries are
  not present. Promotion to `Verified-live-sm_*`
  requires the smoke test to admit on the *positive*
  path: `MissingActiveBinaryError` does NOT fire, the
  combined index is non-empty, and the cached fields
  round-trip cleanly.

- A live sm_86 sweep would also independently confirm
  the binary-content-derived `kernel_signature_sha256`
  is invariant across the two arches' build artifacts
  for the same source. The R58 host-only stability
  test (copying a synthetic tree) is necessary but not
  sufficient.

### (c) Precise operator-facing artifact

- **Helper module**:
  `workloads/simbench_bw/dual_workload_pairing_index.py`
  exposes `WorkloadIdentity` plus three readers
  (`collect_simbench_identities`,
  `collect_portable_lcd_identities`,
  `build_combined_pairing_index`) and 8 typed-diagnostic
  classes (`DualWorkloadKeyCollision`,
  `MalformedManifestError`,
  `SchemaInvalidManifestError`,
  `MissingActiveManifestError`,
  `MissingActiveBinaryError`,
  `SlugNotInRosterError`,
  `MissingRosterError`,
  `BinaryDigestMismatchError`). The `__all__` block
  enumerates every export.

- **Host-only test surface**:
  `workloads/simbench_bw/tests/test_dual_workload_pairing_index.py`
  has 26 tests across 5 classes:
  `CollectSimbenchIdentitiesTests` (7),
  `CollectPortableLcdIdentitiesTests` (11),
  `CombinedPairingIndexTests` (6),
  `WorkloadIdentityKeyShapeTest` (1),
  `LiveRosterSmokeTest` (1).

- **Operator command for sm_86 producer-side live
  evidence**: build the active portable_lcd binaries
  (`make -C workloads/portable_lcd/<active-slug>`),
  capture a simbench timing run that emits a
  `trace_<shape>.json` with the full 5-tuple identity,
  then run the combined builder against the live
  rosters + the live capture tree:

  ```python
  import subprocess
  from pathlib import Path

  from workloads.simbench_bw.dual_workload_pairing_index import (
      build_combined_pairing_index,
  )

  index = build_combined_pairing_index(
      simbench_results_root=Path("workloads/simbench_bw/results"),
      portable_lcd_root=Path("workloads/portable_lcd"),
      repo_root=Path("."),
      simbench_roster_path=Path("workloads/simbench_bw/roster.toml"),
      portable_lcd_roster_path=Path("workloads/portable_lcd/roster.toml"),
  )
  # Live promotion requires BOTH workloads to contribute.
  # `len(index) > 0` alone admits a single-registry result;
  # genuine cross-workload live evidence must show both
  # simbench_bw and portable_lcd identities in the index.
  assert index, "combined index empty"
  registries = {v.workload_registry for v in index.values()}
  assert {"simbench_bw", "portable_lcd"} <= registries, (
      f"index missing a workload: registries={registries}"
  )
  # source_commit MUST equal the receiving session's
  # `git rev-parse HEAD`, not the helper's
  # ``"0" * 64`` zero-fill fallback. The fallback fires
  # silently when `_resolve_source_commit` cannot read
  # `.git/HEAD` (e.g., a fetched clone with packed refs;
  # see queued side issue at the bottom of this ledger
  # entry). The new loop's Round 0 should also verify
  # the helper handles packed refs correctly before
  # claiming AC-15 promotion.
  head = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True,
  ).strip()
  for key in index:
      slug, shape_point, binary_sha256, source_commit, kernel_sig = key
      assert source_commit == head, (
          f"source_commit drift: key={key} expected={head}"
      )
      assert source_commit != "0" * 64, (
          f"source_commit is the zero-fill fallback: {key}"
      )
  ```

  The round summary must record the device/driver/CUDA/
  NVBit build/clocks AND confirm the
  `LiveRosterSmokeTest` admits without firing
  typed-exception paths AND name the slugs from each
  workload that contributed to the index.

  **Latent foot-gun (queued side issue)**:
  `_resolve_source_commit` in
  `workloads/simbench_bw/dual_workload_pairing_index.py`
  manually parses `.git/HEAD` and follows
  `ref: refs/heads/...` to a loose ref file; on `OSError`
  it returns `"0" * 64`. A fetched clone whose refs are
  packed into `.git/packed-refs` will hit the OSError
  path and silently produce zero-filled identities.
  Before claiming AC-15 promotion, either (a) ensure the
  receiving host's repo has a loose ref at
  `.git/refs/heads/<branch>` (a single commit on the
  receiving host materializes one), or (b) extend the
  helper to fall back to `git rev-parse HEAD` /
  `packed-refs`. Either path should land as a separate
  round before promotion.

### (d) sm_120-side prerequisites completed in Loop 2

R56-R59 (across HEADs `32fac6b`, `f447bd1`, `e12f31a`,
`98e2f1a`) landed:

- **R56**: initial scaffolding — `WorkloadIdentity`
  dataclass, three readers, `DualWorkloadKeyCollision`,
  `MalformedManifestError`, 12 host-only tests.
- **R57**: roster-backed + schema-validating rework —
  added `SchemaInvalidManifestError`,
  `MissingActiveManifestError`,
  `MissingActiveBinaryError`,
  `SlugNotInRosterError`; readers now consume
  `workloads/simbench_bw/roster.toml` and
  `workloads/portable_lcd/roster.toml` via `tomllib`;
  portable_lcd manifests validated through
  `validate.py::validate_manifest_files`. 19 tests.
- **R58**: content-derived
  `kernel_signature_sha256` (sha256 of
  `slug | actual_binary_sha256` where the digest comes
  from `binary_path.read_bytes()`); cross-check raises
  `BinaryDigestMismatchError`; new `MissingRosterError`
  raises when the roster file is absent or unparseable;
  strengthened tree-copy stability test asserts both
  `kernel_equal` AND `keys_equal`. 23 tests.
- **R59**: roster-first validation
  (`_load_simbench_roster` runs BEFORE the
  `simbench_results_root.is_dir()` short-circuit);
  active-row slug consistency
  (`manifest["slug"] == slug` required after schema
  validation; mismatch raises
  `MalformedManifestError(slug_mismatch)`);
  `MissingRosterError` and `BinaryDigestMismatchError`
  added to `__all__`. 26 tests.

### (e) Most recent sm_120-side evidence

Host-only test surface at R60 implementation parent
`87e4f05` (handoff documentation commit `de3ea25` is a
descendant carrying only docs/plan/ledger updates with
no behavioral change):

- `python -m pytest workloads/simbench_bw/tests/test_dual_workload_pairing_index.py -q`
  → 26 passed.
- Direct probe (Codex's R58/R59 review style):
  - Tree-copy stability:
    `keys_equal: True`, `kernel_equal: True`.
  - Missing simbench roster + missing results root:
    `MissingRosterError: file_not_found`.
  - Manifest body slug mismatch:
    `MalformedManifestError: slug_mismatch:
    roster=roster_slug manifest=different_slug`.

No live producer-side sweep on sm_120 has been
executed; the receiving sm_86 session can land that
evidence.

## Entry: Loop 2 task20 — sm_86 NVML Covariate Sampler Local-Device Cross-Arch Follow-Up

**Blocked on**: sm_86

### (a) Item name
`loop2-task20-nvml-covariate-sampler-cross-arch-followup`
(sm_86 cross-arch live capture for the local-device NVML
covariate path centered on
`workloads/simbench_bw/cupti_activity_collector.cu`'s NVML
namespace, the cadence constants
`NVML_DEFAULT_POLLING_INTERVAL_NS = 5_000_000` and
`NVML_DEFAULT_CLAIM_SCOPE = "measured_local"` in
`run_suite.py`, the `nvml_metadata_verifier.py` host-only
verifier, and the source-level read-only-NVML guard in
`tests/test_cupti_collector_readonly_nvml_guard.py`).

The Loop 2 sm_120 closure of AC-16 in R60 was
re-classified at handoff time as `Offline-verified`, NOT
`Verified-live-sm_120`, because the 20 focused tests that
landed are entirely host-only: a static source-text scan
of `cupti_activity_collector.cu`, a metadata-pin verifier
walking synthetic sidecars, schema-bound negative
controls. No real NVML capture stream on a live GPU
informed the closure.

### (b) What is blocked
The producer-side live evidence of the local-device
NVML covariate path on EITHER architecture.
Specifically:

- **A real `--mode timing --enable-cupti-activity` run
  on a 3090 Ti** that produces real
  `cupti_activity_<repeat>.json` AND
  `nvml_covariate_<repeat>.json` sidecars sharing the
  host-monotonic clock domain via the CUPTI Activity
  collector's
  `cuptiActivityRegisterTimestampCallback` binding.

- **The `measurement_manifest.json` carries the 4 NVML
  provenance fields populated from the real sidecars**:
  `nvml_polling_interval_ns: 5000000`,
  `nvml_claim_scope: "measured_local"`,
  `nvml_sidecar_paths: ["nvml_covariate_0.json", ...]`,
  `nvml_collector_status: "ok"` (or the appropriate
  union value).

- **`verify_nvml_metadata_pin` admits the live pair**
  cleanly — the cached fields round-trip to the actual
  sidecar contents.

- **At least one repeat reports
  `nvml_collector_status: "ok"`** with a non-empty
  `samples` array (the sm_120 host-only suite admits
  the empty-samples path under `nvml_covariate_unavailable`,
  but live evidence requires a positive `ok` status).

### (c) Precise operator-facing artifact

- **Producer source**:
  `workloads/simbench_bw/cupti_activity_collector.cu` —
  the LD_PRELOAD'd library that emits both CUPTI
  Activity sidecars and NVML covariate sidecars over
  one process lifetime. The NVML namespace dlsym's
  exactly 7 read-only symbols
  (`nvmlInit_v2`, `nvmlShutdown`,
  `nvmlDeviceGetHandleByIndex_v2`,
  `nvmlDeviceGetPowerUsage`,
  `nvmlDeviceGetTemperature`,
  `nvmlDeviceGetCurrentClocksThrottleReasons`,
  `nvmlDeviceGetClockInfo`); the source-level guard at
  `workloads/simbench_bw/tests/test_cupti_collector_readonly_nvml_guard.py`
  rejects any
  `nvmlDeviceSet*` / `nvmlDeviceReset*` regex match
  plus 11 named mutation literals.

- **Cadence + claim-scope constants**:
  `run_suite.py` exposes
  `NVML_DEFAULT_POLLING_INTERVAL_NS = 5_000_000`
  (200 Hz; ≤ schema's 100 Hz floor of 10_000_000 ns)
  and `NVML_DEFAULT_CLAIM_SCOPE = "measured_local"`.
  The single timing-mode call site references both
  constants.

- **Manifest schema additions**:
  `workloads/simbench_bw/schemas/measurement_manifest.schema.json`
  carries 4 new optional NVML fields (sidecar paths,
  polling interval, claim scope, collector-status
  union); `additionalProperties:false` preserved.

- **Verifier helper**:
  `workloads/simbench_bw/nvml_metadata_verifier.py`
  exposes
  `verify_nvml_metadata_pin(manifest_path,
  sidecar_dir=None)` raising
  `NvmlMetadataDriftError(field, expected, actual)`,
  `NvmlSidecarMissingError`,
  `NvmlSidecarSchemaInvalidError`. Manifests without
  ANY cached NVML field admit cleanly (emission is
  opt-in to `--enable-cupti-activity`).

- **Operator command for sm_86 live evidence**:
  ```sh
  make -C workloads/simbench_bw cupti-collector-lib
  bash tools/cloud/preflight.sh --out /tmp/simbench_preflight.json
  python3 workloads/simbench_bw/run_suite.py \
      --mode timing \
      --kernel dram_stream_stride_mix \
      --shape-point timing_00 \
      --enable-cupti-activity \
      --preflight-manifest /tmp/simbench_preflight.json
  ```
  `--preflight-manifest` is a mandatory companion of
  `--enable-cupti-activity` per `run_suite.py`'s
  `EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED`
  guard; the run rejects without it.

  Then verify the live pair:
  ```python
  from pathlib import Path
  from workloads.simbench_bw.nvml_metadata_verifier import (
      verify_nvml_metadata_pin,
  )
  manifest = Path("workloads/simbench_bw/results/.../measurement_manifest.json")
  verify_nvml_metadata_pin(manifest)  # admits or raises typed
  ```
  The round summary must record the device/driver/
  CUDA/NVBit build/clocks AND confirm at least one
  repeat's `nvml_collector_status == "ok"` with a
  non-empty `samples` array.

### (d) sm_120-side prerequisites completed in Loop 2

R60 (HEAD `87e4f05`) landed:

- **Centralized cadence**:
  `NVML_DEFAULT_POLLING_INTERVAL_NS = 5_000_000` and
  `NVML_DEFAULT_CLAIM_SCOPE = "measured_local"` module
  constants in `run_suite.py`; the timing-mode call
  site references the constants.
- **Manifest provenance fields**: 4 new optional fields
  in `measurement_manifest.schema.json` (sidecar paths,
  polling interval, claim scope, collector-status
  union); `build_measurement_manifest` accepts 4 new
  optional kwargs; `handle_timing` populates them on
  `--enable-cupti-activity` runs by walking each
  per-repeat sidecar and reducing the per-repeat
  statuses via `_summarize_nvml_collector_status`.
- **Source-level read-only-NVML guard**: 10 host-only
  tests at
  `workloads/simbench_bw/tests/test_cupti_collector_readonly_nvml_guard.py`
  scanning `cupti_activity_collector.cu` and
  rejecting any
  `nvmlDeviceSet*` / `nvmlDeviceReset*` regex match
  plus 11 named mutation literals.
- **NVML metadata-pin verifier**: new module
  `workloads/simbench_bw/nvml_metadata_verifier.py`
  with 8 host-only tests at
  `workloads/simbench_bw/tests/test_nvml_covariate_metadata_pin.py`
  (3 admit arms + 5 drift-rejection arms).
- **Schema-bound + missing-`timestamp_ns` negative
  controls**: 2 new tests in
  `workloads/simbench_bw/tests/test_nvml_covariate_path.py`
  (`test_rejects_polling_interval_above_100hz_floor`
  and `test_rejects_sample_missing_timestamp_ns`).

### (e) Most recent sm_120-side evidence

Host-only test surface at R60 implementation parent
`87e4f05` (handoff documentation commit `de3ea25` is a
descendant carrying only docs/plan/ledger updates with
no behavioral change):

- `python -m pytest workloads/simbench_bw/tests/test_cupti_collector_readonly_nvml_guard.py -q`
  → 10 passed.
- `python -m pytest workloads/simbench_bw/tests/test_nvml_covariate_metadata_pin.py -q`
  → 8 passed.
- `python -m pytest workloads/simbench_bw/tests/test_nvml_covariate_path.py -q`
  → 27 passed.
- Broad sweep
  `python -m pytest tools/ workloads/simbench_bw/tests/ -q`
  → 1944 passed, 2 skipped, 596 subtests passed.
- AC-14 + plan-token gates GREEN; `validate.py --all`,
  `verify_coverage.py --strict`, and
  `trace_size_estimator.py --validate-all` all `ok`.

No live producer-side NVML sweep on sm_120 has been
executed; the receiving sm_86 session can land that
evidence and round-trip the metadata-pin verifier
against the live sidecars.

## Entry: v6+v7 — sm_120 AC-16 Second-Workload-Eligibility (dram_stream_stride_mix/timing_00)

**Blocked on**: sm_120

### (a) Item name
`v6v7-ac16-dram-stream-stride-mix-cross-arch` (sm_120
cross-arch live capture for AC-16 second-workload-
eligibility centered on
`workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`
— specifically the dlsym hook
`simbench_cupti_record_cuda_event_wall_time_ms` + the
`cudaEventCreate`/`cudaEventRecord`/`cudaEventSynchronize`/
`cudaEventElapsedTime` bracket around the 64-iteration
warmup-+-timed loop, the `-ldl` Makefile linkage, the
`launch_config` shape augmentation in
`workloads/simbench_bw/shapes/dram_stream_stride_mix.toml`,
and the AC-16 strict-admission key path through
`workloads/simbench_bw/cupti_dual_source_live_admission.py`).

The v7 RLCR loop closed at HEAD `5ea52f3` with all 10
plan-scoped sm_86 ACs verified; the AC-16 second-workload-
eligibility leg was promoted to `Verified-live-sm_86`
(NOT `Cross-arch-verified`). The cross-arch leg requires a
live sm_120 capture on the 5060 laptop.

### (b) What is blocked
The producer-side live evidence and `Cross-arch-verified`
promotion of AC-16 second-workload-eligibility for the
dram_stream_stride_mix/timing_00 surface. Specifically:

- **A real `--mode timing --enable-cupti-activity` run
  on a 5060 laptop** with `--kernel
  dram_stream_stride_mix --shape-point timing_00` that
  produces real
  `cupti_activity_<repeat>.json` sidecars carrying
  `cuda_event_wall_time_ms` non-NaN entries paired with
  the CUPTI Activity producer's `cuda_event_id` /
  `correlation_id` keys. The wall-time field MUST come
  from the dlsym hook the v7 loop landed at
  `dram_stream_stride_mix/main.cu`'s timed-loop bracket,
  not a synthetic post-hoc patch.

- **The strict-admission helper at
  `workloads/simbench_bw/cupti_dual_source_live_admission.py`
  admits the live sm_120 capture cleanly** via
  `admit_cupti_dual_source_live_capture(timing_artifact_path,
  sidecar_dir, expected_slug, expected_shape_point)`
  returning `CuptiDualSourceLiveAdmissionResult(ok=True, ...)`
  with ZERO typed defects.

- **Negative-test parity on sm_120**: a wrong-slug
  sentinel injection still surfaces typed defects (slug-
  axis identity mismatches across the per-record contract)
  matching the count and shape of the sm_86 baseline at v7
  R1 HEAD `5ea52f3`. The receiving session captures the
  exact defect count via the helper's `collect_defects=True`
  mode and asserts cross-arch parity.

- **AC-4 cross-arch correctness invariance**:
  `sha256_f32le` digest equality between the sm_86 live
  output and the sm_120 live output for
  dram_stream_stride_mix/timing_00. The kernel uses
  identity output `out[i] = in[i]` with deterministic
  FMA-style accumulation in `sink[0]` (NOT hashed); the
  digest equality is mathematically required, not
  optional.

### (c) Precise operator-facing artifact

- **Continuation plan**:
  `plans/ac16_dram_stream_cupti_hook_sm120_plan.md` is
  the v8 plan covering ONLY the dram_stream_stride_mix
  sm_120 leg. The plan has 11 ACs (AC-1..AC-11) and
  14 tasks (task1, task1b, task2..task13). It is
  Codex-reviewed (v2 incorporates 9 REQUIRED CHANGES +
  3 SUGGESTED IMPROVEMENTS from v1) but NOT yet
  operator-approved; the laptop receiving session
  should re-validate plan readiness via the
  `humanize:plan-compliance-checker` agent before
  starting the v8 RLCR loop.

- **Producer source (already at v7 HEAD; rebuild on
  laptop)**:
  - `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu` —
    dlsym hook + cudaEvent bracket. Architecture-
    independent host-side glue; rebuild required only
    because the .so is host-local.
  - `workloads/simbench_bw/l1/dram_stream_stride_mix/Makefile` —
    `-ldl` linkage. Host-rebuild required.
  - `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml` —
    `launch_config` shape augmentation for `timing_00`
    (grid `[1,1,1]` block `[32,1,1]`). No rebuild
    needed (TOML is host-independent).
  - `workloads/simbench_bw/cupti_activity_collector.cu` +
    `Makefile` cupti-collector-lib target — the
    LD_PRELOAD'd .so. Host-rebuild required.

- **Strict-admission helper (host-only Python; no
  rebuild)**:
  `workloads/simbench_bw/cupti_dual_source_live_admission.py`
  exposes
  `admit_cupti_dual_source_live_capture(timing_artifact_path,
  sidecar_dir, expected_slug, expected_shape_point, *,
  collect_defects=False)` returning
  `CuptiDualSourceLiveAdmissionResult`. The helper rejects
  any identity-axis mismatch (slug, shape_point,
  binary_sha256, source_commit) AND the per-sidecar
  contract (cuda-event-id presence, wall-time-ms
  non-NaN, correlation-id pairing, schema validity).
  When `collect_defects=False` the helper raises on the
  first defect; when `collect_defects=True` it returns
  the full defect list for parity checks.

- **Operator command for sm_120 live evidence on the
  laptop** (per v8 plan task9; the build incantation
  uses the native gcc-14 override):
  ```sh
  NVCC='nvcc -ccbin=/usr/bin/g++-14' make -C \
    workloads/simbench_bw cupti-collector-lib
  NVCC='nvcc -ccbin=/usr/bin/g++-14' make -C \
    workloads/simbench_bw/l1/dram_stream_stride_mix
  bash tools/cloud/preflight.sh \
    --out /tmp/simbench_preflight.json
  python3 workloads/simbench_bw/run_suite.py \
      --mode timing \
      --kernel dram_stream_stride_mix \
      --shape-point timing_00 \
      --enable-cupti-activity \
      --preflight-manifest /tmp/simbench_preflight.json
  ```
  `--preflight-manifest` is a mandatory companion of
  `--enable-cupti-activity` per `run_suite.py`'s
  `EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED`
  guard; the run rejects without it.

  Then verify via the strict-admission helper:
  ```python
  from pathlib import Path
  from workloads.simbench_bw.cupti_dual_source_live_admission \
    import admit_cupti_dual_source_live_capture
  shape_dir = Path(
    "workloads/simbench_bw/results/.../timing_00"
  )
  result = admit_cupti_dual_source_live_capture(
    timing_artifact_path=shape_dir.parent / "timing_timing_00.json",
    sidecar_dir=shape_dir,
    expected_slug="dram_stream_stride_mix",
    expected_shape_point="timing_00",
  )
  assert result.ok, result
  ```
  The round summary must record the device/driver/
  CUDA/NVBit build/clocks AND confirm
  `result.ok is True` AND name the shape point.

### (d) sm_86-side prerequisites completed in v6+v7

The v7 loop (HEAD `5ea52f3`, branch `5060_3090ti_cross_dev`)
landed the following on sm_86:

- **Hook + bracket** (substantive commits):
  - `9f5a2ca` — added the
    `simbench_cupti_record_cuda_event_wall_time_ms`
    dlsym hook to
    `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`
    plus the `cudaEvent*` bracket around the warmup-+-
    timed 64-iteration loop.
  - `14e37bc` — added the `launch_config` shape
    augmentation to
    `workloads/simbench_bw/shapes/dram_stream_stride_mix.toml`
    for `timing_00` (grid `[1,1,1]` block `[32,1,1]`).
  - `eaf5eee` — added regression tests at
    `workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py`
    covering hook presence, bracket placement, and
    the exact-literal `printf("  \"wall_time_ms\":
    %.9f,\n", wall_time_ms);` count (== 2).
  - `30ae840` — text-contract cleanup (carried text
    issues from the v7 R0 Codex review).
  - `5ea52f3` — AC-7 hardening (exact-literal-count
    printf assertion in the static guard).

- **AC-16 strict-admission helper carried v3-era
  state**:
  `workloads/simbench_bw/cupti_dual_source_live_admission.py`
  was introduced at v3 R20-R23 (then iteratively
  tightened to its contract-first design); v7 did not
  modify the helper (architecture-independent host-only
  Python).

- **All 10 plan-scoped sm_86 ACs verified per Codex
  v7 Round-1 ADVANCED verdict**:
  `ACs: 10/10 addressed for the plan-scoped sm_86
  objective | Forgotten items: 0 | Unjustified
  deferrals: 0`. The plan-scoped objective was the
  sm_86 leg only; the cross-arch leg was explicitly
  deferred per the v7 plan's `Path Boundaries → Allowed
  Choices` and is the entry point for this ledger entry.

### (e) Most recent sm_86-side evidence

Live evidence at v7 R1 (HEAD `5ea52f3`):

- **Strict-admission helper positive arm**:
  `admit_cupti_dual_source_live_capture(
    timing_artifact_path=...,
    sidecar_dir=...,
    expected_slug="dram_stream_stride_mix",
    expected_shape_point="timing_00",
  )` returned `ok=True` with the full per-sidecar record
  set admitted and zero typed defects. Captured during
  the v7 R1 sweep on the 3090 Ti host with native gcc-14
  toolchain.

- **Strict-admission helper negative arm**
  (wrong-slug sentinel): the same call with
  `expected_slug="rmsnorm_fwd_bf16"` (or any non-
  matching slug) and `collect_defects=True` returned
  `ok=False` with typed slug-axis identity defects on
  every per-sidecar record (the v6 R0 regression test
  `CuptiDualSourceAdmissionSecondWorkloadParameterizedTest`
  exercises this arm at commit `eaf5eee`).

- **AC-7 static guard hardening** (commit `5ea52f3`):
  `test_wall_time_ms_field_emitted` asserts the exact
  literal `printf("  \"wall_time_ms\": %.9f,\n",
  wall_time_ms);` appears EXACTLY twice in
  `dram_stream_stride_mix/main.cu` (warmup branch +
  timed branch). 2 occurrences → admit; any other
  count → reject.

- **Targeted regression sweep**: `python -m pytest
  workloads/simbench_bw/tests/test_dram_stream_stride_mix_hook_static.py
  workloads/simbench_bw/tests/test_cupti_dual_source_live_admission.py
  -q` admitted clean at v7 R1.
- **Broad sweep**: `python -m pytest tools/
  workloads/simbench_bw/tests/ -q` admitted at v7 R1
  with the 4 known pre-existing baseline failures
  (1 in `tools/test_runbook_command_verification.py` +
  3 classification cases in
  `test_handle_timing_dual_source_hard_gate.py`); the
  v6 + v7 work introduced ZERO new failures (delta-
  green: NO new red), but the broad sweep is NOT
  uniformly green.

The receiving sm_120 session on the 5060 laptop can
land the producer-side live evidence by following the
v8 plan at
`plans/ac16_dram_stream_cupti_hook_sm120_plan.md` and
round-trip the strict-admission helper against the
fresh sm_120 sidecars.
