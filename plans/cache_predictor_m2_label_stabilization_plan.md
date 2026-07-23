# M2 Local-Label Stabilization: Repeatability Sweep + Provenance Enrichment + Live-Evidence Admission (Slice-Per-AC Test-First Lockstep)

## Goal Description

Close parent-plan AC-5 (label stabilization) by shipping the local-label tooling that allows the operator to populate `data/local_labels/label_admitted.json` with at least 5 cells per arch (the smoke-test-floor count pinned in `comparison_contract.toml`'s `coverage_gate.smoke_test_floor_per_arch`). Specifically:

1. Fix the upstream metric-name misalignment (`tools/local_labels/label_driver.sh` and `tools/local_labels/parse_ncu_metrics.py` currently use the legacy `lts__t_sector_hit_rate.pct` while `workloads/simbench_bw/comparison_contract.toml` pins `lts__t_sectors_lookup_hit_rate.pct`); without this fix the runner's contract-pinned join would reject every admitted cell.
2. Author `tools/local_labels/sweep_repeatability_matrix.py`, a parameterized harness that wraps `label_driver.sh` in nested loops over warmup_count × cache_control × repeats axes, supports a `--dry-run` mode that prints each NCU invocation's exact argv without running NCU, emits a deterministic CSV/JSON verdict-table under `data/local_labels/sweeps/<timestamp>/`, and reports a terminal verdict (`admitting_config_found` or `no_admitting_config_found`).
3. Extend `label_driver.sh` to write a `label_run_manifest.json` per measurement run capturing `source_commit` (`git rev-parse HEAD`), `kernel_signature_sha256` (looked up from the sibling `workloads/simbench_bw/.../measurement_manifest.json` or fail-closed with typed exclusion), and the binary's already-recorded `binary_sha256`; extend `tools/local_labels/repeatability_aggregator.py` to read the manifest and enrich admitted cells with the runner-joinable 7-field key plus the runner-consumed `cache_hit_rate_l2_sector_pct` value plus `profiler_version` normalized to the contract's pinned `ncu_metric_version` (`ncu_2026.1.x`) after validating the raw sidecar string matches.
4. Author the admitted-cell schema `workloads/simbench_bw/schemas/local_label_admitted.schema.json` with strict `additionalProperties: false`, bump the on-disk `label_admitted.json` `schema_version` from 1 to 2, and validate every emitted cell against the schema.
5. Author `tools/local_labels/SWEEP_RUNBOOK.md` documenting the operator command syntax, artifact paths, dry-run interpretation, required evidence capture, and the post-commit live-sweep operator workflow per `BL-20260508-post-commit-recapture-discipline`.
6. Run the live sweep on local hardware against an expanded candidate cell set (4-6 cells per arch, including the two prior failed cells `sm_120 fp32_fma_dep_chain/trace_00` and `sm_86 gemm_hmma_bf16_sm80/correctness_shape`) until at least 5 cells per arch admit under the unchanged `stddev_threshold=0.03` gate; record the outcome in `data/local_labels/label_admitted.json` and the round summary. **Per CLAUDE.md "Two-Device Local Development Model" architecture-coupling discipline ("Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other"), AC-5 splits per-arch into AC-5.1 (sm_86 evidence) + AC-5.2 (sm_120 evidence) — see `## Acceptance Criteria` AC-5 for the full split. The receiving 3090 Ti session executing this M2 RLCR loop closes AC-5.1 only; AC-5.2 is queued for a follow-on cross-host transfer back to the 5060 laptop or a separate M2-prime sm_120 loop.**

This plan binds AC-5 closure to a binary external criterion: at the end of the round, `label_admitted.json` contains at least 5 admitted cells of the receiving session's arch (sm_86 on the 3090 Ti receiving session) AND every admitted cell strict-validates against `local_label_admitted.schema.json` AND the M1 comparison runner can consume the cells without `comparison_contract_drift` exclusions. Combined sm_86 + sm_120 admission lifts the round status to `Cross-arch-verified` per CLAUDE.md, but ONLY when both per-arch sweeps have actually landed (the receiving 3090 Ti session can lift to `Verified-live-sm_86` only). The just-closed M1 loop's scope-closure stalemate is avoided by making the closure target externally testable (a JSON file's content + count) rather than by a tooling-shipped-but-no-evidence framing.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The test surface is committed BEFORE the implementation that satisfies it lands (slice-by-AC test-first ladder per `BL-20260509-strict-admission-contract-first`).

- AC-1: Metric-name alignment between `label_driver.sh` and `comparison_contract.toml`'s pinned `ncu_metric.name`.
  - Positive Tests (expected to PASS):
    - `tools/local_labels/label_driver.sh` invokes NCU with `--metrics "lts__t_sectors_lookup_hit_rate.pct,..."`.
    - `tools/local_labels/parse_ncu_metrics.py` extracts `lts__t_sectors_lookup_hit_rate.pct` from NCU output.
    - A new test asserts the driver's metric string matches the contract's pinned `ncu_metric.name` value verbatim (read from the on-disk `comparison_contract.toml`).
    - Existing per-launch sidecars under `data/local_labels/sm_*/...` continue to schema-validate after the rename (the sidecar field carries the new metric name).
  - Negative Tests (expected to FAIL):
    - The legacy `lts__t_sector_hit_rate.pct` literal in `label_driver.sh` or `parse_ncu_metrics.py` is grep-rejected by a static scanner test.
    - A contract whose `ncu_metric.name` differs from the driver's invocation string is detected by the metric-name lockstep test (mirrors `BL-20260507-arch-token-multi-surface-lockstep`).

- AC-2: Sweep harness `tools/local_labels/sweep_repeatability_matrix.py`.
  - Positive Tests:
    - The harness exposes `--dry-run` that prints each NCU invocation's exact argv (sorted, deterministic) without invoking NCU.
    - Live mode (`--live`) writes outputs under `data/local_labels/sweeps/<timestamp>/` (one subtree per sweep run; multiple runs do not collide).
    - The matrix is documented (warmup_count × cache_control × repeats; the `replay_count` axis is dropped per the resolved DEC-7 below since NCU's `--replay-mode` is not a count flag).
    - Fake-NCU smoke test: a monkeypatched subprocess runner records the argv and returns synthetic sidecar bytes; the harness aggregates and emits a verdict-table CSV/JSON whose contents schema-validate against an inline structural pin.
    - Verdict-table re-run discipline: same fixture under same `--seed` produces a byte-identical CSV/JSON.
    - Terminal-verdict shape: every sweep run emits one of `admitting_config_found` (with the specific config) or `no_admitting_config_found` (with documented evidence).
  - Negative Tests:
    - Writes outside `data/local_labels/` are rejected (a static guard test asserts the harness's only output path is under `data/local_labels/sweeps/`).
    - An unsupported NCU CLI option (e.g., a typo'd `--cache-control crank`) produces a typed harness diagnostic, not a silent fallback.
    - Non-deterministic matrix ordering (e.g., a dict iteration without `sorted()`) is rejected by the byte-identical-re-run test.

- AC-3: Provenance enrichment via `label_run_manifest.json`.
  - Positive Tests:
    - `label_driver.sh` writes a `label_run_manifest.json` at the start of each NCU invocation containing `source_commit` (from `git rev-parse HEAD`), `binary_sha256` (computed from the binary file or copied from the existing per-launch sidecar capture path), and `kernel_signature_sha256` (looked up from the sibling simbench_bw `measurement_manifest.json` if present).
    - `repeatability_aggregator.py` reads the manifest and enriches admitted cells with the 7-field join key.
    - `profiler_version` normalization: the aggregator validates that the raw sidecar string (e.g., `"Version 2026.1.1.0 (build 37634170) (public-release)"`) matches the contract's pinned `ncu_metric_version` (`ncu_2026.1.x`) prefix or an explicit version-mapping table; on match, the admitted cell records `profiler_version = "ncu_2026.1.x"` (the contract's pinned form).
  - Negative Tests:
    - Missing `label_run_manifest.json` produces a typed exclusion `label_provenance_unavailable` (no zero-fill of join-key fields).
    - Duplicate `binary_sha256` across two manifests in the same sweep subtree produces typed exclusion `label_provenance_duplicate`.
    - Mismatched `binary_sha256` between manifest and per-launch sidecar produces typed exclusion `label_provenance_mismatch`.
    - A raw `profiler_version` string that does NOT match the pinned protocol version produces typed exclusion `label_profiler_version_drift`.

- AC-4: Admitted-cell schema `workloads/simbench_bw/schemas/local_label_admitted.schema.json`.
  - Positive Tests:
    - The schema exists with `additionalProperties: false`, top-level `required` listing all 7 join-key fields plus `cache_hit_rate_l2_sector_pct`, `n_repeats`, `stddev_l2_sector`, and the schema_version field.
    - The schema's `schema_version` is pinned at `2` (bump from the current `label_admitted.json` shape's implicit v1).
    - The aggregator validates every emitted admitted-cell record against the schema; emission fails-closed if validation fails.
    - The M1 comparison runner's test surface gains a fixture-validation test that asserts a representative admitted-cell record schema-validates AND that the runner can consume it without `comparison_contract_drift`.
  - Negative Tests:
    - A record missing `source_commit` (or any other required field) is schema-rejected.
    - A record carrying an extra unknown field is schema-rejected (`additionalProperties: false`).
    - A record whose `profiler_version` is not the pinned protocol value is schema-rejected.

- AC-5: Live-evidence admission floor: `label_admitted.json` contains at least 5 admitted cells per arch. **Split per-arch per CLAUDE.md "Two-Device Local Development Model" architecture-coupling discipline** ("Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other"):
  - AC-5.1 (sm_86 evidence — `Verified-live-sm_86`): live sweep on the **RTX 3090 Ti** populates `data/local_labels/label_admitted.json` with `len([c for c in cells if c["arch"] == "sm_86"]) >= 5` admitted cells; round summary records the device (`NVIDIA GeForce RTX 3090 Ti`), driver version, CUDA toolkit version, observed clocks/throttle state per the architecture-coupling discipline. The receiving (3090 Ti) RLCR session executes this slice.
  - AC-5.2 (sm_120 evidence — `Verified-live-sm_120`): live sweep on the **RTX 5060 laptop** populates `data/local_labels/label_admitted.json` with `len([c for c in cells if c["arch"] == "sm_120"]) >= 5` admitted cells; same architecture-coupling-discipline metadata recorded. **AC-5.2 is queued for a follow-on cross-host transfer back to the 5060 laptop**; the 3090 Ti receiving session CANNOT independently verify sm_120 evidence (per CLAUDE.md). The M2 plan's lower bound REQUIRES AC-5.1 closure on the 3090 Ti; AC-5.2 closure is upper-bound and queued as a follow-on operator task. Combined AC-5.1 + AC-5.2 is `Cross-arch-verified` per CLAUDE.md.
  - Positive Tests (apply to the arch the round runs on):
    - Per-arch slice: after the live sweep on arch X, `len([c for c in cells if c["arch"] == "X"]) >= 5`.
    - Every admitted cell schema-validates against `local_label_admitted.schema.json` (per AC-4).
    - The M1 comparison runner, when invoked with `--label-admitted-path data/local_labels/label_admitted.json` and a synthetic predictor-bundles set covering the same 7-field keys for arch X, joins ≥5 headline cells for arch X and emits a `result_kind=admitted` output bundle (NOT `typed_unavailable: headline_geomean_admitted_cell_count_below_threshold`).
    - The `stddev_threshold=0.03` gate in `repeatability_aggregator.py` is unchanged (verified by a static-source-scan test).
  - Negative Tests:
    - Any path that loosens `stddev_threshold` from 0.03 is rejected by the static-source-scan test.
    - A cell admitted without the 7-field join-key fields populated produces a runner `comparison_contract_drift` exclusion (verified by an integration test).
    - A cell admitted with `profiler_version` drift produces a runner `comparison_contract_drift` exclusion.
    - A round-summary AC-5 closure claim that omits per-arch evidence metadata (device, driver, CUDA, clock state) is rejected by a round-summary review audit (architecture-coupling discipline lock).

- AC-6: Test-first commit ordering per slice (`BL-20260509-strict-admission-contract-first`).
  - Positive Tests:
    - For each AC slice (AC-1, AC-2, AC-3, AC-4), the failing-test commit precedes the implementation commit; `git log` shows the test commit's SHA before the implementation commit's SHA.
    - The plan-token scanner (`tools/test_plan_token_scanner.py`) returns `23 passed` after every commit in this round.
    - The multi-file widened workflow-marker grep (per `BL-20260509-hygiene-validation-grep-must-be-case-insensitive`) `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` returns no matches (subject to the genuinely-domain `[hit_rate_lower, hit_rate_upper]` exception).
  - Negative Tests:
    - An implementation commit landing without a preceding test commit is detected by a `git log` audit in the round summary.
    - A plan-token marker introduced into any code/test file is detected by the multi-file widened grep gate.

- AC-7: Pre-live operator runbook `tools/local_labels/SWEEP_RUNBOOK.md`.
  - Positive Tests:
    - The runbook exists with sections covering: command syntax (the exact `python3 tools/local_labels/sweep_repeatability_matrix.py --live ...` invocation), artifact paths (where verdict tables and admitted cells land), dry-run interpretation (how to read the verdict-table to pick the admitting config), required evidence capture (which artifacts the operator MUST commit after the live sweep).
    - An integration test exercises the documented dry-run command shape against fake-NCU and asserts the documented artifact paths actually appear (per `BL-20260504-runbook-command-verification`).
  - Negative Tests:
    - The runbook's documented command shape that fails the fake-NCU integration test is the regression guard against the runbook drifting to a broken shape.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. The narrow scope is binding: every AC above must close cleanly, and no widening into deferred ACs is permitted.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes:
- AC-1 through AC-7 closed; **both** AC-5.1 (sm_86 on 3090 Ti) AND AC-5.2 (sm_120 on 5060 laptop) closed and combined to `Cross-arch-verified` per CLAUDE.md "Two-Device Local Development Model" (requires the cross-host pivot back to the 5060 to land AC-5.2 in this M2 loop, OR a separate M2-prime loop after the receiving 3090 Ti loop closes AC-5.1).
- The sweep harness includes a `--candidates <list>` CLI flag accepting a comma-separated list of candidate cell slugs/shape_points; the operator can sweep custom cell sets.
- The verdict-table CSV/JSON includes per-cell-per-config diagnostic fields (per-repeat distribution, drift signature, bimodality detection) so the operator can triage failures without re-running.
- A small in-source linkage test asserts the M1 runner CLI invocation produces an `admitted` output bundle from the post-sweep `label_admitted.json` (closes the M1 → M2 producer/consumer loop).
- Round summary records the exact NCU command shape used for each admitted cell + the git SHA of the binary it ran against + the wall-clock budget consumed + the per-arch architecture-coupling discipline metadata (device, driver, CUDA, clock state).

### Lower Bound (Minimum Acceptable Scope)

The implementation includes:
- AC-1 metric-name alignment closed (the WIRE prerequisite — without it, no admitted cell can satisfy AC-5's runner-consumption test). **Architecture-independent — runs on either arch.**
- AC-2 sweep harness closed (with `--dry-run` AND `--live` modes; the verdict-table CSV/JSON shape is locked by tests). **Architecture-independent — tooling code; tests run on either arch.**
- AC-3 provenance enrichment closed (label-run manifest emission + aggregator side-input + profiler_version normalization). **Architecture-independent — driver glue + aggregator logic.**
- AC-4 admitted-cell schema closed (`local_label_admitted.schema.json` + schema_version=2 bump + aggregator-side validation). **Architecture-independent — schema + validator code.**
- **AC-5.1 (sm_86 evidence on 3090 Ti)** live-evidence admission closed: `label_admitted.json` carries ≥5 admitted cells with `arch == "sm_86"` AND every cell strict-validates against the schema AND the M1 runner consumes them without drift; round summary records the device + driver + CUDA + observed clocks per architecture-coupling discipline. **The receiving 3090 Ti RLCR session executes this slice.**
- **AC-5.2 (sm_120 evidence on 5060 laptop) is QUEUED as a follow-on cross-host operator task**, NOT required for the M2 lower-bound closure on the 3090 Ti receiving session per CLAUDE.md "Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other." The receiving 3090 Ti session closes M2 at `Verified-live-sm_86` evidence status. AC-5.2 closure (lifting to `Cross-arch-verified`) requires either a follow-on cross-host transfer back to the 5060 laptop OR a separate M2-prime sm_120 loop on the 5060.
- AC-6 test-first per slice (every AC slice has its failing-test commit before its implementation commit; verifiable from `git log`).
- AC-7 runbook closed (pre-live documentation; the operator's first live run on the receiving 3090 Ti produces the AC-5.1 evidence the round summary cites).

The lower bound REQUIRES the live sweep to land on the receiving session's arch (sm_86) before the round closes. The user's DEC-1 choice was binary admit ≥5/arch — the round cannot exit COMPLETE on tooling alone. Per the architecture-coupling discipline, "≥5/arch" is closed PER ARCH, not as an aggregate; M2's lower bound on the 3090 Ti is sm_86-only.

### Allowed Choices

- Can use:
  - Standard library only for `sweep_repeatability_matrix.py` (argparse, subprocess, pathlib, csv, json, hashlib, tempfile).
  - The existing `label_driver.sh` argument-parsing loop pattern as the model for new CLI flags.
  - The existing `repeatability_aggregator.py` `_admit()` sequential-gate structure as the model for the provenance-enrichment slot.
  - `jsonschema` for strict schema validation in the aggregator (already a project dependency).
  - The existing simbench_bw `measurement_manifest.json` as the source-of-truth for `kernel_signature_sha256` (the manifest carries this field per the existing schema).
  - `git rev-parse HEAD` (subprocess invocation) for `source_commit` extraction at label-driver invocation time.
  - Synthetic fake-NCU subprocess runners in `pytest`'s `tmp_path` for AC-2/AC-3/AC-7 test surfaces (the existing `tools/local_labels/test_label_driver_integration.py` `_NCU_SHIM_TEMPLATE` is the precedent).
- Cannot use:
  - Any NCU invocation outside `tools/local_labels/` (CLAUDE.md Critical Rule 1).
  - Any GPU-driver-root operation (Critical Rule 2). The sweep harness's `cache_control` axis MUST use NCU's read-only `--cache-control all|none` flag, not driver-state-changing nvidia-smi clock locks.
  - Any privileged CUPTI surface (Critical Rule 3).
  - Any host-driver work (Critical Rule 4).
  - Any simulator revival (Critical Rule 5).
  - Plan-token markers (`AC-N`, `Milestone`, `Phase`, `Step`, `Round N`, `M2 plan`, `upper-bound`/`lower-bound scope`, `DEC-N`, `Plan line N`) in implementation code or comments — covered by the multi-file widened workflow-marker grep gate.
  - Edits to top-level docs (`HANDOFF.md`, top-level `README.md`, per-tree `README.md`s, `BLOCKED_ON_3090TI.md`) — these are tracked-doc-hygiene scope per `BL-20260507-tracked-doc-hygiene-after-substantive-work` and require a dedicated plan.
  - Loosening `stddev_threshold=0.03` in `repeatability_aggregator.py` (the static-source-scan test enforces this).
  - Widening or reordering the 7-field join key in `comparison_contract.toml` (out of scope; M1's contract is binding).

> **Note on Deterministic Designs**: This plan is largely deterministic. The contract surfaces (metric name, 7-field join key, profiler_version normalization, schema_version bump, stddev_threshold) are pinned. The choice axes are the sweep harness's matrix axes (warmup_count × cache_control × repeats levels), the candidate cell set the operator sweeps, and the implementation patterns inside the bounded surfaces. The deferred items (replay_count axis, predictor-bundle-as-provenance-source, bounded-arm admission) are documented in `## Pending User Decisions` as resolved per Codex's first-pass + convergence-round critique.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The natural per-slice task ladder, derived from the test-first discipline:

1. **AC-1 slice (metric-name alignment)**: failing tests committed first that assert the driver's metric string matches the contract's pinned `ncu_metric.name`; implementation commit edits `label_driver.sh` lines 481, 508, 515 + `parse_ncu_metrics.py` to use the new metric name; the existing `tools/local_labels/test_parse_ncu_metrics.py` tests update to the new metric name; the existing per-launch sidecars under `data/local_labels/sm_*/` are NOT regenerated (their old metric value is preserved as historical evidence; the schema field name stays the same, only the value changes when re-collected).

2. **AC-2 slice (sweep harness)**: failing tests committed first for `--dry-run` argv shape, fake-NCU smoke, deterministic CSV/JSON ordering, byte-identical re-run, terminal-verdict shape, write-isolation. Implementation commit lands `sweep_repeatability_matrix.py` with argparse, the matrix enumeration loop, the dry-run branch, the live branch (subprocess.run wrapping `label_driver.sh`), and the verdict-table emitter.

3. **AC-3 slice (provenance enrichment)**: failing tests committed first for `label_run_manifest.json` shape, aggregator side-input enrichment, profiler_version normalization, missing/duplicate/mismatched provenance exclusions. Implementation commit edits `label_driver.sh` to emit the manifest at start of each run + edits `repeatability_aggregator.py` to read the manifest and enrich admitted cells.

4. **AC-4 slice (schema)**: failing tests committed first for `local_label_admitted.schema.json` strict validation. Implementation commit authors the schema file + integrates aggregator-side validation + bumps `label_admitted.json` schema_version to 2 + adds a fixture-validation test in the M1 runner test surface.

5. **AC-7 slice (runbook)**: failing test committed first for the runbook's documented command shape (per `BL-20260504-runbook-command-verification`'s integration-test discipline). Implementation commit authors `SWEEP_RUNBOOK.md` documenting the operator command syntax, artifact paths, dry-run interpretation, required evidence capture.

6. **AC-5 slice (live-evidence admission)**: this slice is OPERATOR work; the operator runs the sweep on local hardware (sm_86 first via 3090 Ti, sm_120 via 5060 laptop) against the candidate cell set, picks the admitting config from the verdict table, and re-runs the aggregator to populate `label_admitted.json`. The round summary records the live-sweep evidence (commit SHA at time of sweep, NCU command shape, wall-clock consumed). AC-6 test-first ordering does not apply to this slice (the deliverable is evidence, not code).

The 5 code slices land as ~10 commits (5 test-first + 5 impl); the live-sweep slice lands as 1-2 evidence commits (one for the post-sweep `label_admitted.json` + sweep verdict table, one for the round summary update). Total ~12 commits.

### Relevant References

- `tools/local_labels/label_driver.sh` (lines 50, 117-120, 478-490) — the existing driver; lines 481/508/515 hold the legacy metric name; lines 87-148 are the argument-parsing loop new flags slot into.
- `tools/local_labels/parse_ncu_metrics.py` — the safe Python parser the driver invokes; the legacy metric name appears in its CLI defaults / required arguments.
- `tools/local_labels/repeatability_aggregator.py` (lines 65-93, 199-257, 310-324) — the `CellSummary` dataclass + `_admit()` gate + admitted-cell dict emitter; the AC-3 provenance enrichment slot is the dict emitter site.
- `tools/local_labels/test_label_driver_integration.py` (lines 380-404, 294-313) — the existing test surface; `WarmupSemanticsTest` demonstrates parameter-passing patterns; `test_multiple_launches_per_repeat_aggregated` shows nested-loop aggregation.
- `tools/local_labels/TASK7_RUNBOOK.md` — the existing runbook precedent for `SWEEP_RUNBOOK.md`.
- `workloads/simbench_bw/comparison_contract.toml` (lines 31-37) — the pinned `ncu_metric.name = "lts__t_sectors_lookup_hit_rate.pct"` and `ncu_metric_version = "ncu_2026.1.x"` AC-1 and AC-3 must match.
- `workloads/simbench_bw/comparison_runner.py` — the M1 consumer; the AC-4/AC-5 runner-fixture-validation test extends this.
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` + `comparison_bundle.schema.json` — the schema-authoring patterns AC-4 mirrors (additionalProperties:false; required-field lists; pinned schema_version).
- `workloads/simbench_bw/schemas/local_label_sidecar.schema.json` — the existing per-launch sidecar schema (validates the input to the aggregator); AC-4's new schema validates the output (admitted-cell records).
- `data/local_labels/label_excluded.json` — the two prior failed cells; AC-5's candidate cell set should include these (sweep them first; if they admit, the floor is partly met).
- `data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json` — the per-launch sidecar shape AC-3 reads (carries `profiler_version` and `binary_sha256`; missing `source_commit` and `kernel_signature_sha256`).
- `workloads/simbench_bw/results/<slug>/<sku>/sm_<N>/measurement_manifest.json` — the simbench_bw build manifest AC-3 looks up `kernel_signature_sha256` from.
- `tools/cloud/preflight.sh` (lines 662-719) — the existing read-only NVML probe pattern (precedent for the upper-bound idle-state preflight; LOWER bound does NOT require this).
- `.humanize/bitlesson.md` — `BL-20260509-strict-admission-contract-first` (test-first per slice), `BL-20260509-rlcr-recovery-round-bounded-scope` (the prior loop's failure mode this plan avoids), `BL-20260509-hygiene-validation-grep-must-be-case-insensitive` (multi-file widened grep), `BL-20260507-arch-token-multi-surface-lockstep` (the metric-name lockstep test pattern), `BL-20260508-post-commit-recapture-discipline` (the live-sweep evidence happens AFTER the implementation commits land), `BL-20260504-runbook-command-verification` (the runbook integration-test discipline).

## Dependencies and Sequence

### Cross-Host Transfer Prelude (binding precondition)

This plan is authored on the **RTX 5060 laptop (sm_120)**. The M2 RLCR loop will execute on the **RTX 3090 Ti (sm_86)** after a cross-host scp transfer per CLAUDE.md "Session Handoff" + the project's `HANDOFF.md`. The transfer direction is **5060 → 3090 Ti** for AC-5.1 sm_86 evidence; AC-5.2 sm_120 evidence is queued for a follow-on transfer back (or a separate M2-prime loop) per CLAUDE.md's architecture-coupling discipline ("Codex review is one-arch-at-a-time").

Mandatory first actions on the receiving 3090 Ti host before starting the M2 RLCR loop:

1. `git -C "$(pwd)" rev-parse HEAD` resolves to the M2 plan commit (or a descendant).
2. `rm -rf .humanize/rlcr/` — the source 5060's loop state (`.humanize/rlcr/2026-05-09_17-04-56/`) is host-local and NOT portable per CLAUDE.md.
3. `nvidia-smi` confirms the device is `NVIDIA GeForce RTX 3090 Ti` with sm_86 compute capability.
4. Rebuild any host-specific binaries that came across via raw scp (NVBit tracer, CUPTI Activity collector library, per-slug binaries) per HANDOFF.md's "Verify CUDA / NVBit / NVML environment + rebuild any built artifacts" section. The 3090 Ti host gcc may differ from the 5060 laptop's; use `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool` (or the host's compatible gcc).
5. Read this plan's `## Acceptance Criteria` section (especially AC-5's per-arch split).
6. Start the M2 RLCR loop with the M2 plan as `--track-plan-file`:
   ```
   /humanize:start-rlcr-loop --max 60 --full-review-round 10 --skip-quiz --privacy --track-plan-file plans/cache_predictor_m2_label_stabilization_plan.md
   ```

### Milestones

This plan is intentionally a single-milestone plan; it advances ONE milestone (M2 — label stabilization, AC-5) within a single RLCR loop on the 3090 Ti. AC-5.2 (sm_120 evidence) is sequenced as a follow-on cross-host operator task, NOT a multi-milestone within THIS loop.

1. **M2 Label Stabilization Closure (sm_86 receiving session)**: the narrow scope above closes within a single RLCR loop on the 3090 Ti, executed slice-by-AC.
   - Stage A: AC-1 metric-name alignment (test commit + impl commit). Wire prerequisite. **Architecture-independent.**
   - Stage B: AC-2 sweep harness (test commit + impl commit). Tooling. **Architecture-independent.**
   - Stage C: AC-3 provenance enrichment (test commit + impl commit). Without the runner-joinable 7-field key, admitted cells cannot satisfy AC-5. **Architecture-independent.**
   - Stage D: AC-4 admitted-cell schema (test commit + impl commit). Wire-format pin for the producer/consumer contract. **Architecture-independent.**
   - Stage E: AC-7 pre-live runbook (test commit + impl commit). Operator-facing documentation per `BL-20260504-runbook-command-verification`. **Architecture-independent.**
   - Stage F1: **AC-5.1 (sm_86 live-evidence on 3090 Ti)** — operator runs the sweep on the 3090 Ti against an expanded candidate cell set (4-6 candidates including the prior failed cell `gemm_hmma_bf16_sm80/correctness_shape`); admitted cells land in `label_admitted.json` with `arch == "sm_86"`; round summary records the architecture-coupling discipline metadata per `BL-20260508-post-commit-recapture-discipline`. **REQUIRED for M2 lower-bound closure on the receiving 3090 Ti session.**
   - Stage F2: **AC-5.2 (sm_120 live-evidence on 5060 laptop)** — operator runs the sweep on the 5060 laptop against an expanded candidate cell set (4-6 candidates including the prior failed cell `fp32_fma_dep_chain/trace_00`); admitted cells land in `label_admitted.json` with `arch == "sm_120"`; round summary records the architecture-coupling discipline metadata. **NOT REQUIRED for M2 lower-bound closure on the 3090 Ti receiving session — queued for a follow-on cross-host transfer back to the 5060 laptop OR a separate M2-prime sm_120 loop.**

The five code stages (A-E) are ordered: A → B → C → D → E and are architecture-independent (any host can run them; the 3090 Ti session executes them). Stage F1 (3090 Ti live evidence) depends on Stages A through E being green. Stage F2 (5060 laptop live evidence) depends on Stage F1 PLUS a cross-host transfer back. AC-5.1 + AC-5.2 combined produces `Cross-arch-verified` per CLAUDE.md.

Subsequent plans (out of scope): Plan C (M3 — HRD coverage scaling, AC-6); Plan D (M4 — bounded-arm runtime production + first reportable real-label geomean, AC-7+AC-8); Plan E (M5 — cross-arch transfer, AC-9); Plan F (M6 — prior-art comparison + L1 bypass scaffold + ablation gates, AC-10+AC-11+AC-12).

### Dependency Graph

- AC-1 has no in-loop dependencies; lands first. **Architecture-independent.**
- AC-2 depends on AC-1 (sweep harness invokes the corrected driver); lands second. **Architecture-independent.**
- AC-3 depends on AC-1 (manifest references the driver's invocation context); lands third. **Architecture-independent.**
- AC-4 depends on AC-3 (schema validates the enriched admitted-cell shape); lands fourth. **Architecture-independent.**
- AC-7 depends on AC-2 (runbook documents the sweep harness); lands fifth. **Architecture-independent.**
- **AC-5.1 (sm_86)** depends on AC-1, AC-2, AC-3, AC-4, AC-7 all green; lands last as the 3090 Ti receiving session's live-evidence stage. **Architecture-coupled to sm_86.**
- **AC-5.2 (sm_120)** depends on AC-5.1 closure + a cross-host transfer back to the 5060 laptop; lands as a follow-on cross-host operator task OR a separate M2-prime loop. **Architecture-coupled to sm_120.**

There are no out-of-loop tooling dependencies (the M1 contract is locked; the simbench_bw measurement_manifest.json schema is unchanged; the existing per-launch sidecar schema is unchanged). The cross-host transfer is a project-discipline dependency (HANDOFF.md + CLAUDE.md "Session Handoff"), NOT a tooling dependency.

## Task Breakdown

Each task carries exactly one routing tag. The slice-by-AC structure produces ~10 code commits + 1-2 evidence commits.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1a | Failing tests for AC-1: assert `label_driver.sh` invokes NCU with the contract-pinned metric name; assert `parse_ncu_metrics.py` extracts by the new name; static scanner test rejecting the legacy literal; metric-name lockstep test asserting driver string == contract value verbatim. Tests fail at this commit. | AC-1, AC-6 | coding | - |
| task1b | Implementation: edit `label_driver.sh` (lines 481, 508, 515) and `parse_ncu_metrics.py` to use `lts__t_sectors_lookup_hit_rate.pct`. Update the existing `test_parse_ncu_metrics.py` fixtures to the new name. Tests from task1a turn green. | AC-1 | coding | task1a |
| task2a | Failing tests for AC-2: `--dry-run` argv shape; fake-NCU smoke; deterministic CSV/JSON ordering; byte-identical re-run under `--seed`; terminal-verdict shape (`admitting_config_found` / `no_admitting_config_found`); write-isolation; unsupported-NCU-option diagnostic. Tests fail at this commit. | AC-2, AC-6 | coding | task1b |
| task2b | Implementation: `tools/local_labels/sweep_repeatability_matrix.py` with argparse (`--dry-run`, `--live`, `--candidates`, `--seed`), matrix enumeration loop over warmup_count × cache_control × repeats, fake-NCU-friendly subprocess invocation, verdict-table CSV/JSON emitter under `data/local_labels/sweeps/<timestamp>/`. Tests from task2a turn green. | AC-2 | coding | task2a |
| task3a | Failing tests for AC-3: `label_run_manifest.json` shape (schema + required fields); aggregator side-input enrichment (admitted cells gain the 7-field key); `profiler_version` normalization round-trip; missing-manifest exclusion (`label_provenance_unavailable`); duplicate-binary_sha256 exclusion (`label_provenance_duplicate`); mismatched-binary_sha256 exclusion (`label_provenance_mismatch`); profiler-version-drift exclusion (`label_profiler_version_drift`). Tests fail at this commit. | AC-3, AC-6 | coding | task2b |
| task3b | Implementation: edit `label_driver.sh` to emit `label_run_manifest.json` at start of each run (capturing `source_commit` from `git rev-parse HEAD`, `binary_sha256` from the binary file, `kernel_signature_sha256` from the sibling simbench_bw `measurement_manifest.json` lookup); edit `repeatability_aggregator.py` to read the manifest and enrich admitted cells with the 7-field key + normalized profiler_version. Tests from task3a turn green. | AC-3 | coding | task3a |
| task4a | Failing tests for AC-4: `local_label_admitted.schema.json` exists and strict-validates the enriched admitted-cell shape; missing required field is schema-rejected; extra unknown field is schema-rejected; `profiler_version` not pinned is schema-rejected; the M1 runner test surface gains a fixture-validation test asserting a representative admitted cell schema-validates AND the runner consumes it without drift. Tests fail at this commit. | AC-4, AC-6 | coding | task3b |
| task4b | Implementation: author `workloads/simbench_bw/schemas/local_label_admitted.schema.json` (additionalProperties: false; required listing 7-field key + cache_hit_rate_l2_sector_pct + n_repeats + stddev_l2_sector + schema_version=2 const); integrate aggregator-side validation; bump emitted `label_admitted.json` schema_version to 2; add the fixture-validation test in the M1 runner test surface. Tests from task4a turn green. | AC-4 | coding | task4a |
| task5a | Failing test for AC-7: integration test that drives the documented `SWEEP_RUNBOOK.md` command shape against fake-NCU and asserts the documented artifact paths actually appear (per `BL-20260504-runbook-command-verification`). Test fails at this commit. | AC-7, AC-6 | coding | task4b |
| task5b | Implementation: author `tools/local_labels/SWEEP_RUNBOOK.md` documenting the operator command syntax (`python3 tools/local_labels/sweep_repeatability_matrix.py --live ...`), artifact paths (`data/local_labels/sweeps/<timestamp>/...`), dry-run interpretation (how to read the verdict table to pick the admitting config), required evidence capture (which artifacts the operator MUST commit after the live sweep). Test from task5a turns green. | AC-7 | coding | task5a |
| task6 | AC-5 live-evidence operator stage: operator runs the sweep on local hardware (sm_86 via RTX 3090 Ti, sm_120 via RTX 5060 laptop) against an expanded candidate cell set (4-6 per arch, including the two prior failed cells); picks the admitting config from the verdict table; the aggregator populates `data/local_labels/label_admitted.json` with ≥5 admitted cells per arch per the contract floor. Round summary records the live-sweep evidence (commit SHA at sweep time, NCU command shape, wall-clock consumed) per `BL-20260508-post-commit-recapture-discipline`. | AC-5 | coding | task5b |

## Claude-Codex Deliberation

### Agreements

- The just-closed M1 loop's scope-closure stalemate failure mode is the binding constraint. M2 must avoid recreating it by binding closure to externally testable evidence (binary admit count + schema validation + runner consumption), not to "tooling shipped" alone.
- The metric-name misalignment between `label_driver.sh` (`lts__t_sector_hit_rate.pct`) and `comparison_contract.toml` (`lts__t_sectors_lookup_hit_rate.pct`) MUST be fixed BEFORE any sweep work; without it, every admitted cell would be rejected by the runner's contract-pinned join.
- The provenance gap (label sidecar lacks `source_commit` and `kernel_signature_sha256`) MUST be closed via a label-run manifest emitted at profiling time (Codex Round 1 review's recommended path), NOT by reaching into post-hoc predictor bundles (which don't exist at label collection time).
- Stddev_threshold=0.03 MUST NOT be loosened. Sweep harness emits a terminal verdict (`admitting_config_found` / `no_admitting_config_found`); both are valid sweep outcomes, but only the former satisfies AC-5 (binary admit ≥5/arch per the user's DEC-1 choice).
- The admitted-cell schema requires a v2 bump because the field set widens materially.
- The slice-by-AC test-first ordering (failing tests + implementation per AC; ~10 commits + 1-2 evidence commits) lowers per-commit risk vs the M1-style 4-commit ladder; Codex's recommendation drove the user's DEC-6 choice.
- The sweep harness's matrix axes drop `replay_count` (NCU's `--replay-mode` is not a count flag; the metric collection drives replays internally per Codex's TECHNICAL_GAPS); the matrix is `warmup_count × cache_control × repeats`.
- AC-5's binary admit ≥5/arch closure requires expanding the candidate cell set beyond the 2 prior failed cells; the user's DEC-5 choice is to sweep 4-6 candidates per arch.

### Resolved Disagreements

- **Tooling-first vs binary-admit closure** (Codex Round 1 REQUIRED_CHANGE / DEC-1): Codex initially recommended tooling-first to minimize circuit-breaker risk; the user chose binary admit ≥5/arch. Resolution: the plan's lower bound REQUIRES the live sweep evidence (Stage F). The risk is mitigated by the slice-by-AC test-first structure (each AC slice is small; one slice's failure does not cascade) and by expanding the candidate cell set so the floor is reachable even if the 2 prior failed cells continue to fail.
- **Provenance source: predictor bundle vs label-run manifest vs label-driver CLI flags** (Codex Round 1 DISAGREE / DEC-4): predictor bundle was rejected by Codex (not available at label collection time); the user chose label-run manifest. Resolution: AC-3 binds `label_driver.sh` to emit `label_run_manifest.json` at run start with `source_commit` from `git rev-parse HEAD`, `binary_sha256` from the binary file, `kernel_signature_sha256` from the sibling simbench_bw `measurement_manifest.json` lookup.
- **NCU `--flush-cache-l2` axis** (Codex Round 1 TECHNICAL_GAP): the flag does not exist in NCU CLI. Resolution: the matrix axis is `cache_control` mapping to NCU's `--cache-control all|none` (where `all` flushes GPU caches before replay iterations).
- **NCU `replay_count` axis** (Codex Round 1 TECHNICAL_GAP): NCU's `--replay-mode` is not a count flag; metric collection drives replays internally. Resolution: the `replay_count` axis is dropped from the sweep matrix.
- **5-task all-in-one round vs slice-by-AC vs prerequisite hygiene PR** (Codex Round 1 OPTIONAL_IMPROVEMENT / DEC-6): the user chose slice-by-AC. Resolution: ~10 code commits across 5 ACs, each AC slice has a failing-test commit then an implementation commit; the test-first commit ordering is verifiable per slice.
- **Schema_version bump v1 vs v2** (Codex Round 1 OPTIONAL_IMPROVEMENT / DEC-2): Codex recommended v2 because downstream semantics change. Resolution: `local_label_admitted.schema.json` pins `schema_version = 2`; the aggregator emits `schema_version: 2` in the new `label_admitted.json` shape.
- **Cell-set scope: 2 failed only vs expand to 4-6 candidates** (Codex Round 1 QUESTIONS_FOR_USER / DEC-5): the user chose expand. Resolution: AC-5's candidate set includes the 2 prior failed cells AND 2-4 additional candidates (operator picks via the sweep harness's `--candidates` flag).
- **Live-sweep deferral to operator runbook vs in-round live evidence** (Codex Round 1 OPTIONAL_IMPROVEMENT): Codex initially suggested deferring; the user's DEC-1 choice forces in-round live evidence. Resolution: Stage F is in-round but follows `BL-20260508-post-commit-recapture-discipline` (the live sweep happens AFTER the code-commits land; the round summary records the evidence post-commit; HEAD-derived fields in `label_run_manifest.json` pin to the post-impl HEAD).
- **AC-7 runbook timing: pre-live vs post-live** (Codex Round 1 REQUIRED_CHANGE): the runbook covers command syntax, artifact paths, dry-run interpretation, required evidence capture (pre-live coverage); post-live observations from Stage F are appended to the runbook as a follow-up commit if the operator's first sweep surfaces useful diagnostic patterns.

### Convergence Status

- Final Status: `partially_converged`. One Codex first-pass review + one convergence round were executed: Codex's first pass surfaced 5 CORE_RISKS, 6 MISSING_REQUIREMENTS, 5 TECHNICAL_GAPS, 5 ALTERNATIVE_DIRECTIONS, 5 QUESTIONS_FOR_USER, and 7 CANDIDATE_CRITERIA. The convergence round Codex review produced 6 REQUIRED_CHANGES, all of which are addressed in the plan above. The user resolved DEC-1, DEC-4, DEC-5, DEC-6 explicitly via AskUserQuestion; DEC-2 (schema bump v2), DEC-3 (no-admitting-config fallback), and DEC-7 (NCU replay_count axis dropped) are recorded as resolved per Codex's recommendations. No items remain `PENDING`.

## Pending User Decisions

All seven Pending User Decisions were resolved during the convergence loop or via direct AskUserQuestion. They are recorded here for traceability; no items remain `PENDING`.

- DEC-1: M2 closure shape — tooling-first vs binary admit ≥5/arch.
  - Claude Position: Initially recommended tooling-first to minimize circuit-breaker risk after the M1 8-round circuit-breaker fire.
  - Codex Position: Codex Round 1 REQUIRED_CHANGE recommended tooling-first for the same risk reasoning.
  - Tradeoff Summary: Tooling-first eliminates hardware-variance dependency from the closure criterion; binary admit ≥5/arch produces an externally verifiable artifact (`label_admitted.json` content + count) that closes AC-5 unambiguously.
  - Decision Status: `RESOLVED: binary admit ≥5/arch`. Implementation note: AC-5's positive tests include both schema validation AND runner-consumption testing of the admitted cells. Stage F (live evidence) is in-round but follows post-commit recapture discipline.

- DEC-2: Schema version bump on `label_admitted.json` — v1 additive vs v2.
  - Claude Position: v2 (additive widening of required fields is technically a breaking change).
  - Codex Position: Codex Round 1 OPTIONAL_IMPROVEMENT recommended v2 because downstream semantics change.
  - Tradeoff Summary: v2 forces lockstep updates of any downstream reader; v1 additive risks silent admission of stale-shape cells.
  - Decision Status: `RESOLVED: v2`. Implementation note: AC-4's positive tests pin `schema_version = 2` as a const in the new schema.

- DEC-3: Fallback if sweep finds no admitting config — bounded labels / typed-unavailable / defer.
  - Claude Position: With binary-admit closure (DEC-1), `no_admitting_config_found` becomes a STALL signal — the round cannot close AC-5. The operator must expand the candidate cell set, change DVFS state, or defer M2 closure to a follow-on plan.
  - Codex Position: Codex Round 1 QUESTIONS_FOR_USER framed this as "bounded labels / typed-unavailable / defer."
  - Tradeoff Summary: Bounded labels would require widening into AC-7 (bounded-arm runtime production) which is out of scope; typed-unavailable cells don't count toward the headline geomean and don't satisfy the smoke-test floor; defer means M2 stalls and a follow-on plan attempts a different approach.
  - Decision Status: `RESOLVED: defer M2 to a follow-on plan IF the live sweep finds no admitting config across the candidate set`. Implementation note: AC-5's negative tests include the `no_admitting_config_found` terminal verdict; the round summary documents the verdict and queues a follow-on M2-prime plan if the floor is not met.

- DEC-4: Provenance source — predictor bundle vs label-run manifest vs widen `label_driver.sh` CLI.
  - Claude Position: label-run manifest (sibling to NCU artifacts at run time).
  - Codex Position: Codex Round 1 DISAGREE rejected predictor-bundle path (not available at label collection time); recommended label-run manifest.
  - Tradeoff Summary: Predictor-bundle path is post-hoc and depends on the predictor having run against the same kernel; label-run manifest is contemporaneous with the NCU artifacts and self-contained inside `tools/local_labels/`; widening `label_driver.sh` CLI flags pushes complexity to the operator's invocation script.
  - Decision Status: `RESOLVED: label-run manifest`. Implementation note: AC-3's positive tests bind `label_driver.sh` to emit `label_run_manifest.json` at run start; AC-3's negative tests cover missing/duplicate/mismatched/provider-version-drift exclusions.

- DEC-5: Cell-set scope — 2 prior-failed only vs expand to 4-6 candidates per arch.
  - Claude Position: Expand (the binary-admit closure requires ≥5/arch; 2 cells alone cannot satisfy that).
  - Codex Position: Codex Round 1 QUESTIONS_FOR_USER framed this as a coverage choice.
  - Tradeoff Summary: 2 prior-failed only is the smallest sweep but cannot satisfy DEC-1's binary-admit closure; expanding to 4-6 candidates per arch increases wall-clock budget but makes the floor reachable.
  - Decision Status: `RESOLVED: expand to 4-6 candidates per arch`. Implementation note: AC-2's `--candidates` CLI flag accepts the operator's chosen list; the SWEEP_RUNBOOK.md documents recommended candidate sets including the 2 prior failed cells.

- DEC-6: Round task structure — M1's 4-commit pattern vs slice-by-AC vs prerequisite hygiene PR.
  - Claude Position: slice-by-AC (lower per-commit risk after the M1 circuit-breaker fire).
  - Codex Position: Codex Round 1 OPTIONAL_IMPROVEMENT recommended either slice-by-AC OR prerequisite hygiene PR.
  - Tradeoff Summary: M1's 4-commit pattern is familiar but carries the same circuit-breaker risk profile; slice-by-AC has more commits but each is small and the test-first ordering is verifiable per slice; prerequisite hygiene PR splits AC-1 into a separate plan, narrowing M2 to 3 tasks.
  - Decision Status: `RESOLVED: slice-by-AC`. Implementation note: ~10 code commits (5 test-first + 5 impl), one per AC slice; the test-first commit precedes the implementation commit in each slice; verifiable from `git log` per slice.

- DEC-7: NCU sweep matrix axes — include `replay_count` axis vs drop it.
  - Claude Position: drop the axis (NCU's `--replay-mode` is not a count flag; the matrix would have unsupported levels).
  - Codex Position: Codex Round 1 TECHNICAL_GAP confirmed that `replay_count` is not a single CLI flag; metric collection drives replays internally.
  - Tradeoff Summary: Including the axis with unsupported levels would produce false-positive verdict-table entries; dropping it narrows the matrix to `warmup_count × cache_control × repeats` which is what NCU actually exposes.
  - Decision Status: `RESOLVED: drop the replay_count axis`. Implementation note: AC-2's matrix is `warmup_count × cache_control × repeats`; the `cache_control` axis maps to NCU's `--cache-control all|none`.

- DEC-8: Cross-host transfer direction + per-arch AC-5 split.
  - Claude Position: This M2 plan is authored on the RTX 5060 laptop (sm_120) but will execute on the RTX 3090 Ti (sm_86) after a cross-host scp transfer. Per CLAUDE.md "Two-Device Local Development Model" architecture-coupling discipline ("Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other"), AC-5 must split into AC-5.1 (sm_86 evidence on 3090 Ti) and AC-5.2 (sm_120 evidence on 5060 laptop). The receiving 3090 Ti session's lower bound REQUIRES AC-5.1 only; AC-5.2 is queued as a follow-on cross-host operator task OR a separate M2-prime sm_120 loop.
  - Codex Position: N/A — this decision was made at plan-refinement time (post-convergence) when the user specified the cross-arch execution model.
  - Tradeoff Summary: Single-AC framing would let the 3090 Ti session claim closure on sm_86 evidence alone (mistakenly conflating per-arch with cross-arch), violating CLAUDE.md's "Codex review is one-arch-at-a-time" discipline. Per-arch split makes the architecture-coupling boundary explicit and forces follow-on planning for sm_120 evidence; combined AC-5.1 + AC-5.2 lifts the round status to `Cross-arch-verified` per CLAUDE.md. The "Cross-Host Transfer Prelude" in `## Dependencies and Sequence` documents the mandatory first actions on the receiving 3090 Ti host (per CLAUDE.md "Session Handoff" + the project's `HANDOFF.md`).
  - Decision Status: `RESOLVED: AC-5 split into AC-5.1 (sm_86, REQUIRED on receiving 3090 Ti) + AC-5.2 (sm_120, QUEUED follow-on)`. Implementation note: AC-5.1 closure on the 3090 Ti is `Verified-live-sm_86`; AC-5.2 closure on the 5060 lifts the combined evidence status to `Cross-arch-verified` per CLAUDE.md. The Anti-Goals section codifies "no cross-arch promotion without explicit live witness on the second arch."

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Phase`, `Step`, `Round N`, `M2 plan`, `upper-bound`, `lower-bound scope`, `DEC-N`, `Plan line N`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead. Examples:
  - Reference the contract terms by their field names (`ncu_metric_version`, `cache_hit_rate_l2_sector_pct`, `stddev_threshold`), not by AC-N labels.
  - Reference the typed exclusion tokens by their literal names (`label_provenance_unavailable`, `label_provenance_duplicate`, `label_provenance_mismatch`, `label_profiler_version_drift`), not by AC-N labels.
  - The `tools/test_plan_token_scanner.py` gate enforces this; it must return `23 passed` after every commit in this round.
  - The multi-file widened workflow-marker grep (`grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/`) must return no matches except for genuinely-domain `[hit_rate_lower, hit_rate_upper]` schema-field references.

### Test-First Commit Ordering Discipline (per AC slice)

- For each of AC-1, AC-2, AC-3, AC-4, AC-7: the failing-test commit lands BEFORE the implementation commit. The test-first ordering is verifiable from `git log` per slice (a future reviewer can `git checkout` the test commit's SHA and observe that the relevant tests fail because the implementation does not yet exist).
- Stage F (AC-5 live-evidence) is operator work; it does not follow the test-first pattern (the deliverable is evidence, not code).
- Round summary records each commit's SHA and the gate output at that SHA verbatim per `BL-20260509-rlcr-recovery-round-bounded-scope`.

### Synthetic-Fixture Discipline

- The AC-1, AC-2, AC-3, AC-4, AC-7 slices use synthetic fixtures (fake NCU subprocess runners; in-memory verdict-table assertions; tmp_path-isolated label_run_manifest.json files). Tests with synthetic fixtures include the literal substring `synthetic` in the test function name.
- Stage F (AC-5) consumes REAL hardware evidence (live NCU sweeps on sm_86 + sm_120). The `data/local_labels/label_admitted.json` cells admitted in Stage F are NOT synthetic; they are operator-collected real labels.

### Architecture-Coupling Discipline (per CLAUDE.md "Two-Device Local Development Model")

- The receiving 3090 Ti session executes the M2 RLCR loop. Per CLAUDE.md "Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other," any AC-5 closure claim must explicitly name its arch. The round summary's evidence status MUST be one of:
  - `Offline-verified` — host-only unit tests pass; NO live NCU sweep ran. Acceptable for AC-1, AC-2, AC-3, AC-4, AC-7 slices that do not touch real hardware (e.g., the test-first commits + the implementation commits whose tests use fake-NCU shims). NOT acceptable for AC-5.1 closure.
  - `Verified-live-sm_86` — required for AC-5.1 closure on the receiving 3090 Ti. The round summary names device (`NVIDIA GeForce RTX 3090 Ti`), driver version, CUDA toolkit version, NVBit build (if relevant), observed clocks/throttle state, and whether evidence is behavioral / timing / SASS-layout-dependent.
  - `Verified-live-sm_120` — applies only after a follow-on cross-host transfer back to the 5060 laptop closes AC-5.2.
  - `Cross-arch-verified` — applies only after BOTH AC-5.1 (sm_86) AND AC-5.2 (sm_120) live evidence has landed across two sessions.
- Producer-side wire format (the `local_label_admitted.schema.json` schema, the `label_run_manifest.json` shape, the `repeatability_aggregator.py` admitted-cell dict) is architecture-independent per CLAUDE.md's wire-extension policy ("an offline validator built before the new value landed will REJECT the new value because the JSON Schema's `enum` list is closed"). The schema's required-field list is closed (additionalProperties: false), but the field VALUES (e.g., `arch` ∈ {`sm_86`, `sm_120`}) are union enums admitting both arches; new arch tokens would require coordinated producer + schema + validator updates per `BL-20260507-arch-token-multi-surface-lockstep`.
- Live-sweep evidence captured on the 3090 Ti is NOT cross-arch evidence on the 5060 (and vice versa). The round summary MUST NOT claim `Cross-arch-verified` until both arch-specific sweeps have landed.

### BitLesson Anchors

- `BL-20260509-strict-admission-contract-first` — the slice-by-AC test-first ordering is the procedural enforcer of the test-first contract.
- `BL-20260509-rlcr-recovery-round-bounded-scope` — the per-round bounded scope discipline is the meta-constraint; M2 narrowly addresses AC-5 only and explicitly defers AC-6 through AC-12 to Plan C-F follow-on RLCR loops.
- `BL-20260509-hygiene-validation-grep-must-be-case-insensitive` — every "absence-of-string" hygiene gate in this round (workflow-marker grep, plan-token scanner, stddev_threshold-loosening static scan) uses the multi-file case-insensitive form with the validation command pinned in the round contract.
- `BL-20260507-arch-token-multi-surface-lockstep` — the AC-1 metric-name lockstep test (driver string == contract value verbatim) is the model for the metric-name lockstep regression guard.
- `BL-20260508-post-commit-recapture-discipline` — Stage F's live sweep captures HEAD-derived fields (`source_commit` in `label_run_manifest.json`, `cache_predictor_bundle.source_commit` if cross-referenced); the live capture happens AFTER the code-commits land so the on-disk manifest pins to the post-impl HEAD.
- `BL-20260504-runbook-command-verification` — AC-7's pre-live runbook is verified by an integration test that exercises the documented command shape against fake-NCU and asserts the documented artifact paths actually appear.
- `BL-20260505-cross-arch-sass-mnemonic-divergence` — the architecture-coupling discipline backstop. M2 does not touch SASS regex surfaces directly, but the lesson's principle ("Never assume an opcode set, register count, SASS layout, helper-function inlining policy, or compression-format distribution is constant across architectures") binds AC-5's per-arch split: AC-5.1 evidence on sm_86 does NOT guarantee AC-5.2 evidence on sm_120, even though both use the same `repeatability_aggregator.py` and the same `stddev_threshold=0.03` gate. NCU's per-arch metric collection paths and the underlying GPU's per-arch DVFS / cache-replacement / boost-clock policies are arch-coupled facts, not invariants — a config that admits sm_86 cells may not admit sm_120 cells (and vice versa). The Stage F1 (sm_86) sweep verdict is NOT predictive of the Stage F2 (sm_120) sweep verdict.

### Anti-Goals (binding for this round)

- No re-implementation of M1 lower-bound surfaces (`comparison_runner.py`, `verify_artifacts.py`, `comparison_bundle.schema.json`, `comparison_contract.toml`) — already closed and verified by Codex's M1 R6 review.
- No widening into AC-6 (HRD coverage scaling, M3 / Plan C), AC-7 (bounded-arm runtime production, M4 / Plan D), AC-8 (real-label geomean reporting, M4 / Plan D), AC-9 (cross-arch transfer, M5 / Plan E), AC-10/AC-11/AC-12 (prior-art comparison + L1 bypass + ablation gates, M6 / Plan F).
- No widening of `tools/test_plan_token_scanner.py` exemption table; if any new file trips the scanner, re-prose the offending text rather than exempt the file.
- No edits to `HANDOFF.md` / top-level `README.md` / per-tree `README.md`s / `BLOCKED_ON_3090TI.md` — these are tracked-doc-hygiene scope per `BL-20260507-tracked-doc-hygiene-after-substantive-work` and require a dedicated plan.
- No widening of tracer arch token sets without lockstep (`BL-20260507-arch-token-multi-surface-lockstep`).
- No `instruction.proto` wire-format changes.
- No 34-bucket reuse-distance partition modifications.
- No `stddev_threshold` loosening from 0.03.
- No completion-gate overclaim ("the only", "sole", "completes" without scope-bounding) — the round summary uses scope-bounded language ("AC-5.1 sm_86 label stabilization is closed", NOT "AC-5 label stabilization is closed" or "the cache predictor's labels are complete").
- **No cross-arch promotion without explicit live witness on the second arch** (per CLAUDE.md "Two-Device Local Development Model" — "Codex review is one-arch-at-a-time. A round verified live on one arch is NOT verified on the other"). The receiving 3090 Ti session MUST NOT claim AC-5 closure (singular) or `Cross-arch-verified` evidence status — only AC-5.1 closure with `Verified-live-sm_86` evidence is reachable in this round. AC-5.2 (`Verified-live-sm_120`) and combined `Cross-arch-verified` status require a follow-on cross-host transfer back to the 5060 laptop.
- No editing of `.humanize/rlcr/` from a prior session's host-local state — the receiving 3090 Ti host MUST `rm -rf .humanize/rlcr/` per CLAUDE.md "Session Handoff" before starting the M2 RLCR loop. Carrying the source 5060's loop state would conflate the M1 8-round circuit-breaker session with the fresh M2 receiving session.
- No reliance on architecture-specific assumptions in the AC-1 through AC-4 / AC-7 implementations (which are architecture-independent). Tests that pin specific PCs, SASS section names, or helper-section resolution paths are architecture-coupled facts and are out-of-scope for this M2 round (per the AC-5 scope and CLAUDE.md's architecture-coupling discipline).

## Knowledge Consulted

The following Skill cards and KB cards informed the M2 plan's framing. Per `.claude/knowledge/INDEX.md` the active project direction is non-root GPU measurement and prediction; cards are cited as reference, not as authority for changes outside CLAUDE.md's Critical Rules.

- `.claude/skills/validating-nonroot-gpu-measurement/SKILL.md` — the meta-Skill for validating non-root GPU measurement code paths; binds AC-1 through AC-7 to the project's non-root constraint and the local-label exception in CLAUDE.md Rule 1.
- `.claude/knowledge/INDEX.md` — fallback inventory of KB subcorpora; consulted to identify the relevant `measurement/` cards below.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — context on per-arch DVFS / clock-state behavior; informs the AC-5 architecture-coupling discipline (the prior failed cells' high stddev may be DVFS-driven, and DVFS policy is per-arch). The `cache_control` axis in AC-2's sweep matrix targets this surface indirectly via NCU's `--cache-control all|none` flag.
- `.claude/knowledge/measurement/nonroot-cloud-contract.md` — the non-root GPU measurement contract; binds CLAUDE.md Rule 1's local-label exception ("NCU MAY be invoked on local devices ... ONLY into the quarantine path `data/local_labels/`. The driver code that performs the local invocation lives ONLY under `tools/local_labels/`") and the no-NCU-on-cloud absolute prohibition.
- `.claude/knowledge/measurement/cupti-activity-api.md` — context on the CUPTI Activity API (the only allowed CUPTI surface per CLAUDE.md Rule 3); confirms that AC-2's sweep harness MUST NOT use Events / Metrics / Range Profiling / PM Sampling / PerfWorks even inside `tools/local_labels/`.
- `.humanize/bitlesson.md` — the project's BitLesson knowledge base (host-local; survives raw scp but not git fetch). The BitLesson IDs cited in `## BitLesson Anchors` above are the operative process-discipline lessons binding M2's slice-by-AC test-first ladder, post-commit recapture, multi-file workflow-marker grep, and runbook command verification.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Parameterized NCU Repeatability Sweep For Label Stabilization

## Original Idea

Brainstorm a repo-grounded, novel, and high-quality research direction for **M2 (Plan B) — Label Stabilization, AC-5** of the parent cache-hit-rate predictor plan. This brainstorm follows the just-closed M1 lower-bound RLCR loop (8 rounds; circuit breaker fired at scope-closure stalemate; M1 lower bound + upper-bound CLI wrapper fully discharged). The brainstorm output will become the input to /humanize:gen-plan and ultimately to a fresh, narrow RLCR loop replacing the just-stalled scope-closure-stalemate loop.

## Project context (binding per CLAUDE.md)

- The repo is a non-root GPU measurement and prediction project for wall-clock behavior and cache hit ratio.
- Two-device local development model: RTX 3090 Ti (sm_86) + RTX 5060 laptop (sm_120). Both have NCU access and are valid live targets for tools/local_labels/ per the local-label exception in CLAUDE.md Critical Rule 1.
- The local-label exception (CLAUDE.md Rule 1): NCU and nv-nsight-cu-cli MAY be invoked on local devices to produce ground-truth labels for the prediction model, but ONLY into the quarantine path data/local_labels/. The driver code that performs the local invocation lives ONLY under tools/local_labels/.
- Critical Rules 2-4 (no GPU-driver-root, no privileged CUPTI, no host-driver work) bind even inside tools/local_labels/.
- Critical Rule 5 (no simulator revival) binds globally.

## What the M1 lower-bound RLCR loop just landed (state to inherit)

- workloads/simbench_bw/comparison_runner.py — library + __main__ CLI wrapper that loads comparison_contract.toml, joins admitted predictor bundles against admitted NCU label cells on the contract-pinned 7-field key (slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version), normalizes NCU percent to fraction, computes the relative-error formula, dispatches scalar/bounded/typed_unavailable arms, applies coverage gate, emits a deterministic schema-valid output bundle. The CLI accepts --contract, --predictor-bundles-glob, --label-admitted-path, --output. The runner cross-checks every label cell's profiler_version against the contract's pinned ncu_metric_version and excludes drifted cells with typed comparison_contract_drift. The runner's per-cell exclusion records carry an externally-emitted diagnostic token (one of comparison_runner_join_key_unmatched, comparison_contract_drift, predictor_typed_unavailable).
- _verify_cache_predictor_run_summaries in workloads/simbench_bw/verify_artifacts.py — verifier hook with 7 typed defect tokens covering manifest_pairing_index mismatches, sidecar digest drift, circular_predictor_input rejection, bounded-arm inversion, bundle_underconfigured cross-check, schema_invalid, and run_summary_missing_path.
- workloads/simbench_bw/schemas/comparison_bundle.schema.json — strict draft-2020-12 output schema with allOf[if/then] conditionals binding bounded-arm diagnostic fields, scalar-arm forbiddance, excluded-cell required-field shapes by reason.
- 47 test methods in tests/test_comparison_runner.py + 15 in tests/test_verify_cache_predictor_run_summaries.py.
- data/local_labels/label_admitted.json is currently {schema_version: 1, cells: []} — empty admitted set. The two prior attempted real cells live in label_excluded.json with typed_reason: label_repeatability_failed. AC-5 is the work to fix that.

## What M2 / Plan B / AC-5 must close

The M1 plan's Dependencies and Sequence section names AC-5 (label stabilization) as Plan B's scope. The acceptance criterion is approximately: the local-label pipeline under tools/local_labels/ produces NCU-derived ground-truth labels stable enough to populate data/local_labels/label_admitted.json with at least the smoke-test-floor count (5 cells per arch per the comparison contract) of admitted cells; admitted cells carry the contract-side profiler_version/ncu_metric_version provenance the comparison runner consumes; the existing stddev_threshold=0.03 repeatability bound is NOT loosened.

## Hard constraints (binding)

- All NCU invocation MUST be inside tools/local_labels/ and write only into data/local_labels/. CLAUDE.md Critical Rule 1.
- No GPU-driver-root operations. CLAUDE.md Critical Rule 2.
- No privileged CUPTI surfaces. CLAUDE.md Critical Rule 3.
- No host-driver work. CLAUDE.md Critical Rule 4.
- No simulator revival. CLAUDE.md Critical Rule 5.
- The non-circular discipline binds the predictor; AC-5 work doesn't touch the predictor input contract.
- stddev_threshold=0.03 MUST NOT be loosened.
- The 7-field join key MUST NOT be widened or reordered.
- The ncu_metric_version pin (ncu_2026.1.x) is the protocol pin; admitted cells must carry a profiler_version matching it.

## Anti-goals (binding for M2)

- No HRD coverage scaling work (AC-6 / Plan C).
- No bounded-arm runtime production work (AC-7 / Plan D).
- No real-label geomean reporting work (AC-8 / Plan D).
- No cross-arch transfer evaluation (AC-9 / Plan E).
- No prior-art comparison / L1 bypass scaffold / ablation gates (AC-10/11/12 / Plan F).
- No re-implementation of M1 lower-bound surfaces.
- No edits to HANDOFF.md / top-level README.md / per-tree README.md / BLOCKED_ON_3090TI.md (separate hygiene plan scope).
- No widening of tools/test_plan_token_scanner.py exemption table.

## Repo paths the brainstorm should reference

- tools/local_labels/ — current label-stabilization driver tree
- tools/local_labels/repeatability_aggregator.py — admitted-cell schema and aggregation logic
- data/local_labels/label_admitted.json (currently empty) and label_excluded.json (carries the two prior failed cells)
- data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json — example per-launch sidecar shape
- workloads/simbench_bw/comparison_contract.toml — pinned ncu_metric_version, 7-field join key, stddev_threshold referent
- workloads/simbench_bw/comparison_runner.py — the M1 consumer that AC-5 unblocks
- .humanize/bitlesson.md (for relevant lessons): BL-20260509-rlcr-recovery-round-bounded-scope, BL-20260509-strict-admission-contract-first, BL-20260507-arch-token-multi-surface-lockstep, BL-20260508-post-commit-recapture-discipline, BL-20260504-runbook-command-verification
- .claude/skills/validating-nonroot-gpu-measurement/SKILL.md

## Output expectations

A directed-swarm idea draft suitable for handoff to /humanize:gen-plan. Quality bar:
- Novel framings considered (not just run NCU on each kernel and see what sticks)
- Clear path-boundary discipline (lower bound = ~3 tasks fits a single round; upper bound bounded above by the plan-token scanner + Critical Rule 1-5)
- Explicit acknowledgement of the just-closed M1 loop's scope-closure stalemate failure mode and how this M2 plan's narrow scope avoids it
- Concrete acceptance-criteria seed candidates that map to AC-5 narrow ONLY (NOT into AC-6/7/8/9/10/11/12)
- Per-task routing tags (coding for Claude-direct work; analyze for /humanize:ask-codex consultation)

## Primary Direction: Repeatability Sweep Matrix

### Rationale

Build a parameterized sweep harness that systematically explores warmup-count × replay-count × repeats × L2-flush-on/off (and any other measurement-policy axes the existing label driver already exposes). Run it on a small kernel set; record per-cell admission outcome across the matrix; identify the minimum-config corner that stabilizes ≥5 cells/arch. Empirical-first approach that lets the data identify the policy floor rather than guessing.

### Approach Summary

**Core design**: Build a parameterized sweep harness (Python script) that wraps `label_driver.sh` in nested loops over:

1. **Measurement-policy axes** (initial set):
   - `warmup_count`: [0, 1, 5] — the two failed cells used warmup=1; probe no-warmup and higher warmup to see if unstable behavior improves or worsens
   - `replay_count` (or `--ncu-replays` / NCU's internal replay setting if exposed): [1, 3, 10] — default is NCU's internal default; probe lower and higher
   - `repeats` (number of driver loop iterations per cell): [3, 5, 10] — baseline is 5; try fewer/more
   - `l2_flush` (on/off) — conditionally, if NCU CLI supports it via a flag like `--flush-cache-l2` or similar
2. **Fixed cell set**: the 2 cells already in `data/local_labels/label_excluded.json` (sm_120 `fp32_fma_dep_chain/trace_00`, sm_86 `gemm_hmma_bf16_sm80/correctness_shape`).
3. **Per-cell measurement**: For each (cell, axis-level) combination, run the driver `repeats` times and invoke `repeatability_aggregator.py` with the existing `stddev_threshold=0.03` gate (NEVER loosen it).
4. **Verdict table**: Emit a structured CSV/JSON table with columns `cell_id, warmup_count, replay_count, repeats, l2_flush, n_sidecars, stddev_l2, stddev_l1tex, admitted, detail`.
5. **Minimum-config discovery**: Filter the table to find the smallest (warmup, replay, repeats, l2_flush) corner where both cells admit (stddev ≤ 0.03) with at least 5 repeats — that corner becomes the documented operator-runbook configuration that drives the rest of M2's label collection.

**Affected components**:
- `tools/local_labels/label_driver.sh` (line 478 NCU collection block; line 87-148 argument-parsing loop): add optional `--replay-count` and `--l2-flush` flags (if NCU supports them); forward them to NCU.
- `tools/local_labels/repeatability_aggregator.py`: NO CODE CHANGES NEEDED; already parameterized via CLI `--stddev-threshold` and `--repeats-required`.
- **New file** `tools/local_labels/sweep_repeatability_matrix.py`: main harness that loops over the axes, invokes `label_driver.sh`, aggregates, writes the verdict table.
- **New file** (optional, upper-bound) `tools/local_labels/SWEEP_MATRIX_RUNBOOK.md`: operator documentation for running the sweep and interpreting the verdict table.

### Objective Evidence

- `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/label_driver.sh` (lines 50, 117-120, 478-490): exposes `--warmup` parameter; hard-codes `--replay-mode kernel`, `--launch-skip`, `--launch-count 1`, and metrics list — all discoverable and patchable.
- `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/repeatability_aggregator.py` (lines 54, 58, 199-258, 260-346, 369-389): implements `_admit()` gate with parametric `stddev_threshold` (default 0.03) and `repeats_required` (default 5); both CLI-overridable via `--stddev-threshold` and `--repeats-required`.
- `/home/jakeshea/3090ti_5060laptop_cross_study/data/local_labels/label_excluded.json`: two cells with confirmed L2 stddev failures — sm_120 `fp32_fma_dep_chain/trace_00` (stddev=0.0605, 2× threshold; per-repeat distribution `[0.8364, 0.9139, 0.8096, 0.8966, 0.96]`); sm_86 `gemm_hmma_bf16_sm80/correctness_shape` (stddev=0.1414, 4.7× threshold; per-repeat distribution `[0.875, 0.5927, 0.8442, 0.8775, 0.6262]`, bimodal). Both cells collected 5 repeats (n_repeats=5), so the floor was reached but stddev was unstable.
- `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/test_label_driver_integration.py` (lines 380-404, 294-313): `WarmupSemanticsTest` class demonstrates parameter-passing pattern (driver `--warmup` overrides, forwarded workload `--warmup`); `test_multiple_launches_per_repeat_aggregated` shows the aggregator already handles multiple launches per repeat, collapsing to mean before cross-repeat stddev — evidence that the framework can handle nested loop structures.
- `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/TASK7_RUNBOOK.md` (lines 195-239, 209-211): documents the 5-iteration driver loop as the operator's manual sweep pattern; no automated sweep harness exists, but the runbook shows the operational model is already manual parametrization.

### Known Risks

- **NCU availability / permission drift**: NCU may require elevated privileges on some host configurations. Sweep harness must handle non-zero NCU exit gracefully (typed exclusion + continue).
- **Stddev orthogonality**: not clear which measurement-policy axes independently control cache-hit-rate variance. Warmup-count may decouple L2 and L1/TEX behavior differently on sm_86 vs. sm_120; full factorial design risks orthogonality collapse.
- **Data volume**: 2 cells × 4 axes × 3 levels × ~5 repeats = ~120 sidecars per arch × 2 arches = ~240 JSON files under `data/local_labels/`. Folder size and aggregation runtime grow but stay within tracked-data tolerance.
- **Replay-mode / L2-flush not yet NCU-documented for local hardware**: the NVIDIA manual may not document default replay-mode behavior or whether NCU on sm_86/sm_120 supports L2-cache-flush flags. Exploratory runs needed to confirm NCU CLI compatibility — log-and-skip if a flag is unrecognized.
- **Wall-clock budget**: NCU adds 10-100x overhead per replay. Conservative estimate: 2-4 hours wall-clock for the full matrix on each arch running sequentially; parallelizing across the two devices could halve that.

## Alternative Directions Considered

### Alt-1: Provenance-First Aggregator Widening

- Gist: Extend `tools/local_labels/repeatability_aggregator.py` to propagate `profiler_version` (and the 6 other join-key fields the comparison runner expects) into admitted-cell records. Re-process the two existing failed cells without changing any measurement-policy axis. The minimum structural change the M1 loop unblocks; tests whether the missing provenance field is the gating factor preventing the runner from consuming existing measurements, separate from any actual measurement-stability work.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/schemas/local_label_sidecar.schema.json` (line 18): the per-launch sidecar schema REQUIRES `profiler_version`; the field already exists on disk in `data/local_labels/sm_120/fp32_fma_dep_chain/trace_00/label_0_0.json` as `"Version 2026.1.1.0 (build 37634170) (public-release)"`.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/repeatability_aggregator.py` (lines 310-324): the `aggregate()` admitted-cell dict currently emits only `device, slug, shape_point, n_repeats, stddev_l2_sector, stddev_l1tex_sector` — missing `profiler_version` AND `binary_sha256, source_commit, kernel_signature_sha256, arch, cache_hit_rate_l2_sector_pct`.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/comparison_runner.py` (lines 721-722, 548-557): the runner expects every admitted cell to carry `profiler_version`; without it the runner emits `comparison_contract_drift`.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/comparison_contract.toml` (line 37): `ncu_metric_version = "ncu_2026.1.x"` is the pinned protocol anchor that admitted cells must match.
- Why not primary: the direction is mechanically correct but blocked by an upstream schema gap. The label sidecar carries `profiler_version` but does NOT carry `source_commit` or `kernel_signature_sha256`. Even after propagating `profiler_version`, the 7-field join will still fail on the missing fields (cells will be excluded with `join_key_unmatched`/`unmatched_field=source_commit` before profiler_version mismatch is ever tested). And the prior-failed cells are actually failing on stddev (0.0605, 0.1414), not on the join key — so propagating provenance wouldn't admit them anyway. Direction folds into the primary as a complementary task: the sweep harness's verdict table proves which configs pass stddev; the provenance-first aggregator widening then admits those passing cells with the right schema. Both tasks needed; primary picks the empirical-first half because it answers the load-bearing question (which config admits?).

### Alt-2: Bootstrap-CI Admission Gate

- Gist: Keep `stddev_threshold=0.03` as a sanity floor unchanged AND ADD a bootstrap CI half-width gate (e.g., 95% CI half-width < 0.02) computed from the per-cell repeats. Admit cells only if BOTH gates pass. Statistical strengthening of admission criterion without loosening any pinned floor; reuses the EXACT bootstrap CI machinery from `comparison_runner._bootstrap_ci_over_cells` and `cache_predictor._bootstrap_ci_over_launches`.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/comparison_runner.py` (lines 626-658): `_bootstrap_ci_over_cells()` implements bootstrap CI with cell-level resampling; takes per-cell values, returns `(ci_low, ci_high, ci_half_width)`; uses `random.Random(seed)` for determinism; pinned `BOOTSTRAP_RESAMPLES=2000`, `CI_CONFIDENCE=0.95`.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/cache_predictor.py` (lines 177-211): `_bootstrap_ci_over_launches()` is the parallel pattern at the launch unit; same constants.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/comparison_contract.toml` (line 92): `ci_half_width_target = 0.05` is pinned at the geomean level over admission cells; the label-side gate could mirror at tighter tolerance.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/repeatability_aggregator.py` (lines 65-93, 199-257): `CellSummary.repeats` already carries per-repeat hit rates; `_admit()` is sequential and CI gate would slot after stddev.
- Why not primary: this direction MAKES ADMISSION STRICTER, which is the opposite of AC-5's goal of reaching the 5-cell-per-arch floor. The two prior failed cells already fail the stddev gate; adding a CI gate on top can only reject MORE cells, not admit more. CI gating is the right NEXT step AFTER cells are admitted (to ensure their stability is high), not the way to reach the floor in the first place. Folds into the primary as a follow-on policy: once the sweep harness identifies an admitting config, the CI gate can be added as a quality bar for future label collection.

### Alt-3: Typed-Failure Taxonomy Expansion

- Gist: Replace the single `label_repeatability_failed` exclusion reason with a finer-grained taxonomy of named failure modes (e.g., `dvfs_drift_during_run`, `single_outlier_repeat`, `bimodal_distribution`, `cold_start_warmup_insufficient`, `launch_overhead_variance_dominates`, `unexplained_high_variance`). Each token names a specific per-repeat distribution shape with a corresponding remediation policy. Diagnostic-first approach that converts binary admit/exclude into a routable workflow.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/repeatability_aggregator.py` (lines 13-21, 199-257): existing typed-reason contract is binary (`label_repeatability_missing`, `label_repeatability_failed`, `label_provenance_inconsistent`).
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/cache_predictor.py` (lines 62-74): `_TYPED_REASONS` frozenset (11 entries) — frozenset+lockstep prior art.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/nvbit_tracer/test_exclusion_taxonomy_parity.py` (lines 109-125): `LOCAL_LABELS_TYPED_REASONS_EXPECTED` frozenset already exists — the parity test would extend cleanly.
  - The two failed cells exhibit distinct distribution shapes (sm_120 high-variance; sm_86 textbook bimodal) — concrete data points for ≥2 distinct taxonomy tokens.
- Why not primary: the direction names failure modes but doesn't implement remediation. Admission count remains zero. The taxonomy is also speculative — only 2 failed cells exist to validate the 5-token shape. Folds into the primary as a downstream enrichment: the sweep harness's verdict table empirically reveals which axis flip moves a cell from one failure mode to another, which is the data needed to validate the taxonomy.

### Alt-4: Easy-Kernel Seed Set

- Gist: Pick a small kernel set deterministic-by-construction (FMA chains, fully-streaming DRAM reads, no atomic ops, no shared memory) as the initial calibration target. Admit those FIRST to validate the end-to-end M1 → real-label pipeline before tackling harder kernels. Operational-first approach.
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/splits.toml`: 45 calibration rows; cross-arch availability filterable from `roster.toml` and per-kernel `.cu` source.
  - 9 candidate easy kernels identified: `dram_stream_stride_mix`, `l2_thrash_pointer_chase`, `fp32_fma_dep_chain`, `fp8_e4m3_qmma_dep_chain`, `tf32_hmma_gemm_traceable`, `int8_imma_gemm_traceable`, `fp4_e2m1_mma_dep_chain` (sm_120 only); `wgmma_bf16_saturation`, `wgmma_fp8_saturation` (sm_90a only — undeployable locally).
  - Only 2 kernels (`dram_stream_stride_mix`, `l2_thrash_pointer_chase`) support sm_86; sm_86 floor is borderline.
  - `fp32_fma_dep_chain` IS in the easy set AND is one of the prior failed cells (stddev=0.0605). Easy-by-construction did not equal stable-in-measurement.
- Why not primary: the prior failed cell (`fp32_fma_dep_chain`) is in the proposed easy set and already failed, contradicting the operational premise. The sm_86 coverage is also borderline (2 kernels). Folds into the primary as a kernel-selection input: the sweep harness can run on the easy set first, but the failure pattern from `fp32_fma_dep_chain` already shows that easy-by-construction doesn't suffice — measurement-policy sweep is the load-bearing variable.

### Alt-5: Idle-State + DVFS-Quiescence Preflight

- Gist: Add a preflight stage to the label driver that checks observable-by-non-root system state (read-only NVML clock readings, ambient temperature, no other GPU processes) before each measurement run. Reject runs where the preflight fails; tag failures with a typed exclusion. Reduces non-stationary-environment noise at COLLECTION TIME (not aggregation time).
- Objective Evidence:
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/cloud/preflight.sh` (lines 662-719): existing NVML capability probe demonstrates read-only `nvidia-smi` queries for clock, temperature, power covariates via `pynvml`; lines 673-710 show production-quality `nvmlInit()`, `nvmlDeviceGetPowerUsage()`, `nvmlDeviceGetTemperature()` with permission-degradation handling.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/nvml_covariate_join.py` (lines 184-202): per-launch NVML covariate aggregation already collects `temp_c`, `power_mw`, `sm_clock_mhz`, `mem_clock_mhz`; regression layer at lines 230-310 fits temperature/power/clock offsets against timing residuals — clock and thermal variance ARE recognized as confounding factors.
  - `/home/jakeshea/3090ti_5060laptop_cross_study/tools/local_labels/label_driver.sh` (lines 36, 478-487): `--clock-control none` flag already embedded; preflight bridges this — NCU is told `--clock-control none` (do not lock); preflight separately observes whether the GPU's clock is already drifting due to DVFS, and rejects if so.
- Why not primary: high false-rejection-rate risk if the quiescence envelope is too strict; ground truth on what envelope reduces stddev requires empirical iteration. Direction is also OPERATIONAL noise reduction, not the same as the parameter sweep that finds the admitting config. Folds into the primary as a complementary upstream gate: once the sweep identifies an admitting config, adding a preflight makes that config more reliably reachable on contended/thermally-noisy systems.

## Synthesis Notes

The five alternatives fold cleanly into the primary's sweep-harness ladder rather than competing with it. Alt-1 (Provenance-First Aggregator Widening) is the schema/structural fix that MUST land alongside the primary — it's the smallest task in the primary's lower-bound (~30 LOC + tests); the primary lower-bound BUNDLES it because without `profiler_version` in admitted cells the runner can't consume them regardless of stddev. Alt-2 (Bootstrap-CI Admission Gate) is a follow-on quality bar to apply AFTER the sweep identifies an admitting config; explicitly out-of-scope for M2 (it strengthens admission, doesn't admit more cells). Alt-3 (Typed-Failure Taxonomy Expansion) is data-driven enrichment: the sweep's verdict table reveals which axis flip moves a cell between failure modes, providing the empirical basis for the taxonomy; can fold in if the verdict table data is rich enough, otherwise defer to a later loop. Alt-4 (Easy-Kernel Seed Set) is a kernel-selection heuristic: the sweep can run on the easy set first, but `fp32_fma_dep_chain`'s prior failure shows easy-by-construction doesn't suffice, so kernel selection is a secondary input not the primary lever. Alt-5 (Idle-State + DVFS-Quiescence Preflight) is collection-time noise reduction: complementary to the sweep but conceptually orthogonal (sweep finds the right config; preflight makes that config reliably reachable); can fold in as an upper-bound element if time permits.

The primary's narrow scope (one round, ~3 tasks, lower-bound = sweep harness + provenance widening + verdict-table-based admit run on the two failed cells) avoids the just-closed M1 loop's scope-closure stalemate failure mode by having a CONCRETE empirical objective (populate `data/local_labels/label_admitted.json` with at least 5 cells per arch) rather than a structural objective (close all the lower-bound surfaces). The acceptance criterion is binary and externally verifiable: after the round, `label_admitted.json` is either ≥5/arch or it isn't. There's no review-heuristic-vs-plan-scope conflict because the deliverable IS the cell admission, not a contract surface.

--- Original Design Draft End ---
