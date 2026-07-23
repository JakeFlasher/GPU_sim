# Memory-Microbench + Activation HEAD-Coherent Retrace + Re-Label For First Verified-Live-Sm_86 Headline Geomean

## Goal Description

Produce the FIRST end-to-end `Verified-live-sm_86` headline geomean predictor-vs-NCU relative-error number for the non-root cache-hit-rate stack on the local RTX 3090 Ti by closing the predictor↔label join on a HEAD-coherent **13-cell cohort**:

- **8 memory-microbench cells** (all at `shape_point=trace_00`): `working_set_capacity_sweep`, `l2_thrash_pointer_chase`, `dram_stream_stride_mix`, `carveout_sensitivity_probe`, `warm_cold_pair_relaunch`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, `register_spill_local_mem`.
- **5 activation cells**: `layernorm_fwd_bf16_sm80/timing_shape`, `rmsnorm_fwd_bf16_sm80/timing_shape`, `silu_fwd_bf16_sm80/correctness_shape`, `silu_fwd_bf16_sm80/timing_shape`, `softmax_safe_bf16_sm80/timing_shape`.

All 13 cells run on `arch=sm_86` and are HEAD-coherent (NVBit traces and NCU labels both at the round's pre-sweep HEAD). The deliverable is a `comparison_bundle.json` with `result_kind="admitted"` for `sm_86`, populated `per_arch_diagnostics.per_arch.sm_86`, and a real `geomean_relative_error` scalar plus bootstrap CI. The number is THE BASELINE that every subsequent predictor improvement (per-record L1 bypass, post-hoc calibration head, hardware-constant calibration, %clock64 latency reconstruction, eBPF + CUPTI bandwidth fusion) measures itself against.

The round is bounded by four discipline-binding facts surfaced through the Codex first-pass + second-pass critiques and the user's resolution of DEC-1 through DEC-6:

1. **`shape_point` is `trace_00`, not `trace_trace_00`.** The draft's `trace_trace_00` is the per-shape result-directory name (`trace_<shape>`); the join-key value is `trace_00`. Confirmed by reading `workloads/simbench_bw/results/working_set_capacity_sweep/rtx_3090_ti/sm_86/trace_trace_00.json` (`"shape_point": "trace_00"`).
2. **Existing NVBit traces are NOT at HEAD; existing activation labels are NOT at HEAD either.** Seven of the 8 microbench traces carry `source_commit=38a353d3...`; one (`dram_stream_stride_mix`) carries `source_commit=5ea52f3a...`. The 5 already-admitted activation labels in `data/local_labels/label_admitted.json` carry `source_commit=0c16e335586009ff2d11f0a311276f2cb317267f`. HEAD is `7b85b1f`. The 7-field comparison-runner join key includes both `source_commit` and `binary_sha256`, so a fresh-build label sweep at HEAD will join ZERO cells against these historical artifacts. **DEC-1 RESOLVED to Option B** — regenerate ALL trace artifacts at HEAD; **DEC-6 RESOLVED to Re-label-at-HEAD** — also re-collect the 5 activation NCU labels at HEAD. This produces a fully HEAD-coherent 13-cell cohort.
3. **No activation NVBit traces exist anywhere in the repo today.** The 5 activation slugs live in `workloads/portable_lcd/<slug>_sm80/` (not `workloads/simbench_bw/l1/<slug>/`) and have no `dynamic_trace.pb` under `workloads/simbench_bw/results/<slug>_sm80/...`. The round must capture them from scratch at HEAD per DEC-2's "Microbench + activation" cohort choice.
4. **The 5 historical activation labels at `0c16e335` remain in `label_admitted.json` after this round; they are SUPERSEDED, not removed.** The comparison runner will naturally report them as `comparison_runner_join_key_unmatched` in `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts` (because no predictor bundle joins them at the HEAD source_commit). The round summary explicitly documents this expected exclusion count of 5 to make the supersession bookkeeping auditable (per AC-10).

The cross-arch follow-on (the same sweep on the RTX 5060 laptop sm_120 host to lift to `Cross-arch-verified` per CLAUDE.md "Two-Device Local Development Model") is QUEUED as a separate cross-host operator task; this round closes only at `Verified-live-sm_86`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Each AC slice's failing-test commit precedes its implementation commit per `BL-20260509-strict-admission-contract-first`.

- AC-1: Pre-live join-key audit harness `tools/local_labels/provenance_audit.py` exists and FAILS CLOSED on any 7-field mismatch between the predictor side (the `manifest_pairing_index` derived from `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/<shape_dir>.json` per the `trace_<shape>` directory convention) and the operator-staged label-sweep measurement manifest. The harness is mandatory — the live NCU sweep cannot launch until the audit prints `PASS` for all 13 candidate cells (8 microbench + 5 activation per DEC-2).
  - **Field provenance:** the 7 join-key fields are sourced as follows by the audit:
    - **Staged in the operator measurement manifest** (4 fields, matching the existing `tools/local_labels/SWEEP_RUNBOOK.md` schema): `slug`, `shape_point`, `binary_sha256`, `kernel_signature_sha256`.
    - **Derived at audit time** (3 fields): `source_commit` from `git -C <repo> rev-parse HEAD` (audit-time HEAD must equal the predictor-side trace's recorded `source_commit` per AC-3); `arch` from the operator-supplied `--arch sm_86` flag cross-checked against `tools/arch_capabilities.json`; `ncu_metric_version` from `comparison_contract.toml`'s `[ncu_metric].ncu_metric_version` (`"ncu_2026.1.x"`).
  - Positive Tests:
    - A staged manifest whose `(slug, shape_point=trace_00, binary_sha256, kernel_signature_sha256)` matches the predictor-side trace metadata, AND audit-time `git rev-parse HEAD` equals the predictor-side `source_commit`, AND `--arch sm_86` is admitted by `arch_capabilities.json`, AND `ncu_metric_version="ncu_2026.1.x"` matches the contract, produces `PASS` and exit 0.
    - The harness reads `comparison_contract.toml` to source the canonical join-key field list (no hard-coded duplicate).
    - The harness's output is byte-deterministic for the same inputs (`--seed`-stable); a regression test pins the canonical output for a fixture pair.
  - Negative Tests (each produces a typed diagnostic and exit 1; covers all 7 join-key fields plus structural-validation modes):
    - A staged manifest with `slug` differing from the predictor's recorded value produces `provenance_audit_slug_mismatch`.
    - A staged manifest with `binary_sha256` differing produces `provenance_audit_binary_sha256_mismatch`.
    - Audit-time `git rev-parse HEAD` differing from the predictor's recorded `source_commit` produces `provenance_audit_source_commit_mismatch` (covers DEC-1 Option B's "regenerate at HEAD" gate).
    - A staged manifest with `kernel_signature_sha256` differing produces `provenance_audit_kernel_signature_sha256_mismatch`.
    - A staged manifest declaring `shape_point: "trace_trace_00"` (the directory name) instead of `"trace_00"` (the join-key value) produces `provenance_audit_shape_point_drift`.
    - The operator's `--arch` flag carrying a value outside `arch_capabilities.json`'s admitted set produces `provenance_audit_arch_unrecognized`.
    - The contract's `ncu_metric_version` differing from the audit's pinned `"ncu_2026.1.x"` (e.g., a contract amendment landed without updating the audit's pin) produces `provenance_audit_ncu_metric_version_drift`.
    - A staged manifest with `schema_version` ≠ 2 (the SWEEP_RUNBOOK.md-pinned manifest schema_version; distinct from `local_label_admitted.schema.json`'s schema_version 3) produces `provenance_audit_manifest_schema_version_drift`.
    - A staged manifest missing any of the 4 required staged fields (slug/shape_point/binary_sha256/kernel_signature_sha256) produces `provenance_audit_manifest_required_field_missing`.
    - A predictor-side trace whose `manifest_pairing_index` is missing any of the 5 fields (slug/shape_point/binary_sha256/source_commit/kernel_signature_sha256) produces `provenance_audit_predictor_manifest_incomplete`.

- AC-2: `shape_point="trace_00"` discipline across all this round's surfaces (not the directory name `trace_trace_00`).
  - Positive Tests:
    - Every staged measurement manifest under `data/local_labels/sweeps/<sweep_id>/manifests/<slug>.json` declares `shape_point: "trace_00"`.
    - The sweep harness invocation `sweep_repeatability_matrix.py --candidates sm_86/<slug>/trace_00 …` accepts the cell-id form per the existing parser.
    - The aggregated admitted cell records in `data/local_labels/label_admitted.json` carry `shape_point: "trace_00"` for the 8 microbench slugs.
  - Negative Tests:
    - A staged manifest with `shape_point: "trace_trace_00"` is rejected by `validate_measurement_manifest.py` with a typed `shape_point_directory_name_drift` diagnostic.
    - A round summary that uses `trace_trace_00` as a `shape_point` value (rather than the directory path component) is rejected by a static-source-scan test.

- AC-3: HEAD-coherent retrace prerequisite — the **13** NVBit traces (8 microbench under `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_trace_00/` + 5 activation under `workloads/simbench_bw/results/<slug>_sm80/rtx_3090_ti/sm_86/<shape_dir>/` per the `trace_<shape>` directory convention) are regenerated at the M-D pre-sweep HEAD (a single repository HEAD that both label-time `git rev-parse HEAD` and predictor-time `git rev-parse HEAD` reference) BEFORE the live NCU sweep launches. Each fresh trace's `source_commit` equals that pre-sweep HEAD; each fresh trace's `binary_sha256` equals the SHA of the freshly-built per-slug binary at the same HEAD. **DEC-1 RESOLVED to Regenerate-at-HEAD; DEC-6 RESOLVED to Re-label activation at HEAD.** The activation slugs are sourced from `workloads/portable_lcd/<slug>_sm80/`; their NVBit-tracer entry points need to be wired into `workloads/simbench_bw/run_suite.py --mode trace` (or an equivalent orchestrator) IF the existing run_suite does not already support portable_lcd kernels — this is a task5b precondition.
  - Positive Tests:
    - For each of the 8 microbench slugs, `jq -r .source_commit workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_trace_00.json` returns the pre-sweep HEAD SHA.
    - For each of the 5 activation slugs, the matching per-shape sidecar (e.g., `workloads/simbench_bw/results/layernorm_fwd_bf16_sm80/rtx_3090_ti/sm_86/<shape_dir>.json`) carries `source_commit` equal to the pre-sweep HEAD SHA.
    - For each of the 13 cells, the trace's `binary_sha256` equals `sha256sum <build_path>` of the freshly-built binary at the same HEAD (microbench: `workloads/simbench_bw/l1/<slug>/<slug>`; activation: `workloads/portable_lcd/<slug>_sm80/<slug>_sm80_run` or the canonical artifact path).
    - The 13 `dynamic_trace.pb` files exist at HEAD.
  - Negative Tests:
    - A trace whose `source_commit` differs from the pre-sweep HEAD is rejected by AC-1's audit before the sweep launches.
    - A trace whose `binary_sha256` differs from the freshly-built binary at HEAD is rejected by AC-1's audit.
    - A retrace attempt that fails for any of the 13 cells (build error, NVBit injection failure, decoder-gap-dropped > 0) blocks the sweep prerequisite gate and surfaces the first-failed slug.
    - An activation slug whose NVBit-tracer entry point is missing from the orchestrator (run_suite.py or equivalent) is detected at the M-B' phase as a typed precondition failure (NOT silently skipped).

- AC-4: NVBit trace coverage — exactly 13 trace bundles exist at HEAD, each schema-validating and admitting through the existing trace-contract verifier.
  - Positive Tests:
    - 13/13 `dynamic_trace.pb` files present (8 microbench + 5 activation).
    - Each per-shape sidecar JSON validates against the existing schema and reports `admitted_record_count == expected_record_count` and `actual_drop_rate == 0.0`.
    - The existing `test_nvbit_trace_contract_live.py` sweep (under `SIMBENCH_REQUIRE_LIVE_GPU=1`) admits all 13 fresh bundles.
  - Negative Tests:
    - A bundle with `bundle_underconfigured == true` is admitted at the trace-contract level but flagged at AC-6 as eligible only for the predictor's bounded arm or typed-unavailable arm (not the scalar arm).
    - A bundle with `decoder_gap_dropped > 0` is rejected.
    - An activation cell whose NVBit trace overhead is too high (kernel slowdown ratio > a documented per-slug threshold) is admitted at the trace-contract level but flagged in the round summary as a covariate concern (the predictor's per_launch hit-rate aggregation is overhead-insensitive in principle, but extreme overheads can change the modeled access pattern via different cache state at trace time).

- AC-5: Live-evidence admission floor — `data/local_labels/label_admitted.json` contains at least 5 cells with `arch == "sm_86"` AND `evidence_kind == "live"` AND `source_commit == <pre-sweep HEAD>` AND `slug ∈ HEADLINE_COHORT_13` (where `HEADLINE_COHORT_13 = {8 microbench slugs, 5 activation slugs}`). Aspirational target is 13 admitted cells; the smoke-test floor is 5/arch per `coverage_gate.smoke_test_floor_per_arch = 5` from `comparison_contract.toml`. Per CLAUDE.md "Two-Device Local Development Model" architecture-coupling discipline, AC-5 closure on this 3090 Ti session is `Verified-live-sm_86` only; the matching sm_120 leg is queued.
  - Positive Tests:
    - Post-aggregation HEAD-coherent cell count `len([c for c in label_admitted.cells if c.arch == "sm_86" and c.slug in HEADLINE_COHORT_13 and c.source_commit == pre_sweep_HEAD and c.evidence_kind == "live"]) >= 5`. Aspirational: 13/13.
    - Each admitted cell strict-validates against `workloads/simbench_bw/schemas/local_label_admitted.schema.json` schema_version 3 (already in tree per the per_arch_diagnostics plan), including the required `evidence_kind` field.
    - Aggregator `tools/local_labels/repeatability_aggregator.py` emits `evidence_kind: "live"` for each admitted record (already implemented at line 381; no aggregator change required).
    - The `stddev_threshold = 0.03` gate in the aggregator is unchanged (verified by the existing static-source-scan test).
    - Round summary records the device (`NVIDIA GeForce RTX 3090 Ti`), driver version, CUDA toolkit version, NVBit build, and observed clocks/throttle state per the architecture-coupling discipline.
    - The 5 historical activation labels at `source_commit=0c16e335...` REMAIN in `label_admitted.json` (NOT removed); they are SUPERSEDED-by-history at this round's HEAD and are accounted for in AC-10's bookkeeping artifact.
  - Negative Tests:
    - Fewer than 5 admitted HEAD-coherent cells produces a documented `no_admitting_config_found` terminal verdict at the operator runbook AFTER the AC-12 full matrix has been exercised AND the comparison-runner output reports `result_kind="typed_unavailable"` with `typed_reason="headline_geomean_admitted_cell_count_below_threshold"`. This is a path to round-CLOSED-WITH-NEGATIVE-EVIDENCE, not round-failure: the round still emits a verifiable artifact.
    - Any path that loosens `stddev_threshold` from 0.03 is rejected by the static-source-scan test.
    - A cell admitted with `arch != "sm_86"` is excluded from this round's per-arch admission count for sm_86 (the cell may still admit and be retained in `label_admitted.json` for future cross-arch use, but does NOT count toward this round's `Verified-live-sm_86` evidence).
    - A round-summary AC-5 closure claim that omits per-arch evidence metadata (device, driver, CUDA, clock state) is rejected by the architecture-coupling discipline review.
    - Removing the 5 historical activation labels from `label_admitted.json` (instead of preserving them as SUPERSEDED) is rejected: the supersession bookkeeping requires the historical records on disk to surface as `comparison_runner_join_key_unmatched` in AC-10.

- AC-6: Predictor bundle production at HEAD — for each of the 13 cells (8 microbench + 5 activation), `cache_predictor.py` is invoked against the HEAD-coherent trace and emits a predictor bundle (e.g., `cache_predictor_run_<slug>_<shape_point>.json`) that strict-validates against `workloads/simbench_bw/schemas/cache_predictor.schema.json` and joins to the corresponding label cell on the 7-field key.
  - Positive Tests:
    - 13 predictor bundles exist; each carries `algorithm_version: "smith_eklund_lru_v1"` (the existing baseline; no algorithm change in this round).
    - Each bundle's `manifest_pairing_index` 5-field cross-check matches the trace's metadata.
    - At least one bundle per cell carries `result_kind in ["scalar", "bounded"]` (i.e., not all 13 collapse to `typed_unavailable`).
    - The verifier hook `_verify_cache_predictor_run_summaries` in `workloads/simbench_bw/verify_artifacts.py` admits all 13 bundles.
  - Negative Tests:
    - A bundle with `typed_reason="circular_predictor_input_rejected"` is rejected (would indicate a regression of the non-circular discipline).
    - A bundle with `typed_reason="input_sidecar_digest_drift"` is rejected.
    - A bundle whose `manifest_pairing_index` mismatches the trace's metadata is rejected by the verifier hook.
    - A predictor bundle for an activation slug whose underlying NVBit trace does not exist (e.g., a missing portable_lcd activation retrace) is detected at the orchestrator level as a typed precondition failure (NOT silently skipped).

- AC-7: Comparison runner end-to-end emits `result_kind="admitted"` for sm_86. The deliverable artifact `comparison_bundle.json` carries `top_level.result_kind == "admitted"`, `per_arch_diagnostics.per_arch.sm_86.coverage_gate_verdict.meets_floor == true`, `per_arch_diagnostics.per_arch.sm_86.headline_count_live >= 5`, and a numeric `geomean_relative_error` plus bootstrap CI. The bundle is produced by invoking `comparison_runner.py` with the admitted labels (post-AC-5) and the predictor bundles (post-AC-6).
  - Positive Tests:
    - `jq -r .result_kind comparison_bundle.json` returns `"admitted"` for the top level.
    - `jq -r .per_arch_diagnostics.per_arch.sm_86.coverage_gate_verdict.meets_floor` returns `true`.
    - `jq -r .per_arch_diagnostics.per_arch.sm_86.relative_error_stats.mean` returns a finite float.
    - Bootstrap CI fields (`geomean_ci_low`, `geomean_ci_high`) are populated and `geomean_ci_high >= geomean_ci_low`.
    - The bundle's join_key field list matches the 7-field contract pinned in `comparison_contract.toml`.
  - Negative Tests:
    - A run where post-join cell count < smoke floor produces `result_kind="typed_unavailable"` with `typed_reason="headline_geomean_admitted_cell_count_below_threshold"`. This is the documented round-CLOSED-WITH-NEGATIVE-EVIDENCE path; the round summary must cite this verdict honestly rather than retry until success.
    - A bundle that silently includes a `typed_unavailable` cell in the geomean computation is rejected by the existing `typed_unavailable_handling="exclude"` runner contract.

- AC-8: Cross-arch claim discipline — the round summary asserts ONLY `Verified-live-sm_86` evidence. No `Verified-live-sm_120` or `Cross-arch-verified` claim is made.
  - Positive Tests:
    - Round summary explicitly states "AC-8 / cross-arch status: Verified-live-sm_86 only; sm_120 follow-on queued for the next cross-host transfer to the RTX 5060 laptop."
    - Round summary records device, driver, CUDA, NVBit build, and observed clocks for the 3090 Ti per BL-20260508-post-commit-recapture-discipline.
  - Negative Tests:
    - A round summary that claims `Cross-arch-verified` for any AC is rejected by the architecture-coupling discipline review.
    - A round summary that omits the queued-sm_120 follow-on note is flagged.

- AC-9: Diagnostic bucket annotation per cell — each admitted cell in the `comparison_bundle.json`'s per-cell records carries a `predictor_design_coverage` enum field with value `predictor_friendly` or `predictor_hostile`. The bucket assignment is contract-pinned (NOT deferred to round-summary commentary) per the rationale below; the `per_arch_diagnostics` block extends to emit per-bucket relative-error stats alongside the headline. Subject to DEC-4 (hostile-probe reporting policy) — if the user resolves DEC-4 to a different option (e.g., "exclude hostile from headline"), AC-9 narrows to that resolution.

  **Bucket assignment with per-cell rationale (contract-pinned in `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` or a `[predictor_design_coverage]` TOML section). The 13 HEADLINE_COHORT_13 cells partition into 8 friendly + 5 hostile:**
  - **`predictor_friendly` (8 cells)** — LRU stack-distance directly models the dominant cache behavior:
    - `working_set_capacity_sweep/trace_00` — sweeps working-set bytes across L1/L2 capacity boundaries; the Smith-Eklund modeled-hit rule (`rd < capacity_bytes // 32`) is in-domain. Volkov-Demmel 2008 / Smith-Eklund 2014 in-domain.
    - `l2_thrash_pointer_chase/trace_00` — pointer-chase exceeds L2 capacity → near-zero hit rate; LRU correctly predicts the floor.
    - `dram_stream_stride_mix/trace_00` — stream pattern with predictable stride; the per-launch reuse-distance histogram captures the access pattern faithfully.
    - `layernorm_fwd_bf16_sm80/timing_shape` — element-wise mean+var reduce + normalize; mostly streaming with O(N) reuse on the input vector inside the reduce; high cache hit rate (94.46% historical) is in-domain for LRU.
    - `rmsnorm_fwd_bf16_sm80/timing_shape` — element-wise RMS reduce + normalize; same reasoning as layernorm; 95.31% historical.
    - `silu_fwd_bf16_sm80/correctness_shape` — element-wise SILU; small working set; 57.21% historical (lower than other activations because correctness_shape uses a smaller N that may thrash). LRU is still the right model class.
    - `silu_fwd_bf16_sm80/timing_shape` — element-wise SILU on larger working set; 97.06% historical; LRU in-domain.
    - `softmax_safe_bf16_sm80/timing_shape` — max-reduce + exp + sum-reduce + normalize; multi-pass over the same vector → high reuse; 95.69% historical; LRU in-domain.
  - **`predictor_hostile` (5 cells)** — the fully-associative LRU baseline does NOT model the dominant cache behavior:
    - `carveout_sensitivity_probe/trace_00` — sweeps L1 / shared-memory carveout ratio (25/50/75); the predictor uses a single fixed `cache_model_capacity_bytes` per arch and has no carveout-aware capacity adjustment.
    - `l2_persisting_window_sweep/trace_00` — exercises L2 persisting-allocation policy; the LRU baseline has no persisting-window awareness.
    - `l2_prefetch_hint_sweep/trace_00` — exercises prefetch-hint cache modifiers; the LRU baseline does not model prefetch-induced fills.
    - `warm_cold_pair_relaunch/trace_00` — relaunch pair preserves cache state across kernel boundaries; the LRU baseline resets per kernel launch (no stateful cross-launch carry-over). RECLASSIFIED from `predictor_friendly` per Codex second-pass: relaunch cache state is a stateful policy behavior the LRU baseline does not model, even though the per-launch access pattern itself is regular.
    - `register_spill_local_mem/trace_00` — spills to local memory (LDL/STL paths); the predictor's sector_stream consumer does not separately model the local-memory cache hierarchy from global-memory access.

  - Positive Tests:
    - Per-cell records carry `predictor_design_coverage` field with one of the two enum values matching the contract pin.
    - `per_arch_diagnostics.per_arch.sm_86.relative_error_stats_per_predictor_design_coverage.{predictor_friendly,predictor_hostile}` is populated for both buckets when both buckets have admitted cells (≥1 each).
    - The bucket assignment is sourced from a pinned constant `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` in `comparison_runner.py` OR a `[predictor_design_coverage]` TOML section in `comparison_contract.toml` (with matching `ComparisonContract` sub-config + loader if the TOML path is chosen, mirroring the `[per_arch_diagnostics_config]` precedent and avoiding the silent-TOML anti-pattern).
    - Set-equality between the constant's 8-slug list and the round's MICROBENCH_8 cohort is enforced by a static-source-scan test.
    - Each bucket assignment carries an in-source rationale comment (one line per slug) referencing the per-slug rationale documented in this plan and in the round summary.
  - Negative Tests:
    - A cell with `predictor_design_coverage` outside the two-token enum is rejected at schema validation.
    - A bucket assignment drift (e.g., a slug moved between buckets without a contract amendment) is detected by the static-source-scan lockstep test.
    - A new microbench slug added to the cohort without a corresponding bucket assignment is detected by the set-equality lockstep test.

- AC-10: Historical-supersession bookkeeping for the 5 activation labels at `source_commit=0c16e335...` — these labels REMAIN in `data/local_labels/label_admitted.json` after the round (NOT removed) and are SUPERSEDED-by-history at HEAD. The comparison runner naturally reports them as `comparison_runner_join_key_unmatched` in `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts` because no predictor bundle joins them at the HEAD source_commit. The round summary explicitly documents this expected exclusion count of 5 to make the supersession bookkeeping auditable. **Reframed from the prior "activation-cell exclusion artifact" framing (which assumed activation cells lacked predictor bundles entirely)** because DEC-2 + DEC-6 resolution puts activation cells INSIDE the headline cohort with HEAD-coherent NVBit traces and HEAD-coherent re-labels.

  **Bookkeeping design:**
  - The historical 5 labels at `0c16e335` stay on disk in `label_admitted.json` for audit trail / regression-test purposes; they are not removed.
  - The HEAD-coherent re-labels of the same 5 activation cells are APPENDED to `label_admitted.json` (the file becomes `13 HEAD cells + 5 historical cells = 18 admitted records` AT MINIMUM, plus any prior microbench-or-other admitted records preserved).
  - The runner's natural per-arch excluded_reason_counts handles the 5 historical cells correctly: each fails join with `comparison_runner_join_key_unmatched` (no predictor bundle exists at `0c16e335`).
  - The round summary explicitly cites: "AC-10 supersession bookkeeping: 5 historical activation labels at source_commit=0c16e335... appear in per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched as expected; these are SUPERSEDED-by-history, not headline-cohort failures."
  - No new schema field is added; no `comparison_bundle.schema.json` amendment; no `[per_arch_diagnostics_config]` vocabulary change. The bookkeeping is purely natural-runner-output + round-summary-text discipline.

  - Positive Tests:
    - `comparison_bundle.json`'s `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched` reports a count `>= 5` (the 5 historical activation cells) when the comparison_runner is invoked over the FULL `label_admitted.json` (NOT a filtered subset).
    - The round summary documents this expected count of 5 with the supersession framing.
    - The 5 historical activation records remain in `label_admitted.json` (verifiable by `jq` over the file before and after the round).
  - Negative Tests:
    - Removing the 5 historical activation labels from `label_admitted.json` (instead of preserving them as SUPERSEDED) is rejected by AC-5's negative-test lockstep.
    - A round summary that silently omits the supersession bookkeeping commentary is flagged by the round-summary review audit.
    - An attempt to widen `[per_arch_diagnostics_config].diagnostic_vocabulary` with a new `historical_supersession` or `predictor_bundle_unavailable` token (instead of using the existing `comparison_runner_join_key_unmatched`) is rejected by the existing per_arch_diagnostics lockstep test (regression guard against scope creep).

- AC-11: Test-first per slice + hygiene gates (BL-20260509-strict-admission-contract-first + BL-20260508-post-commit-recapture-discipline). For each AC slice landing code (AC-1, AC-2, AC-9, AC-10), the failing-test commit precedes the implementation commit. The plan-token scanner `tools/test_plan_token_scanner.py` returns `23 passed` after every commit. The widened workflow-marker grep `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` returns no matches in implementation surfaces (subject to the genuinely-domain `[hit_rate_lower, hit_rate_upper]` exception). Live-sweep-stage test-first uses fake-NCU shims (the precedent from M2's AC-2 sweep harness) so the failing-tests precede live evidence. Subject to DEC-5 (test-first shape for live evidence).
  - Positive Tests:
    - For each AC slice, `git log --oneline` shows the failing-test commit SHA before the implementation commit SHA.
    - `tools/test_plan_token_scanner.py` returns `23 passed` after every commit in this round.
    - The widened workflow-marker grep returns no matches in `tools/local_labels/` or `workloads/simbench_bw/` implementation surfaces.
    - The fake-NCU integration test for the new `provenance_audit.py` (AC-1) admits before live evidence is captured.
  - Negative Tests:
    - An implementation commit landing without a preceding test commit is detected by a `git log` audit in the round summary.
    - A plan-token marker introduced into any code/test file is detected by either the regex scanner or the widened grep.

- AC-12: Full sweep matrix per DEC-3 RESOLUTION. Each of the 13 cells is exercised under the matrix `warmup_count ∈ {0, 1, 5, 10} × cache_control ∈ {all, none} × repeats = 5` = 40 NCU invocations per cell × 13 cells = **520 NCU invocations total** (~4.3 hrs operator wall-clock at ~30s/invocation). The matrix is documented in `SWEEP_RUNBOOK.md` as the canonical sweep shape for THIS round. There is no narrow-first sub-mode in this round (DEC-3 RESOLVED to full matrix; the user accepts the wall-clock cost in exchange for the most thorough characterization).
  - Positive Tests:
    - The invocation of `sweep_repeatability_matrix.py --live --candidates …` uses the full matrix; the verdict-table CSV records the matrix axes used per cell.
    - For each of the 13 cells, all 40 NCU-invocation per-cell tuples are recorded in the verdict table.
    - The `SWEEP_RUNBOOK.md` documents the full matrix as the round's canonical shape.
  - Negative Tests:
    - A run that uses a narrower matrix (e.g., warmup={0,5} only) is rejected by the operator runbook's static-text assertion (the runbook pin enforces full-matrix discipline this round).
    - A run with `repeats != 5` is rejected (the existing aggregator's stddev computation expects 5+ repeats; loosening would weaken the admission gate).
    - An auto-loop / nested-retry implementation is rejected (the matrix is fixed; widening if any cell fails to admit is a separate, post-round follow-on decision OUTSIDE this round's scope).

- AC-13: Anti-goals and CLAUDE.md hard rules honored. All NCU invocations live under `tools/local_labels/`; no Rule 2/3/4/5 violations; the static guard `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` passes after every commit; the local-label exception's scoped allowlist test continues to pass for `tools/local_labels/` and `data/local_labels/`.
  - Positive Tests:
    - `pytest workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py -q` returns `passed` after every commit.
    - The `provenance_audit.py` and any new module in `tools/local_labels/` carries no NCU invocation outside the existing exception scope.
    - The retraced NVBit traces are produced via the existing `tools/nvbit_tracer/` (LD_PRELOAD with `tracer_tool.so`); no privileged CUPTI surface is introduced.
  - Negative Tests:
    - Any new NCU invocation outside `tools/local_labels/` is rejected by the static guard.
    - Any `nvidia-smi` clock-lock / persistence / MIG / power-mutation invocation is rejected by the static guard.
    - Any privileged CUPTI surface (Events, Metrics, Range Profiling, PM Sampling, PerfWorks) is rejected by the static guard.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes AC-1 through AC-13 closed; the headline geomean number lands as `result_kind="admitted"` for sm_86 with all 13 cells admitted (8 microbench + 5 activation, all HEAD-coherent); the `predictor_design_coverage` per-cell bucket annotation is fully wired into `per_arch_diagnostics` with the 8-friendly + 5-hostile partition contract-pinned; the historical-supersession bookkeeping per AC-10 surfaces the 5 historical activation cells in `comparison_runner_join_key_unmatched`; the full sweep matrix per AC-12 (520 NCU invocations) admits all 13 cells; the round summary cites Smith-Eklund 2014, NVArchSim HPCA 2021, Volkov-Demmel 2008 as prior art for the predictor-friendly / predictor-hostile partition; the existing 1329-test suite returns the same passing count as the pre-merge baseline; the per_arch_diagnostics-plan deliverables (evidence_kind: live, per-arch coverage gate, comparison-runner contract drift exception) remain green. The cross-arch sm_120 follow-on is QUEUED (with a documented operator handoff to the next 5060-laptop session) but NOT executed in this round.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes AC-1 through AC-8 closed AND AC-11 + AC-13 (test-first hygiene + anti-goals) closed. AC-9 (diagnostic buckets), AC-10 (historical-supersession bookkeeping), and AC-12 (full matrix) are upper-bound; their lower-bound substitutes are: (a) a documented bucket partition ONLY in the round summary text (no schema-level field) for AC-9; (b) a round-summary text note ONLY (no enforced excluded_reason_counts assertion) for AC-10; (c) any matrix shape that exercises ≥10 NCU invocations per cell as long as the smoke floor (5 admitted cells per arch) is met for AC-12.

The lower bound REQUIRES: (1) the live NCU sweep to land on the 3090 Ti before round close (operator-evidence stage F per the M2 plan precedent); (2) the comparison_bundle.json to carry `result_kind="admitted"` for sm_86 (which means ≥5 of the 13 candidates admit AND join to predictor bundles); (3) the round summary to record device + driver + CUDA + NVBit + clocks per architecture-coupling discipline. If the live sweep reaches `no_admitting_config_found` despite the full matrix, the round closes at AC-7's negative-test path (`result_kind="typed_unavailable"` with `typed_reason="headline_geomean_admitted_cell_count_below_threshold"`) — this is a documented round-CLOSED-WITH-NEGATIVE-EVIDENCE outcome that still produces a verifiable comparison_bundle.json artifact, NOT a round-failure.

The 13-cell aspirational target (8 microbench + 5 activation HEAD-coherent) is upper-bound; the 5-cell smoke floor is lower-bound. The mid-bound case (e.g., 8 of 13 admit) lands a `result_kind="admitted"` bundle with a meaningful headline geomean even if the publication-grade floor (10/arch) is not reached.

### Allowed Choices

- Can use:
  - The existing `tools/local_labels/sweep_repeatability_matrix.py`, `tools/local_labels/label_driver.sh`, `tools/local_labels/repeatability_aggregator.py`, `tools/local_labels/validate_measurement_manifest.py`, `tools/local_labels/parse_ncu_metrics.py` infrastructure unchanged.
  - The existing `workloads/simbench_bw/cache_predictor.py` with `algorithm_version: "smith_eklund_lru_v1"` unchanged (no per-record L1 bypass, no calibration head, no algorithm bump in this round).
  - The existing `workloads/simbench_bw/comparison_runner.py` library + CLI; AC-9/AC-10 extensions are additive (new optional fields in the schema; backward-compatible).
  - The existing `tools/nvbit_tracer/tracer_tool/tracer_tool.so` rebuilt at HEAD per the M2 plan's `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool` recipe.
  - The existing `workloads/simbench_bw/run_suite.py --mode trace` orchestration to regenerate the 8 microbench traces at HEAD.
  - Fake-NCU subprocess shims in `pytest`'s `tmp_path` for AC-1/AC-11 test surfaces (precedent: `tools/local_labels/test_label_driver_integration.py::_NCU_SHIM_TEMPLATE`).
  - `jsonschema` for strict schema validation (already a project dependency).
  - `tomllib` (Python 3.11+) for any TOML pin (e.g., `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` if expressed as a contract section rather than an in-source constant).
  - Python stdlib for `provenance_audit.py` (argparse, subprocess, pathlib, json, hashlib).
- Cannot use:
  - Any NCU invocation outside `tools/local_labels/` (CLAUDE.md Critical Rule 1).
  - Any GPU-driver-root operation (Critical Rule 2).
  - Any privileged CUPTI surface (Critical Rule 3).
  - Any host-driver work (Critical Rule 4).
  - Any simulator revival (Critical Rule 5).
  - Loosening `stddev_threshold = 0.03` in `repeatability_aggregator.py` (the existing static-source-scan test enforces this).
  - Widening or reordering the 7-field join key in `comparison_contract.toml` (out of scope; M1's contract is binding).
  - Bumping `cache_predictor.py`'s `algorithm_version` (this round uses the v1 baseline; algorithm changes are queued for follow-on rounds and are explicitly Alt-1, Alt-2, Alt-3 in the draft).
  - Bumping `comparison_bundle.schema.json` `schema_version` (AC-9/AC-10 fields land as additive optional; existing bundles without them must continue to validate).
  - Bumping `local_label_admitted.schema.json` `schema_version` (it was just bumped to 3 in the per_arch_diagnostics plan; this round inherits and uses).
  - Plan-token markers (`AC-N`, `Milestone`, `Phase`, `Step`, `Round N`, `M2 plan`, `upper-bound`/`lower-bound scope`, `DEC-N`, `Plan line N`) in implementation code or comments — covered by AC-11.
  - Edits to top-level docs (`HANDOFF.md`, top-level `README.md`, per-tree `README.md`s, `BLOCKED_ON_3090TI.md`) — these are tracked-doc-hygiene scope per `BL-20260507-tracked-doc-hygiene-after-substantive-work` and require a dedicated plan.
  - Edits to `plans/cache_predictor_per_arch_diagnostics_plan.md` or any prior plan file (`BL-20260510-plan-file-frozen-during-active-rlcr-session`).

> **Note on Deterministic Designs**: Several AC surfaces are deterministic and have no choice axes left after the Codex first-pass critique:
> - `shape_point` value (`trace_00`, NOT `trace_trace_00`) — pinned by the trace metadata at `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_trace_00.json`.
> - 7-field join-key field list — pinned by `comparison_contract.toml`.
> - `evidence_kind: live` field on admitted cells — pinned by the per_arch_diagnostics plan.
> - `algorithm_version: "smith_eklund_lru_v1"` — pinned by this round's "no algorithm change" boundary.
> - `stddev_threshold = 0.03` — pinned by the existing static-source-scan test.
> All 6 choice axes (DEC-1 through DEC-6) RESOLVED via AskUserQuestion. The plan is now fully deterministic on: provenance policy (DEC-1 = Regenerate at HEAD), headline cohort (DEC-2 = Microbench + activation, 13 cells), sweep matrix shape (DEC-3 = Full matrix, 520 invocations), hostile-probe reporting (DEC-4 = Single geomean + diagnostic buckets), test-first shape for live evidence (DEC-5 = Commit runbook + schema + integration first), and activation provenance interaction (DEC-6 = Re-label activation at HEAD).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The natural per-slice ladder, derived from the test-first discipline and the slice dependency graph (incorporates DEC-1 through DEC-6 resolution):

1. **AC-1 + AC-2 slice (provenance audit + shape_point normalization)**: failing tests committed first asserting the new `provenance_audit.py` rejects every documented mismatch (10 modes per the expanded AC-1 list: slug / binary_sha256 / source_commit / kernel_signature_sha256 / shape_point=trace_trace_00 / arch unrecognized / ncu_metric_version drift / staged-manifest schema_version drift / manifest required-field missing / predictor-side manifest_pairing_index incomplete) and accepts a matched fixture; the audit is a small (~250 LOC) Python module that reads the trace metadata JSON and a staged manifest JSON, derives 3 of the 7 join-key fields at audit time (source_commit from `git rev-parse HEAD`, arch from `--arch` cross-checked against `arch_capabilities.json`, ncu_metric_version from `comparison_contract.toml`), and computes the 7-tuple equality. Implementation lands the module + fixture pair + the integration test that wires it into `sweep_repeatability_matrix.py`'s preflight gate.

2. **AC-3 + AC-4 slice (HEAD-coherent retrace, 13 cells)**: failing tests committed first asserting the trace-shape sidecars at HEAD have `source_commit == git_rev_parse(HEAD)` and `binary_sha256 == sha256(fresh_binary)`; the existing `test_nvbit_trace_contract_live.py` admits all 13 fresh bundles (8 microbench + 5 activation). Implementation is operator-driven: rebuild the tracer at HEAD, build all 13 binaries (8 microbench under `workloads/simbench_bw/l1/<slug>/` + 5 activation under `workloads/portable_lcd/<slug>_sm80/`), run `python3 workloads/simbench_bw/run_suite.py --mode trace --kernel <slug> --shape-point <shape_point>` for each (or extend the orchestrator to support portable_lcd activation slugs if needed).

3. **AC-11 hygiene slice (test-first per slice)**: continuous (every commit). The plan-token scanner + widened workflow-marker grep are post-commit gates per `BL-20260508-post-commit-recapture-discipline`.

4. **AC-12 full-matrix sweep slice**: documentation-only AC; lands as a `SWEEP_RUNBOOK.md` extension that documents the full-matrix operator discipline (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5` = 40 invocations per cell × 13 cells = 520 invocations, ~4.3 hrs operator wall-clock at ~30s/invocation).

5. **AC-5 slice (live NCU sweep, 13 cells)**: OPERATOR work. The operator runs `python3 tools/local_labels/sweep_repeatability_matrix.py --live --candidates sm_86/<slug>/<shape_point> ... --measurement-manifests <staged>` for all 13 cells after the AC-1 audit prints `PASS` for all 13. The aggregator `tools/local_labels/repeatability_aggregator.py` populates `data/local_labels/label_admitted.json` (appending the 13 HEAD-coherent cells; the historical 5 activation cells at 0c16e335 remain on disk for AC-10's supersession bookkeeping). Wall-clock: ~4.3 hrs for 520 NCU invocations. AC-6 test-first ordering does NOT apply to this slice (the deliverable is evidence, not code).

6. **AC-6 slice (predictor bundle production at HEAD, 13 cells)**: failing test committed first asserting the verifier hook admits 13 fresh predictor bundles and rejects circular-input / digest-drift / mpi-mismatch fixtures. Implementation lands the orchestrator script (e.g., `tools/run_predictor_for_headline_cohort.py`) that invokes `cache_predictor.predict_cache_hit_rate()` for each of the 13 traces and writes the bundle to a deterministic path.

7. **AC-7 slice (comparison runner end-to-end)**: failing tests committed first asserting the runner admits the 5+ HEAD-coherent joined cells and emits a non-typed-unavailable bundle with `result_kind="admitted"` for sm_86. Implementation lands the operator-runnable command shape (e.g., a thin CLI wrapper at `tools/run_comparison_for_headline_cohort.py`) that invokes `comparison_runner.run_comparison()` with the post-AC-5 labels and post-AC-6 bundles.

8. **AC-9 + AC-10 slice (diagnostic buckets + supersession bookkeeping)**: failing tests committed first for the contract pin and the supersession-count assertion. Implementation lands `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant in `comparison_runner.py` (or `[predictor_design_coverage]` section in `comparison_contract.toml`) covering all 13 cells with the 8-friendly + 5-hostile partition; per-bucket relative_error_stats computation in `comparison_runner.py`. AC-10 is round-summary-text + natural-runner-output discipline; no schema field added (the supersession bookkeeping surfaces as `comparison_runner_join_key_unmatched` in the existing per-arch diagnostics naturally because the 5 historical activation cells at 0c16e335 lack predictor bundles at HEAD).

9. **AC-8 + AC-13 slice (architecture-coupling discipline + anti-goals)**: documentation + static-guard-test slice. The round summary records the Verified-live-sm_86 evidence metadata; the static guard tests continue to admit.

The 9 code slices land as ~16-20 commits (test-first + impl per slice + AC-12 doc + AC-3/AC-4 retrace as operator evidence), plus 1-2 round-summary commits. Total ~18-22 commits.

### Relevant References

- `workloads/simbench_bw/cache_predictor.py` — Smith-Eklund LRU predictor (732 LOC); `algorithm_version = "smith_eklund_lru_v1"` (line 46); `predict_cache_hit_rate()` API at line 403; modeled-hit rule at line 420; bootstrap CI at lines 177-211.
- `workloads/simbench_bw/comparison_runner.py` — predictor↔label join (~1500 LOC); `_bundle_join_key()` / `_cell_join_key()` at lines 692-720; per_arch_diagnostics block; coverage_gate_verdict; `--predictor-bundles-glob` at line 38.
- `workloads/simbench_bw/comparison_contract.toml` — 7-field join key (lines 61-70); `[ncu_metric]` `name = "lts__t_sector_hit_rate.pct"` (line 40); `[coverage_gate]` `smoke_test_floor_per_arch = 5` (line 118); `[per_arch_diagnostics_config]` (lines 151-157).
- `workloads/simbench_bw/schemas/cache_predictor.schema.json` — predictor bundle schema; oneOf scalar/bounded/typed_unavailable; `manifest_pairing_index` 5-field constraint.
- `workloads/simbench_bw/schemas/comparison_bundle.schema.json` — comparison bundle schema; AC-9 adds OPTIONAL additive fields (no schema_version bump); AC-10 is round-summary-text + natural-runner-output discipline (no new schema field).
- `workloads/simbench_bw/schemas/local_label_admitted.schema.json` — admitted-cell schema_version 3 (just bumped in the per_arch_diagnostics plan); requires `evidence_kind: ["live", "synthetic"]`.
- `tools/local_labels/sweep_repeatability_matrix.py` — operator harness; `--candidates`, `--dry-run`, `--live`, `--measurement-manifests` flags; existing matrix axes warmup × cache_control × repeats.
- `tools/local_labels/label_driver.sh` — only NCU caller; lines 28-31 + 99 document `--shape-point <shape_point>` for arbitrary shape names; `--measurement-manifest <path>` (line 166).
- `tools/local_labels/repeatability_aggregator.py` — `evidence_kind: "live"` emission at line 381 (already implemented; no aggregator change required); `stddev_threshold = 0.03` immutable.
- `tools/local_labels/validate_measurement_manifest.py` — 13634-byte validator; AC-2 extends with `shape_point_directory_name_drift` typed reason.
- `tools/local_labels/SWEEP_RUNBOOK.md` — operator workflow; AC-12 extends with the full-matrix operator discipline section per DEC-3.
- `tools/nvbit_tracer/tracer_tool/tracer_tool.so` — built dynamically; AC-3 retrace prerequisite (rebuild via `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`).
- `workloads/simbench_bw/run_suite.py --mode trace --kernel <slug> --shape-point trace_00` — the per-slug retrace command shape.
- `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_trace_00.json` — per-shape sidecar with the canonical `manifest_pairing_index` fields (slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, etc.); AC-1's audit reads from here.
- `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` — repo-wide static guard; AC-13 verifies it stays green.
- `workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py` — live trace contract sweep; AC-4 verifies it admits all 13 fresh bundles (8 microbench + 5 activation).
- `tools/test_plan_token_scanner.py` — the 23-passing scanner gate; AC-11 enforces.
- `data/local_labels/label_admitted.json` — current state has 5 sm_86 activation cells at `source_commit=0c16e335...` (schema_version 3); this round APPENDS up to 13 HEAD-coherent cells (8 microbench + 5 activation re-labeled at HEAD per DEC-1 + DEC-6); the 5 historical activation cells at 0c16e335 REMAIN on disk as SUPERSEDED-by-history per AC-10's bookkeeping (they surface naturally as `comparison_runner_join_key_unmatched` in the per-arch diagnostics).
- `.humanize/bitlesson.md`:
  - `BL-20260507-arch-token-multi-surface-lockstep` — the contract-vs-runner enum lockstep pattern AC-9 mirrors.
  - `BL-20260509-strict-admission-contract-first` — test-first per slice (AC-11).
  - `BL-20260509-rlcr-recovery-round-bounded-scope` — narrow round scope discipline.
  - `BL-20260509-hygiene-validation-grep-must-be-case-insensitive` — multi-file widened grep (AC-11).
  - `BL-20260508-post-commit-recapture-discipline` — live-evidence ordering (AC-3, AC-5).
  - `BL-20260507-tracked-doc-hygiene-after-substantive-work` — top-level doc edits out of scope.
  - `BL-20260510-plan-file-frozen-during-active-rlcr-session` — prior plan files frozen.
  - `BL-20260504-runbook-command-verification` — runbook fake-* command verification (AC-12 docs).
- Prior-art papers cited in the round summary (per AC-9 bucket rationale):
  - `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` — Smith-Eklund foundation; predicts the predictor-friendly bucket.
  - `references/papers/knowledge_base/active/2021__need_for_speed_…_nvarchsim.pdf` — NVArchSim microbench-anchored constants.
  - `references/papers/knowledge_base/active/2009__gpgpu_sim*.pdf`, `2020__accel_sim*.pdf`, `2022__gcom_*.pdf` — additional contextual prior art.

## Dependencies and Sequence

### Cross-Host Transfer Prelude (binding precondition for follow-on)

This plan is authored on the **RTX 3090 Ti (sm_86)**. The round's RLCR loop will execute on the same 3090 Ti host. AC-5.2-style follow-on (the matching sm_120 sweep on the **RTX 5060 laptop**) is QUEUED for a follow-on cross-host operator transfer per CLAUDE.md "Session Handoff" + the project's `HANDOFF.md`.

### Milestones

1. **M-A: Provenance Audit + shape_point Normalization** (closes AC-1, AC-2). Architecture-independent.
   - Phase A1: failing tests for `provenance_audit.py` (positive matched fixture; 10 negative mismatch modes per the expanded AC-1 list).
   - Phase A2: implementation of `tools/local_labels/provenance_audit.py` (~250 LOC stdlib-only) + integration into `sweep_repeatability_matrix.py`'s `--live` preflight gate; extension of `validate_measurement_manifest.py` with `shape_point_directory_name_drift` typed reason.

2. **M-B: HEAD-Coherent Retrace — Microbench (8 slugs)** (closes AC-3 microbench-side, AC-4 microbench-side). Operator stage; DEC-1 RESOLVED to Regenerate-at-HEAD.
   - Phase B1: rebuild tracer at HEAD via `NVCC_CCBIN=g++-14 make -C tools/nvbit_tracer/tracer_tool`.
   - Phase B2: build all 8 microbench binaries at HEAD via `make -C workloads/simbench_bw/l1/<slug>` (+ `NVCC='nvcc -ccbin=/usr/bin/g++-14'` if the host default differs).
   - Phase B3: regenerate the 8 NVBit traces at HEAD via `python3 workloads/simbench_bw/run_suite.py --mode trace --kernel <slug> --shape-point trace_00` per slug.
   - Phase B4: confirm `test_nvbit_trace_contract_live.py` admits 8/8 fresh microbench bundles.

3. **M-B': HEAD-Coherent Retrace — Activation (5 cells)** (closes AC-3 activation-side, AC-4 activation-side). Operator stage; DEC-2 RESOLVED to Microbench+activation, DEC-6 RESOLVED to Re-label-at-HEAD.
   - Phase B'1: build all 5 activation binaries at HEAD via `make -C workloads/portable_lcd/<slug>_sm80` (or the canonical portable_lcd build incantation; rebuild necessary because the existing labels are at `0c16e335...`).
   - Phase B'2: confirm/extend the orchestrator (run_suite.py or equivalent) to support NVBit-tracer entry for the portable_lcd activation slugs at the right shape_point (`timing_shape` for layernorm/rmsnorm/silu_timing/softmax_timing; `correctness_shape` for silu_correctness).
   - Phase B'3: capture the 5 activation NVBit traces at HEAD; emit per-shape sidecars carrying `source_commit == HEAD`, `binary_sha256 == sha256(fresh_activation_binary)`.
   - Phase B'4: confirm `test_nvbit_trace_contract_live.py` admits 5/5 fresh activation bundles.

4. **M-C: Full Sweep Matrix Documentation** (closes AC-12). Documentation-only; DEC-3 RESOLVED to Full matrix.
   - Phase C1: extend `tools/local_labels/SWEEP_RUNBOOK.md` with the full-matrix operator discipline section (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5` = 40 invocations per cell × 13 cells = 520 invocations, ~4.3 hrs operator wall-clock at ~30s/invocation).
   - Phase C2: AC-12 fake-* runbook-command-verification test admits the documented full-matrix command shape.

5. **M-D: Live NCU Sweep — 13-cell cohort** (closes AC-5). Operator stage; depends on M-A audit + M-B + M-B' retraces being green.
   - Phase D1: stage 13 measurement manifests under `data/local_labels/sweeps/<sweep_id>/manifests/` carrying `(slug, shape_point, binary_sha256, kernel_signature_sha256)` from the M-B + M-B' trace metadata.
   - Phase D2: run `provenance_audit.py` against the staged manifests; require `PASS` for all 13 candidates.
   - Phase D3: run the full matrix (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5`) via `sweep_repeatability_matrix.py --live --candidates …` for all 13 cells (~4.3 hrs operator wall-clock).
   - Phase D4: aggregate via `repeatability_aggregator.py`; admit cells passing `stddev_l2_sector ≤ 0.03`.
   - Phase D5: confirm `len([c for c in label_admitted.cells if c.arch == "sm_86" and c.slug in HEADLINE_COHORT_13 and c.source_commit == pre_sweep_HEAD and c.evidence_kind == "live"]) >= 5`. Aspirational: 13. If <5, the round closes at the AC-7 typed-unavailable negative path (negative-evidence outcome).

6. **M-E: Predictor Bundle Production at HEAD — 13 cells** (closes AC-6). Code stage; depends on M-B + M-B' retraces.
   - Phase E1: failing test for the verifier hook (admit 13 fresh bundles; reject circular/digest-drift/mpi-mismatch fixtures).
   - Phase E2: implementation of `tools/run_predictor_for_headline_cohort.py` orchestrator (or equivalent) that invokes `cache_predictor.predict_cache_hit_rate()` per cell and writes the bundle.
   - Phase E3: `verify_artifacts.py --results` admits the 13 bundles via `_verify_cache_predictor_run_summaries`.

7. **M-F: Comparison Runner End-to-End + Diagnostic Buckets + Supersession Bookkeeping** (closes AC-7, AC-9, AC-10). Code stage; depends on M-D + M-E.
   - Phase F1: failing tests for AC-7 (admitted joins ≥5; result_kind="admitted"; bootstrap CI populated) + AC-9 (per-bucket geomean per arch; static-source-scan lockstep on the per-cell bucket constant covering all 13 cells) + AC-10 (per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched >= 5 from the 5 historical activation cells at 0c16e335).
   - Phase F2: implementation of `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant (or `[predictor_design_coverage]` TOML section) covering all 13 cells with the 8-friendly + 5-hostile partition; per-bucket geomean computation in `comparison_runner.py`; round-summary text discipline pin for AC-10's supersession bookkeeping.
   - Phase F3: end-to-end run produces `comparison_bundle.json` with `result_kind="admitted"` for sm_86.

8. **M-G: Architecture-Coupling Discipline + Anti-Goals + Round Summary** (closes AC-8, AC-13, AC-11 final hygiene gate).
   - Phase G1: round summary records device, driver, CUDA, NVBit build, observed clocks/throttle state for the 3090 Ti per BL-20260508.
   - Phase G2: round summary asserts `Verified-live-sm_86` only; queues sm_120 follow-on with explicit cross-host operator handoff section.
   - Phase G3: round summary explicitly cites AC-10 supersession bookkeeping (5 historical activation cells at 0c16e335 in `comparison_runner_join_key_unmatched`).
   - Phase G4: confirm `pytest workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py -q` passes; confirm `tools/test_plan_token_scanner.py` returns `23 passed`; confirm widened workflow-marker grep returns no matches in implementation surfaces.

### Dependency Graph

- **M-A** has no in-loop dependencies; lands first as the WIRE prerequisite (without the audit, M-D's live sweep risks burning ~4.3 hrs of operator wall-clock on a no-join run).
- **M-B** (microbench retrace) has no in-loop dependencies (operator stage); lands in parallel with M-A and M-B'.
- **M-B'** (activation retrace) has no in-loop dependencies (operator stage); lands in parallel with M-A and M-B.
- **M-C** depends on M-A (the runbook documents the harness invocation).
- **M-D** depends on M-A (audit must pass) AND M-B (HEAD-coherent microbench traces) AND M-B' (HEAD-coherent activation traces) AND M-C (operator runbook must exist).
- **M-E** depends on M-B + M-B' (HEAD-coherent traces are the predictor input).
- **M-F** depends on M-D (admitted labels) AND M-E (predictor bundles).
- **M-G** depends on all prior milestones (round summary aggregates the evidence).

There are NO out-of-loop tooling dependencies. The cross-host transfer back to the 5060 laptop for AC-5.2-style follow-on is a project-discipline dependency (HANDOFF.md + CLAUDE.md "Session Handoff"), NOT a tooling dependency for this plan.

## Task Breakdown

Each task carries exactly one routing tag. The slice-by-AC structure produces ~14-18 commits.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1a | Failing tests for AC-1 + AC-2: `provenance_audit.py` accepts a matched fixture; rejects ALL documented mismatches (slug / binary_sha256 / source_commit / kernel_signature_sha256 / shape_point=trace_trace_00 / arch unrecognized / ncu_metric_version drift / staged manifest schema_version drift / staged manifest required field missing / predictor manifest_pairing_index incomplete); `validate_measurement_manifest.py` rejects `shape_point: "trace_trace_00"` with typed `shape_point_directory_name_drift`. Tests fail at this commit. | AC-1, AC-2, AC-11 | coding | - |
| task1b | Implementation: author `tools/local_labels/provenance_audit.py` (stdlib-only Python; reads predictor-side `trace_<shape>.json` and operator-staged measurement manifest; sources 4 staged join-key fields from the manifest, derives 3 fields from `git rev-parse HEAD` + `--arch` flag + `comparison_contract.toml`'s `[ncu_metric].ncu_metric_version`; computes 7-tuple equality; deterministic byte-identical output). Wire into `sweep_repeatability_matrix.py` `--live` preflight gate. Extend `validate_measurement_manifest.py` with the new typed reason. Tests from task1a turn green. | AC-1, AC-2 | coding | task1a |
| task2 | AC-12 documentation slice: extend `tools/local_labels/SWEEP_RUNBOOK.md` with the full-matrix operator discipline section per DEC-3 (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5` = 520 invocations × ~30s ≈ 4.3 hrs); add fake-NCU runbook-command-verification test admits the documented full-matrix command shape. | AC-12, AC-11 | coding | task1b |
| task3a | Failing test for AC-3 + AC-4: AC-3 fixture validator asserts each of the 13 traces' `source_commit == git_rev_parse(HEAD)` and `binary_sha256 == sha256(fresh_binary)`; AC-4 admits 13/13 fresh bundles via the existing trace-contract verifier. Test FAILS at this commit because the existing microbench traces are at historical commits `38a353d3` (7) + `5ea52f3a` (1), AND no activation NVBit traces exist at all in `workloads/simbench_bw/results/<activation_slug>_sm80/...`. The failure is the audit-gate signal that prerequisites are not yet met. | AC-3, AC-4, AC-11 | coding | task1b |
| task3b | M-B operator stage (microbench retrace): rebuild `tools/nvbit_tracer/tracer_tool/tracer_tool.so` at HEAD; build all 8 microbench binaries at HEAD; regenerate 8 NVBit traces at HEAD via `run_suite.py --mode trace --kernel <slug> --shape-point trace_00`; confirm `test_nvbit_trace_contract_live.py` admits all 8 fresh microbench bundles. Operator-evidence task; AFTER task3a's failing tests committed. | AC-3, AC-4 | coding | task3a |
| task3b' | M-B' operator stage (activation retrace + portable_lcd orchestrator wiring): build all 5 activation binaries at HEAD via `make -C workloads/portable_lcd/<slug>_sm80`; if `run_suite.py --mode trace` does not yet support portable_lcd activation slugs, extend it (or author a parallel orchestrator under `tools/local_labels/` since this preserves the local-label exception scope) to invoke the NVBit tracer against the portable_lcd activation binaries at the right shape_point (`timing_shape` for layernorm/rmsnorm/silu_timing/softmax_timing; `correctness_shape` for silu_correctness); capture the 5 activation NVBit traces at HEAD; confirm `test_nvbit_trace_contract_live.py` admits all 5 fresh activation bundles. Operator-evidence + light-code task; AFTER task3a. | AC-3, AC-4 | coding | task3a |
| task3c | Implementation: integrate the AC-3/AC-4 fixture validator into the round's pre-sweep gate; document in `SWEEP_RUNBOOK.md` that the validator must pass before live NCU launch. Tests from task3a turn green AFTER both task3b AND task3b' complete. | AC-3, AC-4 | coding | task3a, task3b, task3b' |
| task5 | M-D operator stage (live sweep, 13 cells, full matrix): stage 13 measurement manifests under `data/local_labels/sweeps/<sweep_id>/manifests/` (carrying `slug, shape_point, binary_sha256, kernel_signature_sha256` from the M-B + M-B' trace metadata); run `provenance_audit.py` (must `PASS` for all 13 candidates); run the FULL sweep matrix (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5` = 40 invocations × 13 cells = 520 invocations, ~4.3 hrs) via `sweep_repeatability_matrix.py --live --candidates …`; aggregate via `repeatability_aggregator.py`; confirm `len(HEAD-coherent admitted sm_86 cells) >= 5` (aspirational 13). If <5, the round closes at AC-7 typed-unavailable negative path. Operator-evidence task closing AC-5. | AC-5, AC-12 | coding | task1b, task2, task3c |
| task6a | Failing test for AC-6: predictor verifier hook `_verify_cache_predictor_run_summaries` admits 13 fresh bundles; rejects circular-input / digest-drift / manifest-pairing-index-mismatch fixtures. Test fails at this commit. | AC-6, AC-11 | coding | task3a |
| task6b | Implementation: author `tools/run_predictor_for_headline_cohort.py` (or equivalent) orchestrator that invokes `cache_predictor.predict_cache_hit_rate()` per cell (8 microbench at trace_00 + 5 activation at their respective shape_points) and writes per-cell predictor bundles. Tests from task6a turn green. | AC-6 | coding | task6a, task3b, task3b' |
| task7a | Failing tests for AC-7: comparison runner admits ≥5 HEAD-coherent joined cells; `result_kind="admitted"` for sm_86; per_arch_diagnostics.per_arch.sm_86.coverage_gate_verdict.meets_floor=true; bootstrap CI populated. Tests fail at this commit. | AC-7, AC-11 | coding | task1b |
| task7b | Implementation: author `tools/run_comparison_for_headline_cohort.py` (or equivalent) thin CLI wrapper invoking `comparison_runner.run_comparison()` with the post-AC-5 labels and post-AC-6 bundles. Tests from task7a turn green AFTER task5 + task6b complete. | AC-7 | coding | task5, task6b, task7a |
| task8a | Failing tests for AC-9 + AC-10: per-cell `predictor_design_coverage` enum; per-bucket geomean per arch (8 friendly + 5 hostile); `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant lockstep test (set-equality vs the 13 HEADLINE_COHORT_13 cells); per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched >= 5 from the 5 historical activation cells at 0c16e335 (supersession bookkeeping). Tests fail at this commit. | AC-9, AC-10, AC-11 | coding | task7b |
| task8b | Implementation: add `_PREDICTOR_DESIGN_COVERAGE_BY_CELL` constant (or `[predictor_design_coverage]` TOML section + ComparisonContract sub-config + loader) covering all 13 cells with the 8-friendly + 5-hostile partition; add per-bucket relative_error_stats computation in `comparison_runner.py`. AC-10 is round-summary-text + natural-runner-output discipline; no schema field added (the supersession bookkeeping surfaces as `comparison_runner_join_key_unmatched` in the existing per-arch diagnostics naturally). Tests from task8a turn green. | AC-9, AC-10 | coding | task8a |
| task9 | M-G round summary stage: round summary records device + driver + CUDA + NVBit + observed clocks for the 3090 Ti; asserts `Verified-live-sm_86` only; queues sm_120 follow-on with cross-host operator handoff section; explicitly cites AC-10 supersession bookkeeping (5 historical activation cells at 0c16e335 in comparison_runner_join_key_unmatched); confirms `test_no_root_or_ncu_static_guard.py` and `tools/test_plan_token_scanner.py` pass. | AC-8, AC-10, AC-11, AC-13 | coding | task1b, task2, task5, task7b, task8b |
| task10 | OPTIONAL Codex review pass on the comparison_bundle output: ask Codex to read the `comparison_bundle.json` artifact, validate the geomean computation against the contract, and surface any silent-fallback patterns or wording overclaims in the per-arch diagnostics output (especially the friendly-vs-hostile bucket geomeans and the supersession bookkeeping count). Output preserved as the round's external review. | AC-7, AC-9, AC-10 | analyze | task8b |

## Claude-Codex Deliberation

### Agreements

- The predictor↔label join produces ZERO cells today; this round's binding deliverable is the FIRST non-zero `comparison_bundle.json` with `result_kind="admitted"` for sm_86. (Source: draft + Codex first-pass alignment.)
- After DEC-2 + DEC-6 resolution, the headline cohort is **13 cells (8 memory-microbench + 5 activation)**, all HEAD-coherent. The 5 activation cells need NVBit traces captured at HEAD (M-B') alongside the 8 microbench retraces (M-B); the 5 historical activation labels at `0c16e335` are SUPERSEDED-by-history per AC-10.
- `shape_point` value is `trace_00`, NOT `trace_trace_00`; the directory naming convention `trace_<shape>` is the source of confusion. (Source: Codex first-pass technical gap; verified at `workloads/simbench_bw/results/working_set_capacity_sweep/rtx_3090_ti/sm_86/trace_trace_00.json`.)
- The 7-field comparison-runner join key includes `binary_sha256` and `source_commit`; the existing microbench traces are NOT at HEAD (7 at `38a353d3...`, 1 at `5ea52f3a...`; HEAD is `7b85b1f`); the existing activation labels are at `0c16e335...`, also not HEAD. A pre-live join-key audit + a HEAD-coherent retrace strategy (DEC-1 RESOLVED to Option B) is mandatory. (Source: Codex first-pass core risk; verified by reading the trace metadata files + label_admitted.json.)
- The Smith-Eklund LRU baseline does NOT model carveout / persisting-window / prefetch-hint / spill-to-local / cross-launch-cache-state policies; the 13 cells partition into **8 predictor_friendly cells** (3 microbench: working_set_capacity_sweep, l2_thrash_pointer_chase, dram_stream_stride_mix; 5 activation: layernorm/rmsnorm/silu_correctness/silu_timing/softmax) and **5 predictor_hostile cells** (5 microbench: carveout_sensitivity_probe, l2_persisting_window_sweep, l2_prefetch_hint_sweep, warm_cold_pair_relaunch, register_spill_local_mem). The first geomean WILL look heterogeneous; honest reporting via single-headline + diagnostic-bucket geomeans (DEC-4 RESOLVED to Option A).
- `evidence_kind: "live"` is already emitted by `repeatability_aggregator.py` (line 381); no aggregator code change needed.
- `algorithm_version: "smith_eklund_lru_v1"` stays unchanged in this round; algorithm improvements are queued for follow-on rounds (Alt-1 SASS-aware L1 bypass, Alt-2 calibration head, Alt-3 %clock64 latency reconstruction, Alt-4 eBPF + CUPTI bandwidth fusion, Alt-5 hardware-constant calibration suite — all documented in the draft).
- The cross-arch sm_120 follow-on is QUEUED, not in this round; per CLAUDE.md "Two-Device Local Development Model" the round closes at `Verified-live-sm_86` only.
- The `stddev_threshold = 0.03` immutable in `repeatability_aggregator.py` is unchanged; if the live sweep cannot admit ≥5 HEAD-coherent cells from the 13-candidate cohort, the round closes at the AC-7 typed-unavailable negative path (a documented round-CLOSED-WITH-NEGATIVE-EVIDENCE outcome that still produces a verifiable artifact).

### Resolved Disagreements

- **shape_point semantic ambiguity** (Codex first-pass core risk): the draft used `trace_trace_00` throughout; Codex correctly flagged this as the directory name (the join-key value is `trace_00`). Resolution: AC-2 pins `shape_point="trace_00"` across all surfaces; AC-1's audit rejects `trace_trace_00` as a `shape_point` value with typed `shape_point_directory_name_drift`.
- **Provenance policy implicit assumption** (Codex first-pass core risk): the draft assumed the existing traces could be labeled at HEAD; Codex correctly flagged that this would produce zero joinable cells because of `binary_sha256` + `source_commit` divergence. Resolution: AC-1 mandates the pre-live join-key audit; DEC-1 surfaces the user-facing choice between regenerate-at-HEAD vs label-historical-commits vs preflight-then-decide; AC-3 follows DEC-1 to either retrace at HEAD (default) or pin labels to historical commits.
- **Activation-cell handling** (Codex first-pass missing requirement, then DEC-2 + DEC-6 RESOLVED): the draft tentatively excluded activation cells; Codex first-pass surfaced the silent-exclusion bookkeeping risk (the runner only reports bundle-side join misses); Claude proposed an explicit exclusion artifact (AC-10 first version). The user RESOLVED DEC-2 to "Microbench + activation" (include activation in the headline cohort) and DEC-6 to "Re-label activation at HEAD" (capture activation NVBit + relabel at HEAD). Final resolution: AC-10 reframed as historical-supersession bookkeeping (the 5 historical activation labels at `0c16e335` REMAIN in `label_admitted.json` and surface naturally as `comparison_runner_join_key_unmatched` in `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts`); no new schema field added.
- **Wall-clock budget naivete** (Codex first-pass technical gap; DEC-3 RESOLVED): the draft proposed the full matrix (320 invocations, ~2.5 hrs for 8 cells); Codex recommended narrow-first (80 invocations, ~40 min) with widen-on-failure. The user RESOLVED DEC-3 to "Full matrix" — 13 cells × 40 invocations = 520 invocations, ~4.3 hrs — accepting the wall-clock cost in exchange for the most thorough characterization. AC-12 pins the full matrix; SWEEP_RUNBOOK.md documents.
- **Heterogeneous geomean publication risk** (Codex first-pass core risk; DEC-4 RESOLVED): the draft framed the first number as a single geomean with caveat; Codex offered four reporting options (single+caveat / single+buckets / two-geomeans / exclude-hostile). The user RESOLVED DEC-4 to single+buckets per AC-9; the bucket partition is contract-pinned (8 friendly + 5 hostile across the 13 cells); per-bucket relative_error_stats appear alongside the headline geomean.
- **Test-first for live evidence** (Codex first-pass missing requirement; DEC-5 RESOLVED): the draft did not specify how test-first applies to operator-evidence stages; Codex offered three options (commit runbook/schema/integration first / mocked failing tests for every artifact / post-run validation). The user RESOLVED DEC-5 to commit-runbook-first; AC-11 pins the test-first per-slice ladder with fake-NCU shims following the `_NCU_SHIM_TEMPLATE` precedent at `tools/local_labels/test_label_driver_integration.py`.

- **AC-1 negative-test list completeness** (Codex second-pass REQUIRED_CHANGE): the first-draft AC-1 only enumerated 5 mismatch modes (binary_sha256 / source_commit / kernel_signature_sha256 / shape_point / predictor-incomplete); Codex flagged this as incomplete for a 7-field audit. Resolution: AC-1 negative-test list expanded to 10 modes and a new "Field provenance" subsection clarifies that 4 join-key fields are STAGED in the operator manifest while 3 are DERIVED at audit time (source_commit from `git rev-parse HEAD`, arch from the `--arch` flag cross-checked against `arch_capabilities.json`, ncu_metric_version from `comparison_contract.toml`).

- **AC-3 / DEC-1 Option A operational complexity** (Codex second-pass REQUIRED_CHANGE): Codex flagged that `label_driver.sh` records `git rev-parse HEAD` at label time, so a single historical-label sweep cannot match all 8 traces (split across commits `38a353d3` and `5ea52f3a`). Resolution: DEC-1 documents that Option A requires either split-worktree double-batch OR partial retrace, strengthening the case for Option B (regenerate at HEAD).

- **AC-9 bucket rationale documentation** (Codex second-pass REQUIRED_CHANGE): the first-draft AC-9 documented bucket assignments without per-slug rationale and tentatively classified `warm_cold_pair_relaunch` as `predictor_friendly`. Codex flagged that relaunch cache state is a stateful policy behavior the LRU baseline does not model. Resolution: AC-9 now carries a per-slug rationale block in the plan (and requires it in-source via comments); `warm_cold_pair_relaunch` is reclassified to `predictor_hostile`. Final partition: 3 friendly (working_set_capacity_sweep, l2_thrash_pointer_chase, dram_stream_stride_mix) + 5 hostile (carveout_sensitivity_probe, l2_persisting_window_sweep, l2_prefetch_hint_sweep, warm_cold_pair_relaunch, register_spill_local_mem).

- **AC-10 schema strategy clarification** (Codex second-pass REQUIRED_CHANGE): the first-draft AC-10 was ambiguous about whether the new exclusion field would land inside per_arch_diagnostics or at the top level of the comparison_bundle. Codex correctly noted that adding a new token to `[per_arch_diagnostics_config].diagnostic_vocabulary` would require a coordinated 4-surface contract amendment + lockstep updates in `comparison_runner.py`'s `_EXCLUDED_DIAGNOSTIC_VALUES` constant. Resolution: AC-10's `excluded_label_cells_lacking_predictor_bundle` field lives at the TOP LEVEL of the comparison_bundle (additive optional; no schema_version bump); the per-arch `excluded_reason_counts` vocabulary stays the contract-pinned 3-token set unchanged.

- **AC-3/AC-4 task ordering** (Codex second-pass REQUIRED_CHANGE): the first-draft task table had `task3` (operator retrace) as no-deps + `task4a` (failing tests) depending on `task1b`. Codex correctly flagged that this lets `task3` land before `task4a`, making the failing tests pass-by-accident at commit time and violating BL-20260509-strict-admission-contract-first. Resolution: task table reordered as `task3a` (failing tests, depends on `task1b`) → `task3b` (operator retrace, depends on `task3a`) → `task3c` (impl wiring, depends on `task3b`); the failing tests are genuinely failing at commit time because the existing 8 traces are at historical commits `38a353d3` (7 traces) + `5ea52f3a` (1 trace), NOT current HEAD.

### Optional Improvements (Codex second-pass; non-blocking)

- AC-5/AC-7 closure ordering: the typed-unavailable negative-evidence closure (`headline_geomean_admitted_cell_count_below_threshold`) happens ONLY after the AC-12 full matrix has been completed (520 invocations across all 13 cells). Per DEC-3 the round commits to the full matrix unconditionally; if fewer than 5 cells admit despite the full matrix, the round closes at the typed-unavailable negative path with the widened-evidence verdict already documented.
- Predeclared rescue cohort (NOT in this round; documented for follow-on): `carveout_sensitivity_probe`, `l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, and `register_spill_local_mem` already have `trace_50` and/or `trace_75` shape variants in addition to `trace_00` (per `workloads/simbench_bw/results/carveout_sensitivity_probe/rtx_3090_ti/sm_86/trace_trace_{00,50,75}/`). A future round can expand the headline cohort by labeling these additional shape variants (3-4× more cells per slug) to reduce per-cell variance and lift the publication-grade floor (10/arch). Not in this round to keep DEC-2 narrow.

### Convergence Status

- Final Status: `converged`. All 6 user decisions (DEC-1 through DEC-6) RESOLVED via AskUserQuestion; all Codex REQUIRED_CHANGES from both passes incorporated. Two Codex passes executed:
  - **First-pass (planning analysis)**: surfaced 6 CORE_RISKS, 8 MISSING_REQUIREMENTS, 6 TECHNICAL_GAPS, 5 ALTERNATIVE_DIRECTIONS, 5 QUESTIONS_FOR_USER, and 10 CANDIDATE_CRITERIA. The plan's AC-1 through AC-13 incorporate every CORE_RISK, MISSING_REQUIREMENT, TECHNICAL_GAP, and CANDIDATE_CRITERIA item.
  - **Second-pass (reasonability review)**: surfaced 5 REQUIRED_CHANGES — all incorporated:
    1. AC-1 negative-test list expanded from 5 to 10 mismatch modes (added slug drift, arch unrecognized, ncu_metric_version drift, staged-manifest schema_version drift, manifest required-field missing); added the explicit "field provenance" subsection clarifying which 4 fields are staged in the manifest vs which 3 are derived at audit time.
    2. AC-3 / DEC-1 Option A operational complexity documented: `label_driver.sh` records `git rev-parse HEAD` at label time, so a single historical-label sweep cannot match all 8 traces (split across commits `38a353d3` + `5ea52f3a`); Option A requires split-worktree double-batch OR partial retrace.
    3. AC-9 bucket rationale moved into the plan body with per-slug justification; `warm_cold_pair_relaunch` reclassified from `predictor_friendly` → `predictor_hostile` (relaunch cache state is a stateful policy behavior the LRU baseline does not model). Final partition: 3 friendly + 5 hostile.
    4. AC-10 schema strategy clarified: the `excluded_label_cells_lacking_predictor_bundle` field lives at the TOP LEVEL of the comparison_bundle (additive optional, no schema_version bump); the per-arch `excluded_reason_counts` 3-token vocabulary is contract-pinned and stays unchanged (avoiding a coordinated 4-surface contract amendment that would expand round scope).
    5. AC-3/AC-4 task ordering reordered: `task3a` (failing tests) lands BEFORE `task3b` (operator retrace); the failing tests are genuinely failing at commit time because the existing 8 traces are at historical commits, not HEAD.
  - **All six user decisions RESOLVED** (DEC-1 through DEC-6) via AskUserQuestion in the gen-plan Phase 6:
    - **DEC-1: Regenerate at HEAD (Option B)** — DEC-1 + DEC-6 jointly produce the HEAD-coherent 13-cell cohort.
    - **DEC-2: Microbench + activation (13-cell cohort)** — expanded headline cohort beyond the Claude-default microbench-only.
    - **DEC-3: Full sweep matrix** (`warmup={0,1,5,10} × cache_control={all,none} × repeats=5`, 520 invocations, ~4.3 hrs) — most thorough characterization.
    - **DEC-4: Single geomean + diagnostic buckets (8 friendly + 5 hostile)** — honest baseline + diagnostic explanation.
    - **DEC-5: Commit runbook + schema + integration first (Option A)** — test-first per slice with fake-NCU shims.
    - **DEC-6: Re-label activation at HEAD** — 5 activation cells get HEAD-coherent labels lockstep with HEAD-coherent NVBit traces; the historical 5 labels at 0c16e335 are SUPERSEDED-by-history per AC-10's bookkeeping.

## Pending User Decisions

- **DEC-1: Provenance policy for closing the predictor↔label join.**
  - Claude Position: Option B — regenerate the 8 NVBit traces at HEAD before labeling.
  - Codex Position: Option C — preflight then decide.
  - Tradeoff Summary: Option A is operationally complex (split-worktree double-batch OR partial retrace); Option B is the cleanest provenance story but adds an extra retrace step (~1 hr operator wall-clock); Option C is the safest meta-choice but adds a layer of decision indirection.
  - Decision Status: **`RESOLVED: Option B (Regenerate at HEAD)`** — user accepted Claude's recommendation. AC-3, AC-4, AC-5 all pin to HEAD-coherent retrace. M-B and M-B' regenerate the 8 microbench + 5 activation NVBit traces at HEAD before the live NCU sweep.

- **DEC-2: Headline cohort scope.**
  - Claude Position: Option A — microbench-only sm_86 cells (8 candidates).
  - Codex Position: agrees Option A.
  - Tradeoff Summary: Option A is bounded scope; Option B (microbench + activation NVBit captures) doubles the operator stage work; Option C (runner-implicit exclusion) is silent and review-hostile.
  - Decision Status: **`RESOLVED: Option B (Microbench + activation)`** — user OVERRODE Claude's recommendation in favor of the larger 13-cell cohort. The round now captures NVBit traces for both microbench (8) and activation (5) at HEAD; AC-5 admission floor is 5/arch from a 13-candidate pool; AC-9 bucket partition expanded to 8 friendly (3 microbench + 5 activation) + 5 hostile (5 microbench).

- **DEC-3: Sweep matrix policy.**
  - Claude Position: Option C — narrow-first with operator-driven widen-on-failure (~40 min).
  - Codex Position: agrees Option C.
  - Tradeoff Summary: Option A (narrow only) risks AC-5 floor failure; Option B (full matrix always, ~2.5 hrs for 8 cells / ~4.3 hrs for 13 cells) is most thorough; Option C is adaptive.
  - Decision Status: **`RESOLVED: Option B (Full matrix)`** — user OVERRODE Claude's recommendation in favor of the most thorough characterization. 13 cells × 4 warmup × 2 cache_control × 5 repeats = 520 NCU invocations × ~30s ≈ 4.3 hrs operator wall-clock. AC-12 pins the full matrix as the round's canonical sweep shape; SWEEP_RUNBOOK.md documents.

- **DEC-4: Predictor-hostile probe reporting policy.**
  - Claude Position: Option A — single heterogeneous geomean + diagnostic per-bucket geomeans (predictor_friendly / predictor_hostile).
  - Codex Position: agrees Option A.
  - Tradeoff Summary: Option A preserves honesty; Option B (two official geomeans) introduces editorial choice in the partition itself; Option C (exclude hostile) is cherry-picking and review-hostile.
  - Decision Status: **`RESOLVED: Option A (Single geomean + diagnostic buckets)`** — user accepted Claude's recommendation. AC-9 pins the bucket partition (8 friendly + 5 hostile); per-bucket relative_error_stats appear alongside the headline geomean.

- **DEC-5: Test-first discipline shape for live-evidence stages.**
  - Claude Position: Option A — commit runbook + schema + integration expectations BEFORE live sweep.
  - Codex Position: agrees Option A.
  - Tradeoff Summary: Option A locks the contract before evidence; Option B (mocked failing tests for every artifact) is more rigorous but expands code surface; Option C (post-run validation) is least disciplined.
  - Decision Status: **`RESOLVED: Option A (Commit runbook + schema + integration first)`** — user accepted Claude's recommendation. AC-11 pins the test-first per-slice ladder; fake-NCU shims follow the `_NCU_SHIM_TEMPLATE` precedent at `tools/local_labels/test_label_driver_integration.py`.

- **DEC-6: Activation-cell provenance interaction with DEC-1 + DEC-2.**
  - Claude Position: Option A — re-label activation at HEAD (capture activation NVBit traces at HEAD; re-label all 5 activation cells at HEAD with the same sweep matrix as microbench; ~1.5hr added to the operator label sweep; fully HEAD-coherent provenance for all 13 cells).
  - Codex Position: implicit agreement (the convergence-round critique drove the DEC-6 question itself; Codex's recommendation favored a single HEAD-coherent provenance over hybrid HEAD/historical paths).
  - Tradeoff Summary: Option A is the cleanest provenance story but adds operator wall-clock; Option B (trace activation at 0c16e335) keeps existing labels but creates a 2-commit cohort that conflicts with DEC-1's clean-HEAD intent; Option C (hybrid HEAD/historical) is operationally complex.
  - Decision Status: **`RESOLVED: Option A (Re-label activation at HEAD)`** — user accepted Claude's recommendation. The 5 activation cells get HEAD-coherent NVBit traces (M-B') AND HEAD-coherent NCU labels (M-D phase D3 covers all 13 cells in one matrix run). The historical 5 activation labels at `0c16e335` REMAIN in `label_admitted.json` as SUPERSEDED-by-history; AC-10 pins the supersession bookkeeping that surfaces them naturally as `comparison_runner_join_key_unmatched` in `per_arch_diagnostics.per_arch.sm_86.excluded_reason_counts`.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "Round N", "M-A", "M-B", "DEC-N", "Plan line N", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- The widened workflow-marker grep `grep -RInE 'DEC-|[Pp]lan line|M1 plan|M2 plan|upper-bound|lower-bound scope' tools/local_labels/ workloads/simbench_bw/` MUST return no matches in implementation surfaces (subject to the genuinely-domain `[hit_rate_lower, hit_rate_upper]` exception).
- The `tools/test_plan_token_scanner.py` 23-passing scanner gate enforces the regex-based vocabulary (`AC-\d+`, `Milestone`, `Phase`, `Step`, `Round \d+`, `M2 plan`).
- Together both gates cover the full plan-token vocabulary; both must be green after every commit.

--- Original Design Draft Start ---

# First Headline Geomean via Memory-Microbench Label Sweep

## Original Idea

Goal: Improve end-to-end accuracy of the non-root cache-hit-rate predictor against NCU ground truth on the local RTX 3090 Ti (sm_86), with cross-arch generalization to the RTX 5060 laptop (sm_120) as a binding discipline. Long-term goal: a non-root toolchain (no NCU, no nvidia-smi clock control, no privileged CUPTI) that recovers wall-clock per kernel and L1/L2 cache-hit ratio with extremely low geomean relative error against NCU labels.

CRITICAL FINDING from repo audit (please verify): the comparison runner harness (workloads/simbench_bw/comparison_runner.py) and per-arch diagnostics layer have landed; the local-label sweep on the 3090 Ti admitted 5 sm_86 cells (layernorm_fwd_bf16_sm80/timing_shape, rmsnorm_fwd_bf16_sm80/timing_shape, silu_fwd_bf16_sm80/{correctness,timing}_shape, softmax_safe_bf16_sm80/timing_shape — see data/local_labels/label_admitted.json). But there are ZERO matching NVBit predictor inputs — workloads/simbench_bw/results/<slug>/... does not exist for any of those slugs, and workloads/portable_lcd/<slug>/traces/ only carries gemm_hmma_bf16_sm80. The "headline geomean predictor-vs-NCU relative error" cannot be computed today because no admitted predictor bundle joins to any admitted label cell on the 7-field key. The whole accuracy pipeline therefore has zero joinable cells; AC-5 admission floor in comparison_contract.toml fires `headline_geomean_admitted_cell_count_below_threshold`.

Active constraints (CLAUDE.md): NO NCU/Nsight Compute or `.ncu-rep` artifacts on cloud/server GPUs (rule absolute); local-label exception is bounded to tools/local_labels/ + data/local_labels/; NO nvidia-smi clock locking, persistence, MIG, power, compute-mode mutation; NO privileged CUPTI (Events, Metrics, Range Profiling, PM Sampling, PerfWorks) — CUPTI Activity only; NO host-driver work; NO simulator revival; cross-arch must be tested on BOTH sm_86 + sm_120 — never claim Cross-arch-verified from a single arch.

Predictor today (workloads/simbench_bw/cache_predictor.py): Smith-Eklund LRU stack-distance over per-launch reuse-distance histograms recomputed from dynamic_trace.pb / sector_stream sidecars via the Fenwick-tree tracker at tools/hrd_profiler/reuse_distance_tracker.py. Fully-associative LRU approximation. L2-only sector basis (32B). No L1 modeling. No set-associative refinement (RRIP/DRRIP/ATD all deferred behind ablation gate). No warp-aware contention or bank-conflict modeling. No L2 carveout / persisting-window awareness. Per-launch hit rate averaged, bootstrap CI launch unit, 2000 resamples.

Comparison contract (workloads/simbench_bw/comparison_contract.toml): NCU metric `lts__t_sector_hit_rate.pct`, ground truth fraction normalization, 7-field join key (slug, shape_point, binary_sha256, source_commit, kernel_signature_sha256, arch, ncu_metric_version), error formula `|pred-gt|/max(gt, 0.01)`, log-space geomean, smoke floor 5/arch, publication floor 10/arch.

Excluded labels: `gemm_hmma_bf16_sm80/correctness_shape` sm_86 (stddev 0.14 > 0.03 threshold) and `fp32_fma_dep_chain/trace_00` sm_120 (stddev 0.06 > 0.03). The 0.03 stddev_threshold in repeatability_aggregator.py cannot be loosened (static-source-scan test enforces this).

Available evidence-side surfaces (committed or buildable on this 3090 Ti host today): NVBit tracer (tools/nvbit_tracer/ — built dynamically; emits dynamic_trace.pb + sector_stream + threadblocks); CUPTI Activity collector library (workloads/simbench_bw/libsimbench_cupti_activity_collector.so); preflight + read-only NVML covariates (clock/temp/throttle); HRD profiler with 34-bucket reuse-distance partition; sector_stream and divergence_proxies sidecars; static SASS (cuobjdump/nvdisasm); inline PTX %clock64/%globaltimer phase probes; perturbation harness (cache_modifier_cs across Ampere LDG.E.CA/CG/CS + Blackwell LDG.E.STRONG.SM/STRONG.GPU/EF). NCU is allowed locally inside tools/local_labels/ for label collection only.

Brainstorming question: Identify the highest-leverage moves to actually CLOSE the accuracy gap to <10% (and ideally <5%) geomean relative error against NCU L2 sector hit rate on the 3090 Ti, given that the binding constraint is cross-arch decoupling (sm_120 must also work). Concretely:

1. SHOULD the next round prioritize CLOSING THE OVERLAP (i.e. produce NVBit traces for the 5 labeled portable_lcd kernels so the comparison can actually run end-to-end and we get a first headline number) BEFORE any predictor-side modeling work? Or invert — collect NCU labels for the simbench_bw memory-microbench slugs we already have NVBit traces for (working_set_capacity_sweep, l2_thrash_pointer_chase, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep, register_spill_local_mem) — these were designed as memory-behavior probes and would be much more diverse hit-rate distributions than the activation kernels?

2. Once joinable cells exist, what's the predictor-side improvement order? Candidates from CLAUDE.md / first plan: per-record L1 bypass routing (Ampere LDG.E.CA/CG vs Blackwell LDG.E.STRONG.SM/GPU/EF — already detected by tools/perturbation_harness/sass_confirm.py); set-associative refinement (RRIP/DRRIP/ATD); L2 carveout/persisting-window awareness; warp-aware contention / merging at the L1tex/L2 boundary; data-only L1 bypass for stores; sector-fill amplification (a 1B request can pull a 32B sector); mishandling of atomics; per-PC reuse pattern features instead of bulk LRU.

3. ARE THERE FUNDAMENTALLY DIFFERENT MEASUREMENT SURFACES that could materially improve accuracy without violating rules? The user asked about eBPF specifically. Consider: (a) eBPF on the userspace-driver side (uprobes on libcuda.so) to capture allocator and stream-ordered events without driver root; (b) eBPF kprobes (these need root and would violate rule 4); (c) PCIe BAR1 / BAR0 read-only sampling via /dev/nvidia*; (d) NVBit memory-trace at higher fidelity (per-thread cache modifier passed through inject_funcs.cu); (e) fork-and-replay where we re-run small kernels with NVBit-injected `%clock64` phase probes between every memory operation; (f) using Nsight Systems (allowed) for per-kernel CUPTI Activity timeline and joining its output offline; (g) static analysis predictor — use cuobjdump/nvdisasm to extract per-PC memory access types + footprint statically, then combine with a cheap bytecode-execution model rather than running the kernel under NVBit at all.

4. What is the SMALLEST defensible step that PRODUCES A REAL FIRST NUMBER on the 3090 Ti this round? (Goal is to land a Verified-live-sm_86 evidence claim WITH a real geomean relative error number, however imperfect, so we have a baseline to improve from.)

Please return a draft idea (with prior-art-grounded reasoning where relevant — Smith-Eklund 2014, NVArchSim HPCA 2021, Accel-Sim MICRO 2020, GPGPU-Sim, GCom 2022) that the user can refine with /humanize:gen-plan. Prefer concrete file paths and AC-style criteria over vague directions.

## Primary Direction: First Number via Microbench Label Sweep

### Rationale

Closes the predictor↔label overlap by running the local NCU label sweep on the simbench_bw memory-microbench slugs (working_set_capacity_sweep, l2_thrash_pointer_chase, dram_stream_stride_mix, carveout_sensitivity_probe, warm_cold_pair_relaunch, l2_persisting_window_sweep, l2_prefetch_hint_sweep, register_spill_local_mem) where NVBit traces already exist on the 3090 Ti host, producing the first end-to-end Verified-live-sm_86 geomean relative-error number without any predictor-side or measurement-side change. Treats the gap as a coverage problem, not a modeling problem — directly addresses sub-question 4 of the brainstorm ("smallest defensible step that produces a REAL FIRST NUMBER").

### Approach Summary

Execute the local-label sweep harness (`tools/local_labels/sweep_repeatability_matrix.py`) on the 8 memory-microbench slugs targeting `sm_86` shape point `trace_trace_00` on the 3090 Ti host. The sweep operates end-to-end:

1. Build each slug's binary via `make -C workloads/simbench_bw/l1/<slug>` (all 8 expose `all: run-bin` per existing Makefiles).
2. Stage operator-supplied measurement manifests carrying `kernel_signature_sha256` per the SWEEP_RUNBOOK.md protocol; validate via `tools/local_labels/validate_measurement_manifest.py`.
3. Invoke `sweep_repeatability_matrix.py --live --candidates …` with matrix axes `warmup_count ∈ {0,1,5,10}`, `cache_control ∈ {all, none}`, `repeats ≥ 5` to run `label_driver.sh` × NCU per `(cell, config, repeat)` tuple.
4. Aggregate per-cell sidecars via `repeatability_aggregator.py`; cells with `stddev_l2_sector ≤ 0.03` admit into `data/local_labels/label_admitted.json` (schema_version 3, `evidence_kind: live`).
5. Run the existing predictor `cache_predictor.py` against the matching dynamic_trace.pb / sector_stream sidecars under `workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_trace_00/` to emit `cache_predictor_run_*.json` bundles whose 7-field `manifest_pairing_index` joins to the admitted label cells.
6. Run `comparison_runner.py` to compute the FIRST headline geomean predictor-vs-NCU relative error per arch (along with the per_arch_diagnostics block already wired in), bootstrap CI per the 2000-resample contract, and emit a `comparison_bundle.json` whose `result_kind == "admitted"` instead of `typed_unavailable: headline_geomean_admitted_cell_count_below_threshold`.

The accuracy claim from this round is **a baseline number**, not a publication-grade result — the 8 memory-microbench slugs are a deliberately heterogeneous calibration target (some predictor-friendly: working_set_capacity_sweep maps directly onto the Smith-Eklund 2014 stack-distance modeled-hit rule; some predictor-hostile: l2_persisting_window_sweep and carveout_sensitivity_probe exercise policies the fully-associative LRU approximation does not model). The expected outcome is a heterogeneous geomean (heuristic estimate 20–50% relative error before any model improvement). The value is that this number BECOMES THE FLOOR every subsequent improvement round measures itself against. Every alternative direction below (SASS-aware L1 bypass, calibration head, hardware-constant calibration, %clock64 reconstruction, eBPF fusion) consumes this baseline as the comparison anchor.

Cross-arch discipline: the round's lower bound is `Verified-live-sm_86` only. AC-5.2-style follow-on (the same sweep on the 5060 laptop sm_120 host) is queued as a separate cross-host operator task to lift to `Cross-arch-verified` per CLAUDE.md "Two-Device Local Development Model."

Affected components (no code changes required to any of them):

- `tools/local_labels/label_driver.sh` — invoked per `(cell, config, repeat)`.
- `tools/local_labels/sweep_repeatability_matrix.py` — operator harness; existing `--candidates` flag accepts the 8-slug list.
- `tools/local_labels/repeatability_aggregator.py` — per-config sidecar aggregation; `stddev_threshold = 0.03` unchanged.
- `workloads/simbench_bw/cache_predictor.py` — existing Smith-Eklund predictor; consumed unchanged.
- `workloads/simbench_bw/comparison_runner.py` — already library-first and glob-discovers predictor bundles (line 38, `--predictor-bundles-glob`).
- `workloads/simbench_bw/l1/<slug>/Makefile` — 8 buildable targets.
- `data/local_labels/label_admitted.json` — target output; merge new sm_86 cells via `merge_admitted_cells.py`.

### Objective Evidence

- NVBit traces verified for all 8 target slugs on the 3090 Ti:
  - `workloads/simbench_bw/results/working_set_capacity_sweep/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/l2_thrash_pointer_chase/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/carveout_sensitivity_probe/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/warm_cold_pair_relaunch/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/l2_persisting_window_sweep/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/l2_prefetch_hint_sweep/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
  - `workloads/simbench_bw/results/register_spill_local_mem/rtx_3090_ti/sm_86/trace_trace_00/dynamic_trace.pb`
- All 8 slugs have buildable Makefiles at `workloads/simbench_bw/l1/<slug>/Makefile` with `all: run-bin` targets; all are present in the active roster `workloads/simbench_bw/roster.toml`.
- Label-collection infrastructure proven: `tools/local_labels/label_driver.sh` (818 LOC, full provenance capture per AC-2/R4/R5); `tools/local_labels/sweep_repeatability_matrix.py` (matrix enumeration with dry-run/live modes, verdict-table CSV/JSON emission, deterministic byte-identical re-run); `tools/local_labels/SWEEP_RUNBOOK.md` operator workflow with dry-run → live → aggregation → merge → verification steps.
- ZERO of the 8 target memory-microbench slugs appear in any existing sweep verdict table (`grep -RInE 'working_set_capacity_sweep|l2_thrash_pointer_chase|dram_stream_stride_mix|carveout_sensitivity_probe|warm_cold_pair_relaunch|l2_persisting_window_sweep|l2_prefetch_hint_sweep|register_spill_local_mem' data/local_labels/sweeps/*/verdict-table.csv` returns empty) — there is genuine new evidence to collect, not a re-run of existing work.
- Comparison runner consumes new slugs without code changes: `workloads/simbench_bw/comparison_runner.py` lines 692–720 (`_bundle_join_key()` / `_cell_join_key()`) join purely on the contract-pinned 7-field key; bundle discovery is glob-driven (line 38, `--predictor-bundles-glob`).
- Repeatability baseline is favorable: prior sweep `data/local_labels/sweeps/sweep_20260509T144232Z_…` admitted layernorm and softmax with `stddev_l2_sector ≤ 0.0153` under `(warmup=5, cache_control=none, repeats=5)`; the memory-microbench workloads (deterministic patterns by design) are likely to admit similarly.
- Measurement-manifest protocol fully supported: `label_driver.sh --measurement-manifest <path>` (line 166); `sweep_repeatability_matrix.py --measurement-manifests REQUIRED in --live` (lines 68–73); `validate_measurement_manifest.py` provides bidirectional `--binary` ↔ manifest coverage check.
- Prior art directly relevant:
  - Smith-Eklund 2014 (`references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`) — the predictor's foundational algorithm; working_set_capacity_sweep is a direct in-domain test for it.
  - NVArchSim HPCA 2021 (`references/papers/knowledge_base/active/2021__need_for_speed_…_nvarchsim.pdf`) — uses microbench-anchored hardware constants; l2_persisting_window_sweep / l2_prefetch_hint_sweep / carveout_sensitivity_probe map directly to NVArchSim microbench targets.
  - Volkov-Demmel 2008 ("Benchmarking GPUs to Tune Dense Linear Algebra," HPCA) — working_set_capacity_sweep is a Volkov-Demmel-style L1/L2 capacity sweep.
- Cross-arch readiness: `tools/arch_capabilities.json` declares both `sm_86` and `sm_120-laptop` as supported; the same 8-slug sweep can be replayed on the 5060 laptop in a follow-on AC-5.2 cross-host operator task per CLAUDE.md's two-device discipline.

### Known Risks

- **Binary build fragility on the 3090 Ti host.** Eight Makefiles must rebuild cleanly under the host's `g++-14` + `nvcc 13.2`. Any per-slug build failure blocks that cell. Mitigation: dry-run each Makefile before staging measurement manifests; use the `NVCC='nvcc -ccbin=/usr/bin/g++-14'` override per HANDOFF.md.
- **Shape-point semantic risk.** The 8 slugs use `trace_trace_00` — a custom shape — instead of the standard `correctness_shape`/`timing_shape`. `label_driver.sh` enforces strict shape-point binding at lines 298–301; mismatches fail closed. Mitigation: verify each `main.cu` parses `--shape-point trace_trace_00` correctly via dry-run.
- **stddev admission gate inflexibility.** The `stddev_l2_sector ≤ 0.03` threshold is pinned in `repeatability_aggregator.py` and enforced by a static-source-scan test; if any slug's intrinsic variance exceeds 0.03 across all sweep configs, that cell never admits. Mitigation: start with the wider matrix `warmup ∈ {0,1,5,10} × cache_control ∈ {all,none}`; only declare per-slug failure after exhausting all axes; explicitly tolerate up to 3 of 8 failing as long as ≥5 admit (smoke floor).
- **Heterogeneous baseline accuracy is publication-hostile.** The first geomean number will mix predictor-friendly slugs (working_set_capacity_sweep) with predictor-hostile slugs (l2_persisting_window_sweep, carveout_sensitivity_probe). The published geomean may be 30–50%, which looks bad in isolation. Mitigation: report per-slug relative errors alongside the geomean (the per_arch_diagnostics block already wires per-slug records into the comparison_bundle); frame the round explicitly as a baseline, not a publication number.
- **Join-key drift between trace-time and label-time `source_commit`.** The 7-field join key requires exact match on `source_commit`. If labels are collected at one commit and traces at another, no joins occur. Mitigation: re-capture NVBit traces at the same HEAD as the label sweep (or confirm the existing trace's `source_commit` matches the planned label-sweep HEAD) before launching the live NCU sweep.
- **Wall-clock budget for operator.** 8 binaries × 4 warmup levels × 2 cache_control levels × 5 repeats = 320 NCU invocations; at ~30s/invocation including warmup, that is ~2.5 hours of GPU-time. Mitigation: dry-run validates the matrix before live execution; the operator can split into two ~80-minute sessions.
- **Cross-arch follow-on is gated, not free.** The matching sm_120 sweep on the 5060 laptop is a separate cross-host operator task; this round can lift only to `Verified-live-sm_86`, not `Cross-arch-verified`. The user's binding cross-arch discipline is preserved by explicitly queuing AC-5.2-style follow-on.

## Alternative Directions Considered

### Alt-1: SASS-Aware Per-Record L1 Bypass Predictor

- Gist: Implement AC-11 of the first cache-predictor plan — extend `cache_predictor.py` with a new `tools/cache_predictor/l1_bypass_classifier.py` module that consumes per-record `sector_stream` entries plus the per-PC SASS classifier output (cross-arch regex at `tools/perturbation_harness/sass_confirm.py:45-53` covering Ampere `LDG.E.CA/CG/CS` and Blackwell `LDG.E.STRONG.SM/STRONG.GPU/EF` per BL-20260505) and routes each access to L1 vs L2 individually. The Smith-Eklund modeled-hit rule (`rd < capacity_bytes // 32`) is branched on per-record bypass status. Bumps `algorithm_version` to `smith_eklund_lru_l1bypass_v1` (no `instruction.proto` modification per AC-11 negative test). Prior art: NVArchSim HPCA 2021 explicitly models L1 cache modifiers; Accel-Sim MICRO 2020 has a per-instruction memory hierarchy router; GPGPU-Sim MICRO 2009 introduced the L1/L2/global path distinction.
- Objective Evidence:
  - SASS classifier present at `tools/perturbation_harness/sass_confirm.py:45-53` (Ampere + Blackwell cross-arch regex; 259 LOC; full cuobjdump/nvdisasm extraction pipeline).
  - Per-record PC tracking already in place: `tools/sector_stream/consume_trace.py:228-232` collects per-instruction `vpc` in provenance alongside every sector_stream record.
  - Modeled-hit rule at `tools/hrd_profiler/observation_builder.py:287` (`sector_rd < capacity_bytes // 32`) operates per access and is a simple branch point.
  - `cache_predictor.py` `_aggregate_launches_from_observations()` (lines 214–276) walks per-access observations; the hit/miss/cold aggregation can accept an optional per-record bypass flag.
  - Schema `workloads/simbench_bw/schemas/cache_predictor.schema.json` has an optional `evidence` block; no schema bump required for the per-record routing decision.
  - `instruction.proto` carries `pc` and `vpc` (lines 99–109) but no cache-modifier; AC-11's "instruction.proto wire format is NOT modified" constraint is honored by the post-hoc PC→SASS lookup approach.
- Why not primary: Requires joinable cells to measure improvement, and the project has zero joinable cells today. Should land AFTER Direction 1 establishes the baseline geomean against which "did L1 bypass move the needle?" can be answered.

### Alt-2: Post-Hoc Calibration Head

- Gist: Treat the existing Smith-Eklund prediction as one input feature and fit a small (≤3-parameter) monotone calibration head — e.g., Pool-Adjacent-Violators isotonic regression or a 2-parameter monotone sigmoid — over `(predicted_hit_rate, sector_count, divergence_proxies, NVML clock covariate)`. Pure-Python (no sklearn/torch/tensorflow per project policy); reuses the 3×3 Gaussian-elimination pattern at `workloads/simbench_bw/nvml_covariate_join.py:147-175`; resamples residuals via the existing 2000-resample bootstrap CI machinery. Register as a new `algorithm_version = "smith_eklund_lru_v1_calibrated_v1"`. Prior art: GCom 2022 (probabilistic GPU performance models cite calibration as a post-hoc refinement).
- Objective Evidence:
  - 45/16/16 calibration/validation/holdout split confirmed at `workloads/simbench_bw/splits.toml`.
  - Bootstrap CI machinery at `cache_predictor.py:177-211` (`BOOTSTRAP_RESAMPLES=2000`, launch-unit resampling) and `tools/hrd_profiler/profiler.py:40-66`.
  - Pure-Python regression precedent at `workloads/simbench_bw/nvml_covariate_join.py:147-175` (`_solve_3x3` with partial pivoting).
  - `algorithm_version` already an open string field at `cache_predictor.py:46`; schema admits new variant.
  - Per-arch model variant supported by the 7-field join key (already includes `arch`).
- Why not primary: Small-sample overfitting risk with only 5 admitted cells per arch; head should be fit ONLY after Direction 1 expands the joinable set (combining the 8 microbench cells with the 5 activation cells gives ~13 cells/arch — enough to fit 1–2 parameters per arch with leave-one-out CV).

### Alt-3: %clock64 Hit/Miss Latency Reconstruction

- Gist: Add NVBit-injected per-LD/ST `%clock64` phase probes (extending the existing per-phase probe at `tools/nvbit_tracer/tracer_tool/tracer_tool.cu:225, 443, 1120-1136` and `inject_funcs.cu:159-213`) to capture per-access cycle deltas intra-SM, then label each access as L1-hit / L2-hit / DRAM-miss directly via Volkov-Demmel-style calibrated latency thresholds — bypassing the LRU stack-distance abstraction entirely. Emit a new `algorithm_version = "latency_classification_v1"` predictor variant that joins on the same key. Prior art: NVBit MICRO 2019 (clock-aware injection); Mei-Chu 2017 ("Dissecting GPU memory hierarchy through microbenchmarking"); Volkov-Demmel 2008.
- Objective Evidence:
  - Existing per-phase `%clock64` probe at `inject_funcs.cu:159-213` uses `asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock64));` — directly reusable.
  - Phase-injection orchestration at `workloads/simbench_bw/phase_inject_orchestrator.py:71-179` already pairs no-injection / with-injection runs with admission-threshold overhead validation.
  - Phase clock decode + aggregation at `workloads/simbench_bw/phase_cycle_ledger_aggregator.py:96-113, 338-412` (`PhaseClockRecord` dataclass; `(launch_index, cta_id, sm_id, warpid, phase_id)` pairing).
  - Memory-op classifier at `tracer_tool.cu:101-147` already filters LD/ST opcodes (GLOBAL, LOCAL, SHARED, TEXTURE, SURFACE, ATOMIC, TMA, ASYNC_COPY, CLUSTER_DSMEM).
  - DVFS card `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` permits intra-SM cycle deltas; absolute conversion is documented as unreliable but per-access deltas remain meaningful.
- Why not primary: Per-instruction NVBit injection can slow kernels 10–100× per `.claude/knowledge/measurement/nvbit-injection-playbook.md`, raising Heisenbug risk that may invalidate the latency measurement itself. DVFS-induced threshold drift adds calibration fragility. Lower-leverage than producing a baseline first (Direction 1) or fitting hardware constants (Direction 5).

### Alt-4: eBPF + CUPTI Bandwidth Fusion

- Gist: Add userspace eBPF uprobes on `libcuda.so` (cuMemAlloc, cuStreamSynchronize, cuLaunchKernel — no driver root required) to capture stream-ordered allocator and launch events, fuse with the existing CUPTI Activity per-kernel timestamps and `sector_stream` 32B-sector byte counts, compute per-kernel effective DRAM bandwidth, and back-derive an implied L2 hit rate as `1 - (B_eff / DRAM_peak_per_arch)`. Register as `algorithm_version = "bandwidth_back_derivation_v1"`. Per-arch DRAM peak constants calibrated via `working_set_capacity_sweep`. Prior art: Volkov-Demmel 2008; Mei-Chu 2017 use bandwidth-as-hit-rate-proxy; bpftrace + libcuda probes are common in the broader Linux observability literature.
- Objective Evidence:
  - LD_PRELOAD + dlsym pattern is already established by `workloads/simbench_bw/cupti_activity_collector.cu` — the same constructor pattern eBPF uprobes would adopt.
  - CUPTI Activity per-kernel `start_ns` / `end_ns` / `duration_ns` already collected and schema-validated (`schemas/cupti_activity_sidecar.schema.json`).
  - Per-kernel byte counts available via `sector_stream.touched_32B_sectors` aggregation.
  - `working_set_capacity_sweep` is the natural per-arch DRAM peak calibration surface.
  - CLAUDE.md Rule 4 (no host-driver work) is satisfied by userspace uprobes; container bootstrap can grant `CAP_BPF` via standard package install.
  - **No existing eBPF precedent in the repo** — `grep -RIn 'bpf\|bpftrace\|uprobe' tools/ workloads/ .claude/` returns zero matches.
- Why not primary: Fundamentally different measurement surface with no in-repo precedent; cloud `CAP_BPF` availability is uncertain (locks the direction to local-only initially); the bandwidth-vs-hit-rate back-derivation conflates instruction throughput, occupancy, and divergence with cache effects unless a covariate-control story is added. High-novelty / high-uncertainty; should land only after Direction 1 baseline shows the LRU model has a residual that bandwidth-fusion could plausibly close.

### Alt-5: Hardware-Constant Calibration Suite

- Gist: Reuse the same 8 memory-microbench slugs as Direction 1, but reframe them as a per-arch CALIBRATION corpus that fits the predictor's free hardware constants (effective L1/L2 capacity, sector replacement bias, set-associativity surrogate, L1-bypass routing fraction) BEFORE running on the 5 admitted activation kernels (treated as the holdout set). Store per-arch fits at `workloads/simbench_bw/calibration_constants_<arch>.json` (mirroring the structure of the existing `golden/cupti_dual_source_tolerance.json`). Prior art: NVArchSim HPCA 2021 microbench-anchored constants; Smith-Eklund 2014 calibration follow-on; GCom 2022.
- Objective Evidence:
  - All 8 microbench `.toml` shape files exist at `workloads/simbench_bw/shapes/<slug>.toml` with explicit working-set / L2-thrash design axes (e.g., `working_set_capacity_sweep.toml` line 21 sweeps `working_set_bytes` from 1 MiB through 32 MiB).
  - All 8 are in the calibration split per `splits.toml` — design intent matches.
  - `cache_predictor.py:407` already parameterizes `cache_model_capacity_bytes` as a caller-supplied input; turning it into a fitted constant requires only a calibration loop, not API redesign.
  - Calibration-fixture precedent at `workloads/simbench_bw/calibrate_dual_source_tolerance.py:1-99` (`golden/cupti_dual_source_tolerance.json` per-arch fixture with calibration_status/policy/git_sha/timestamp metadata).
  - `tools/arch_capabilities.json` declares both arches supported; the same calibration loop runs on either.
  - `workloads/simbench_bw/schemas/cache_predictor.schema.json` schema description records per-arch L2 capacity (`sm_120 = 33554432; sm_86 = 6291456`) as documentation only — fitting these as live constants is a natural extension.
- Why not primary: Direction 5 is the natural NEXT round after Direction 1. The same operator effort that produces Direction 1's labels also produces Direction 5's calibration data; the difference is whether the round's deliverable is the FIRST geomean number (Direction 1, baseline measurement) or the FIRST CALIBRATED geomean number (Direction 5, baseline + improvement). Sub-question 4 of the brainstorm explicitly asks for the smallest defensible step that produces a real first number — that is Direction 1, with Direction 5 sequenced behind it.

## Synthesis Notes

The five alternatives form a natural roadmap behind the primary direction. Direction 1 produces THE coverage that every other direction needs: it establishes the predictor↔label join, populates the comparison_bundle with non-`typed_unavailable` cells, and sets the geomean floor every subsequent round measures itself against. Direction 5 (hardware-constant calibration) is the obvious immediate follow-on because it reuses the same 8 microbench slugs as parameter-fitting anchors instead of as targets — the operator effort overlaps almost completely; the only delta is whether the round's deliverable is "the first number" or "the first calibrated number." A defensible sequencing is Direction 1 → Direction 5 → Direction 2 (SASS-aware L1 bypass) → Direction 3 (calibration head over the now-larger joinable cell set) → Direction 4 (%clock64 latency reconstruction, if model-side improvements plateau above the user's <5% target) → Direction 4-eBPF (only if all model-side moves are exhausted and a fundamentally different measurement surface is needed). Folding alternatives into the primary: the Direction 5 calibration loop can land lockstep with Direction 1's label sweep if the round bandwidth allows (one extra commit reading the admitted cells back into a parameter-fit step); the Direction 2 L1-bypass classifier can land as a prepared scaffold (off-by-default) so Direction 1's baseline number can be re-run with `algorithm_version = smith_eklund_lru_l1bypass_v1` immediately after — turning Direction 2's "did the L1 bypass help?" answer into a same-round delta. The cross-arch discipline (CLAUDE.md "Two-Device Local Development Model") binds every direction equally: each can lift to `Verified-live-sm_86` from this 3090 Ti session and queue an AC-5.2-style follow-on for sm_120 lift to `Cross-arch-verified`.

## Knowledge Consulted

- `.claude/knowledge/INDEX.md` — KB inventory pointer.
- `.claude/skills/validating-nonroot-gpu-measurement/SKILL.md` — non-root measurement specialty router (CUPTI Activity, NVBit, %clock64, NVML covariates, no-NCU contract).
- `.claude/knowledge/measurement/nonroot-cloud-contract.md` — forbidden vs. allowed measurement surfaces; binds every direction to the no-NCU / no-driver-root / no-privileged-CUPTI contract.
- `.claude/knowledge/measurement/cupti-activity-api.md` — CUPTI Activity is the only allowed CUPTI surface; per-kernel timestamps drive Direction 4-eBPF's bandwidth fusion.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — intra-SM `%clock64` deltas remain meaningful under DVFS; absolute wall-clock conversion is unreliable; binds Direction 4 (%clock64 latency reconstruction).
- `.claude/knowledge/measurement/nvbit-injection-playbook.md` — per-instruction tracing can slow kernels by orders of magnitude; documented Heisenbug risk that constrains Direction 4.
- `.claude/skills/analyzing-sass-nvbit-traces/SKILL.md` — per-PC SASS classification; the Ampere `LDG.E.CA/CG/CS` vs Blackwell `LDG.E.STRONG.SM/STRONG.GPU/EF` divergence per BL-20260505 underpins Direction 2.
- `.claude/skills/tuning-blackwell-sm120-kernels/SKILL.md` — sm_120 (RTX PRO 6000 / RTX 5060 laptop) cross-arch context for the cross-host follow-on.
- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf` — Smith-Eklund foundation for the existing predictor; the calibration discussion frames Direction 3 / Direction 5.
- `references/papers/knowledge_base/active/2021__need_for_speed_…_nvarchsim.pdf` — NVArchSim's microbench-anchored hardware-constant calibration approach (Direction 5 prior art).
- `references/papers/knowledge_base/active/2020__accel_sim*.pdf`, `2009__gpgpu_sim*.pdf` — per-instruction memory-hierarchy router precedents for Direction 2.
- `references/papers/knowledge_base/active/2022__gcom_a_detailed_gpu_core_model*.pdf` — probabilistic post-hoc calibration prior art for Direction 3.
- `CLAUDE.md` "Two-Device Local Development Model" — binds every direction to `Verified-live-<arch>` discipline; no `Cross-arch-verified` claim from a single arch.
- `HANDOFF.md` and `MANIFEST.md` — current 3090 Ti session context; v3 / v6 / v7 sm_86 evidence baseline; M1 / M2 closure state.

--- Original Design Draft End ---
