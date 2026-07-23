# Counter-Free GPU Cache Observability — Multi-Architecture Data Collection Toolchain

## Current Progress (Round 24 closed `Verified-live-sm_86`; this RLCR session cancelled; fresh loop starts on 5060)

> **Live-evidence base**: 3090 Ti (sm_86) only. Cross-arch
> validation on 5060 laptop (sm_120) is the next session's
> first responsibility. See CLAUDE.md "Two-Device Local
> Development Model" for the discipline; the
> `Verified-live-sm_86` / `Verified-live-sm_120` /
> `Cross-arch-verified` evidence statuses live in that
> section.
>
> **AC roll-up** (full per-round detail in
> `.humanize/rlcr/2026-05-03_16-17-22/goal-tracker.md`):
>
> - **AC-1** Contract amendment + repo-wide artifact-path
>   scan: **MET** (Rounds 0-2; commits `b8ee2de` /
>   `def5a15` / `aee63b3`). The static guard at
>   `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
>   excludes `data/local_labels/` and `tools/local_labels/`
>   and enforces Rules 2-4 inside both subtrees.
> - **AC-2** Local NCU label driver: **partial** — offline
>   schema + importer + driver verified Rounds 3-7. `task7`
>   (3090 Ti hardware acceptance) and `task8` (5060
>   hardware acceptance) remain pending the cross-arch
>   validation pivot.
> - **AC-3** Repeatability aggregator: **partial** —
>   offline N>=5 / 3pp / `binary_sha256` consistency
>   verified Round 3-4. Hardware repeatability evidence
>   blocked on `task7` / `task8`.
> - **AC-4** NVBit per-instruction trace contract:
>   **partial through Rounds 8-23 plus Round 24 in-flight**.
>   - `task9` schema + `task9b` STANDARD_MEM producer +
>     `task9c` classifier with cluster-DSMEM `.MMIO`
>     heuristic + `task9d` helper-aware live witness:
>     **landed**.
>   - `task11c` 32-entry lane-address sentinel format
>     end-to-end via reconstruction-aware validator:
>     **landed Rounds 17-19**.
>   - `task11d` validator coverage: scalar-presence on
>     `vpc` / `active_mask` / `predicate_mask` (Round 16);
>     `function_unique_id_invalid` /
>     `vpc_inconsistent_with_pc` /
>     `lane_mask_inconsistent` /
>     `lane_vector_length_invalid` /
>     `compressed_format_malformed` /
>     `lane_position_mismatch` (Rounds 15, 17, 18).
>     **Allocation/range identity portion of `task11d`
>     remains outstanding** (load
>     `traces/extra_info/allocations.json`, validate
>     sidecar shape, require `allocation_id` on resolvable
>     STANDARD_MEM records, reject unknown IDs, verify
>     range identity).
>   - `task10` allocator hooks for allocation/range
>     identity: **synchronous coverage landed Rounds 20-22**
>     (per-context registry keyed by `(CUcontext, base)`,
>     full synchronous family — `cuMemAlloc{,_v2}` /
>     `cuMemFree{,_v2}` / `cuMemAllocPitch{,_v2}` /
>     `cuMemAllocManaged` — plus `allocation_id` wire
>     field). **Async / pool / virtual-memory variants
>     remain outstanding** (`cuMemAllocAsync` / `_ptsz`,
>     `cuMemFreeAsync` / `_ptsz`,
>     `cuMemAllocFromPoolAsync` / `_ptsz`,
>     `cuMemAddressReserve`, `cuMemMap`, `cuMemUnmap`,
>     `cuMemAddressFree`).
>   - `task11b` producer-side typed-exclusion sidecars:
>     `unsupported_memory_path` half **landed Round 23**
>     (6 explicit excluded classes) and **extended
>     Round 24 to `kUnspecified`** via additive
>     `"UNSPECIFIED"` enum widening. Live-witness
>     verification on the 3090 Ti shows
>     `kernel_cp_async_no_sync` LDGSTS records routed
>     correctly. **`unmapped_memory_range` remains
>     outstanding** (depends on per-lane allocation
>     resolution + cross-allocation gather/scatter
>     detection — Round 25+).
> - **AC-5 through AC-20**: **NOT MET**. Sector stream,
>   HRD/cache model, locality/projections, divergence
>   proxies, perturbation harness, adaptive sampler,
>   cloud bootstrap, budgets, microbenchmarks,
>   dual-workload validation, NVML, portability,
>   runbook, reproducibility/taxonomy, code-style
>   scanner — all pending. The plan moves into AC-5
>   only after `task10` async/pool/VM allocator coverage,
>   `task11b` `unmapped_memory_range` sidecar, and
>   `task11d` allocation/range identity are landed plus
>   cross-arch verified on the 5060.
>
> **Active session pivot (Round 24 closed `Verified-live-sm_86`)**:
> Round 24 closed cleanly on the 3090 Ti with proper live-GPU
> evidence (860 passed / 2 skipped, sidecar admits ASYNC_COPY
> entry, zero UNSPECIFIED entries — the sm_86-tuned classifier
> covers every microbench opcode so kUnspecified is producer/
> offline/Codex-reviewed and live-regression-verified but not
> positively exercised by live evidence on sm_86). The RLCR
> session at `.humanize/rlcr/2026-05-03_16-17-22/` is now
> CANCELLED (`/humanize:cancel-rlcr-loop`) and stays as
> historical record only. A FRESH RLCR loop starts on the
> 5060 laptop in a new session directory against this same
> plan file; the 5060 session does NOT inherit the cancelled
> session's state.
>
> **5060 session bootstrap responsibilities** (Round 0 of the
> fresh loop):
> 1. `git pull` to the latest master HEAD.
> 2. scp the persistent Claude memory at
>    `~/.claude/projects/-home-jakeshea-NVIDIA-Sims-Predict-NCU/memory/`
>    from the 3090 Ti to the 5060 (10 entries; this is
>    machine-local, not in git).
> 3. Verify `.humanize/` does NOT exist on the 5060 (or
>    remove it); `start-rlcr-loop` refuses on existing loop
>    directories.
> 4. Run `/humanize:start-rlcr-loop plans/nonroot_gpu_cache_observability_plan.md
>    --track-plan-file --max 36` (36 = 60-round budget minus
>    the 24 already spent).
> 5. The new session's Round 0 contract reads (a) this plan's
>    Current Progress section + (b) the persistent memory's
>    `project_rlcr_state.md` and `reference_ac4_map.md` to
>    bootstrap an AC-progress baseline in the new
>    goal-tracker.
> 6. Round 1 = first live-GPU sweep on sm_120 →
>    `Verified-live-sm_120` evidence captured; surface or
>    confirm arch-coupling in the three flagged live tests
>    (`test_nvbit_trace_contract_live.py:777` compression
>    distribution, `test_phase_injection_smoke.py:337`
>    LDGSTS/DEPBAR opcodes, `memory_op_class.h:101` ASYNC_COPY
>    classifier mapping). Once both sm_86 and sm_120 evidence
>    is recorded for the same round of work, that round earns
>    `Cross-arch-verified` per CLAUDE.md.
> 7. After cross-arch baseline lands, continue the AC-4
>    sequence: Round 2+ = `unmapped_memory_range` + per-lane
>    allocation resolution; later = async/pool/VM allocator
>    hooks; later = `task11d` allocation/range validator
>    coverage.

## Goal Description

Build the data-collection toolchain (NVBit tracer extensions, feature extractors, cache-tomography perturbation harness, microbenchmark suite, multi-architecture collection scripts, schemas, and an operator runbook) that produces every non-root signal needed to later predict NCU-equivalent L1/TEX and L2 sector hit-rate metrics for the kernels in `workloads/simbench_bw/` AND `workloads/portable_lcd/`, on each of the four cloud server architectures targeted (RTX PRO 6000 Blackwell sm_120 / GB202, H100 sm_90a, A100 sm_80, L40S sm_89), and produce matching NCU-derived ground-truth labels on the two local devices (RTX 3090 Ti sm_86 and RTX 5060 laptop sm_120).

This plan delivers SCRIPTS the user runs on GPU servers; it does NOT execute server collection itself, does NOT train predictor models, and does NOT validate prediction accuracy. Model training, prediction, validation, baselines, RQ1-RQ6 evaluation, paper writing, and the five paper contributions outlined in the original draft are explicitly deferred to a follow-up plan that consumes the artifacts produced by this toolchain.

To reconcile NCU label generation on local devices with the existing repository contract, this plan amends `CLAUDE.md` Rule 1 to permit `ncu` invocation only on local devices and only into a quarantined directory (`data/local_labels/`), and relaxes the static guard at `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` accordingly. The non-root server contract (Rules 2-5) is unchanged and remains binding for every cloud collection script.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Contract amendment — `CLAUDE.md` Rule 1 is rewritten to permit `ncu` only on local devices and only into the quarantine path `data/local_labels/`. The static guard adds `data/local_labels/` to its `EXCLUDED_PREFIXES` and adds a corresponding scoped exclusion for the new local-label tooling path. Server contract (Rules 2-5) is unchanged. Existing repo-wide static guard test still passes after the amendment.
  - Positive Tests:
    - Running the existing static guard test from a clean checkout passes after the amendment.
    - A new test asserts that local-only NCU invocation under the quarantine path is permitted (the test scans `data/local_labels/` and the new `tools/local_labels/` subtree against a scoped allowlist).
    - `CLAUDE.md` Rule 1 wording explicitly names the quarantine path and explicitly retains the no-server-NCU clause.
  - Negative Tests:
    - Any `ncu` token outside the quarantine path or `tools/local_labels/` subtree fails the static guard with a typed diagnostic.
    - Any `.ncu-rep*` artifact path outside `data/local_labels/` fails the guard.
    - Any nvidia-smi mutation flag, privileged CUPTI surface, kernel-module operation, or driver mutation continues to fail the guard exactly as before.

- AC-2: Local label generation pipeline — A labeling driver under `tools/local_labels/` invokes NCU on local 3090 Ti (sm_86) and local 5060 laptop (sm_120), exports per-launch sector-hit-rate metrics for both L1/TEX and L2 targets, and writes a per-launch sidecar to `data/local_labels/<device>/<slug>/<shape_point>/label_<repeat>_<launch>.json`. The sidecar carries the mandatory provenance contract.
  - Positive Tests:
    - A smoke run on an `sm_86`-runnable kernel from `workloads/portable_lcd/` writes a sidecar that schema-validates and contains numeric `cache_hit_rate_l2_sector` in `[0,1]` with a populated `confidence_interval` of shape `{lower, upper, confidence_level=0.95, units="sector_hit_rate_fraction", aggregation="kernel"}` and `lower <= upper`.
    - The sidecar contains all mandatory provenance fields: `profiler_family` (default `"nsight_compute"`), `profiler_version`, `cuda_version`, `driver_version`, `gpu_uuid_sha256`, `replay_mode`, `aggregation`, `warmup_count`, `binary_sha256`, `schema_version`.
    - The pipeline runs end-to-end on at least one kernel from `workloads/simbench_bw/` whose `arch_gate` allows the local 5060 laptop sm_120.
  - Negative Tests:
    - A sidecar with any missing mandatory field is rejected by the importer.
    - A sidecar with `confidence_interval.lower > confidence_interval.upper` is rejected by the importer.
    - A sidecar referencing a forbidden field name (e.g. embedding raw HW-counter event identifiers) is rejected.

- AC-3: Label repeatability gate — A label cell is admissible only when N>=5 repeated NCU runs at fixed config produce per-target stddev <= 3 percentage points (preregistered; tightened via DEC-3 if needed).
  - Positive Tests:
    - A repeatability-aggregator script ingests N>=5 repeated label sidecars and emits a `label_repeatability.json` summary; cells passing the threshold are written to a `label_admitted.json` index.
    - Cells exceeding the threshold are written to `label_excluded.json` with typed reason `label_repeatability_failed` and the actual stddev.
  - Negative Tests:
    - A single-run label without repeatability evidence is rejected with typed reason `label_repeatability_missing`.
    - A label whose imported sidecars disagree on `binary_sha256` is rejected with typed reason `label_provenance_inconsistent`.

- AC-4: NVBit per-instruction trace contract — The NVBit tracer emits each memory-instruction record with the required fields below.
  - Positive Tests:
    - Required fields populated for every memory-instruction record: lane addresses (32 entries; sentinel for inactive lane), `active_mask`, `predicate_mask`, CTA coords `(x,y,z)`, `warpid_tb`, `warpid_sm`, `function_unique_id`, `vpc` (function-relative PC), `data_width`, `memory_op_class` (one of `global`, `shared`, `local`, `texture`, `surface`, `atomic`, `tma`, `async_copy`, `cluster_dsmem`).
    - `sm_id` is populated on architectures where it is exposed (sm_86, sm_89, sm_90a, sm_120) and is `null` only when documented unavailable.
    - Allocation/range identity is populated via NVBit `cudaMalloc`/`cuMemAlloc` family hooks; ops on unidentified ranges receive a typed exclusion `unmapped_memory_range` and the record is omitted from feature extraction.
    - The proto/JSON schema versions are bumped and the existing tracer build still succeeds.
  - Negative Tests:
    - A trace record missing any required field fails the schema validator and the record is excluded with typed reason.
    - A `tma` / `async_copy` / `cluster_dsmem` record whose lane addresses cannot be reconstructed produces typed exclusion `unsupported_memory_path` and is NOT silently treated as a normal global load.
    - The schema-version bump test fails when the producer side is updated without the consumer side.

- AC-5: Warp coalescer & sector stream generator — A deterministic generator maps `(lane_addresses, active_mask, predicate_mask)` to `(touched_32B_sectors, touched_128B_lines, sectors_per_request)` plus a classification in `{uniform, broadcast, overlap, strided, scattered}`.
  - Positive Tests:
    - Synthetic-trace unit tests pass exact-output checks for: full broadcast, partial predicate, single 32B-aligned access, 32B-crossing access, 128B-crossing access, misaligned 16B vector load, all-inactive-lane case, fully scattered access.
    - Output schema validates against `workloads/simbench_bw/schemas/sector_stream.schema.json` (new schema).
  - Negative Tests:
    - A record with predicate mask AND active mask both zero produces typed `no_lanes_active` and emits no sector.
    - A record with malformed lane address vector (length != 32) is rejected with typed `lane_address_length_invalid`.

- AC-6: Conditional HRD profiler with cache-model spec — Sector → line → allocation → region hierarchy with conditional descent (consult coarser layer only on infinity/cold at the finer layer). The HRD profile is a feature, not the hit rate. A per-target cache-model spec accompanies every emitted feature bundle.
  - Positive Tests:
    - A reference kernel produces an HRD profile whose per-bin bootstrap CI half-width is `<= 0.05` hit-rate-fraction units across all reported bins.
    - The cache-model spec for each evaluated target carries: capacity (sourced from device attributes / static device cards), line/sector geometry (128 B / 32 B), associativity-approximation policy, fill/write behavior assumption, L1-bypass case list (`.cg`, atomic).
    - Profile artifacts validate against `workloads/simbench_bw/schemas/hrd_profile.schema.json` (new schema).
  - Negative Tests:
    - A cache-model spec with missing `capacity` is rejected by the spec validator with typed `cache_model_capacity_missing`.
    - A spec whose declared capacity contradicts CUDA device attributes is rejected with typed `cache_model_capacity_inconsistent`.
    - When any reported bin's bootstrap CI half-width exceeds 0.05, the feature bundle is annotated `cache_model_underconfigured` and a downstream consumer that requests a single hit-rate value receives a refusal, not a default.

- AC-7: GPU-Shalom sharing-locality features (grounded vs projection) — Three GROUNDED classes computed directly from trace (intra-lane, intra-warp, inter-warp same-CTA) and four PROJECTED classes derived from order projections / SM-residency proxies (inter-CTA same-SM, inter-CTA cross-SM, sequential-kernel, private/cold).
  - Positive Tests:
    - Each emitted classification carries `provenance: "grounded"` or `provenance: "projection"`.
    - Each PROJECTED class carries `projection_disagreement_variance` computed across the order projections of AC-8.
    - A reference kernel that exercises only intra-warp reuse produces a non-zero `intra_warp` count with `provenance: "grounded"` and zero `inter_cta_cross_sm` with `provenance: "projection"`.
  - Negative Tests:
    - A pure inter-CTA cross-SM workload that the trace cannot directly observe is NEVER reported as a `grounded` value; if reported, only as `projection` with explicit disagreement variance.
    - Emitting a grounded class without `provenance` is a schema failure.

- AC-8: Schedule-marginalized order projections — Six projections (lane / warp-local / CTA-local / SM-local where SM-id available / kernel-L2 / sequential-allocation). The HRD profiler consumes a feature bundle over projections plus a per-kernel disagreement-variance scalar.
  - Positive Tests:
    - For a deterministic kernel with single CTA / single SM, all projections agree within numeric tolerance and the disagreement variance is below a preregistered noise floor.
    - For a multi-CTA kernel without SM-id evidence (or where SM-id is unavailable), the SM-local projection is omitted and the omission is recorded as typed `projection_unavailable_sm_local`.
  - Negative Tests:
    - A projection bundle missing the `disagreement_variance` scalar fails the schema validator.
    - A projection that requires `sm_id` data on an architecture where it is unavailable but is silently emitted (rather than typed-excluded) fails a regression test.

- AC-9: Memory-divergence and MSHR-pressure proxies — Eight proxies emitted per kernel per launch.
  - Positive Tests:
    - All eight proxies emitted: `unique_sectors_per_warp_inst`, `unique_lines_per_warp_inst`, `sectors_per_request`, `md_loads_per_kilo_inst`, `outstanding_load_density_by_phase`, `pc_repeated_miss_likely_streams`, `stride_entropy`, `throughput_saturation_slope_under_resident_warp_sweep`.
    - Each proxy has a synthetic-kernel unit test that pins the expected output to within numeric tolerance.
    - Proxy bundle validates against `workloads/simbench_bw/schemas/divergence_proxies.schema.json` (new schema).
  - Negative Tests:
    - A proxy whose computation requires evidence the trace did not capture emits typed exclusion (e.g. `outstanding_load_density_unavailable_no_phase_records`) instead of a default zero.
    - A unit test asserting "uniform broadcast warp must produce sectors_per_request == 1" fails when the coalescer regresses.

- AC-10: Cache-tomography perturbation harness with SASS confirmation — Matched perturbation runs across the families enumerated below; every perturbed binary verified by SASS disassembly to contain the intended modifier on the target memory instruction class.
  - Positive Tests:
    - Perturbation families implemented and runnable on each target architecture where supported: `.ca`/`.cg`/`.cs` cache modifiers; `.L2::64B`/`.L2::128B`/`.L2::256B` prefetch hints (sm_75+/sm_80+); L2 persisting access window (sm_80+); working-set sweeps (parametric); stride and misalignment sweeps; warm/cold pairs; eviction-buffer sweeps.
    - For every produced perturbation binary, the SASS-confirmation step parses `cuobjdump --dump-sass` output and asserts the intended modifier appears on the target instruction class; the binary's confirmation hash is committed to `perturbation_run_ledger.json`.
    - Per-architecture legality is checked at compile time; unsupported families produce typed exclusion `arch_feature_unavailable_<feature>`.
  - Negative Tests:
    - A perturbation binary whose SASS lacks the intended modifier produces typed exclusion `modifier_optimized_away` and the run is dropped from the dataset (NOT silently treated as no-op).
    - A perturbation requiring `.L2::256B` on sm_75 fails the legality check at compile time with typed `arch_feature_unavailable_l2_prefetch_256b`.
    - L2 persisting access window on a target where MIG/MPS/driver disables it produces typed `arch_feature_unavailable_l2_persisting`.

- AC-11: Condens-style adaptive NVBit sampling — Trace-window iteration that stops when max change across selected MRC quantiles is `<= 0.02` hit-rate-fraction units for K=3 consecutive windows; honors existing `trace_volume_exceeded.json` budget mechanism.
  - Positive Tests:
    - Adaptive controller emits a `trace_convergence_confidence.json` sidecar per launch with per-quantile residual.
    - On a synthetic stationary trace, the controller stops within a bounded number of windows.
    - On a non-stationary synthetic trace, the controller continues sampling until the budget cap and then emits typed `trace_convergence_not_reached_budget_cap`.
  - Negative Tests:
    - Disabling the controller's stopping rule on a stationary trace causes the controller to exceed budget; the regression test catches the regression.
    - The controller never silently truncates a trace; truncation always emits typed `trace_volume_exceeded`.

- AC-12: Server collection script package — A single-command bootstrap (`tools/cloud/collect_all.sh` or equivalent name) runs all collection flows on each of {sm_120 RTX PRO 6000 Blackwell, sm_90a H100, sm_80 A100, sm_89 L40S}, gracefully degrades when arch features unavailable, and emits all sidecars per the existing dual-run manifest contract. Admissibility budgets per AC-13.
  - Positive Tests:
    - Bootstrap runs preflight (`tools/cloud/preflight.sh`), records a preflight manifest, then runs CUPTI-Activity timing, NVBit trace, NVML covariate sampling, microbenchmark suite, and perturbation harness in the documented order; emits the dual-run manifest at the end.
    - The bootstrap exits with a typed exit code when the preflight manifest's compute capability is not in the supported list.
    - On an architecture where a perturbation family is unsupported, the bootstrap continues and the missing family is recorded in the run summary with typed `arch_feature_unavailable_<feature>`.
  - Negative Tests:
    - A bootstrap on an unsupported architecture (e.g. sm_70) terminates immediately with typed `arch_unsupported`.
    - A bootstrap that detects forbidden surfaces (privileged CUPTI, clock locking, root-required device-state mutation attempts) terminates with typed `forbidden_surface_attempted` and a non-zero exit code.
    - A bootstrap that finds no kernels admissible for the target architecture writes an empty-but-well-formed run summary instead of crashing.

- AC-13: NVBit admissibility budgets — Per-launch budgets enforced by both the run controller and post-collection verifier: slowdown `<= 200x`, trace volume `<= 16 GiB`, dropped records `<= 5%`.
  - Positive Tests:
    - A run that exceeds slowdown 200x emits typed exclusion `nvbit_slowdown_budget_exceeded` and writes the actual ratio.
    - A run that exceeds 16 GiB emits typed `trace_volume_exceeded` (existing sidecar).
    - A run with > 5% dropped records emits typed `nvbit_drop_rate_exceeded`.
    - The verifier (`workloads/simbench_bw/verify_artifacts.py`) reads each run summary and rejects any artifact violating the budgets.
  - Negative Tests:
    - A budget update that is not reflected in both the run controller and the verifier is caught by an internal-consistency unit test.
    - A run that silently truncates and reports success without typed-exclusion sidecar fails the verifier.

- AC-14: Microbenchmark suite — Concrete kernels under `workloads/portable_lcd/microbench/` (or new sibling directory; placement chosen during execution) implement the full suite: pointer chase for latency bands; working-set sweep for capacity knees; stride/misalignment sweep for sector amplification; warm/cold cache pair; `.ca`/`.cg`/`.cs` variants; L2 prefetch hint variants where supported; L2 persisting-window sweep where supported; carveout sensitivity probe; bandwidth saturation sweep; resident-warp pressure sweep.
  - Positive Tests:
    - Each microbenchmark kernel compiles and runs on each supported architecture; per-arch SASS slices verified by the existing `verify_kernel_contracts.py`-style discipline.
    - Each microbenchmark emits a per-run sidecar that schema-validates and is consumable by the feature extractors of AC-5 through AC-9.
    - The suite is exercised against fixtures derived from BOTH `workloads/simbench_bw/` AND `workloads/portable_lcd/` (kernel registries are queried; both registries contribute kernels).
  - Negative Tests:
    - A microbenchmark that compiles only against `sm_120a` / `sm_120f` / `sm_90a`-conditional features fails the LCD discipline check (per `workloads/portable_lcd/common.mk` rules) and is rejected at build time.
    - A microbenchmark that does not have a matching shape file fails the offline validator.

- AC-15: Validate on both workload directories — Every collection script, schema, feature extractor, and perturbation harness is exercised against fixtures derived from BOTH `workloads/simbench_bw/` AND `workloads/portable_lcd/`. CI tests pass on both rosters.
  - Positive Tests:
    - `pytest workloads/simbench_bw/tests/` passes after toolchain integration.
    - `pytest workloads/portable_lcd/tests/` passes after toolchain integration.
    - A combined integration test discovers admissible kernels from both registries and runs the toolchain end-to-end on at least one kernel from each.
  - Negative Tests:
    - A change that breaks consumption of `workloads/portable_lcd/roster.toml` is caught by the combined integration test.
    - A schema change applied to one registry but not the other is caught by the schema-parity test.

- AC-16: NVML covariate collection — NVML covariate sampling runs on all four server architectures using read-only fields per `workloads/simbench_bw/schemas/nvml_covariate.schema.json`. Sample cadence preregistered (default 100 ms; configurable via env var); preflight enforces matching cadence between CUPTI and NVBit runs for join validity.
  - Positive Tests:
    - NVML sidecar emitted per run with `temp_c`, `power_mw`, `sm_clock_mhz`, `mem_clock_mhz`, `timestamp_ns` per sample.
    - The dual-run manifest emitter accepts the NVML sidecar from the new architectures (sm_80, sm_89, sm_90a were already supported by schema; sm_120 path added).
    - A preflight that detects mismatched NVML cadence between CUPTI and NVBit runs fails with typed `nvml_cadence_mismatch`.
  - Negative Tests:
    - An NVML sample without `timestamp_ns` is rejected by the validator.
    - An attempt to use NVML mutation APIs (lock clocks, set persistence) fails the static guard test.

- AC-17: Architecture portability layer — Every script and feature extractor declares its supported `sm_*` set; missing hardware features produce typed exclusions; never silent failure or wrong defaults.
  - Positive Tests:
    - Each script exposes a `supported_arches: list[str]` constant or equivalent metadata; a portability-coverage test enumerates them.
    - On an unsupported `sm_*`, the script emits a typed `arch_unsupported` exclusion sidecar and a non-zero exit code.
  - Negative Tests:
    - Adding a new script without declaring its supported `sm_*` set fails the portability-coverage test.
    - Silently returning a default value when an architecture-conditional feature is unavailable is caught by the typed-exclusion regression test.

- AC-18: Operator runbook & verification — A README per workload directory plus a top-level `tools/cloud/RUNBOOK.md` documents preflight, run, and verify steps. A new operator can complete a fresh-container collection on each server architecture without author assistance. Offline tests run without GPU.
  - Positive Tests:
    - `tools/cloud/RUNBOOK.md` exists and documents the bootstrap, the four target architectures, the kernel registries, the artifact layout, and the verifier commands.
    - Each workload directory's README is updated with the data-collection invocation.
    - All offline tests under `workloads/simbench_bw/tests/`, `workloads/portable_lcd/tests/`, and `tools/cloud/test_*.py` pass without a GPU.
  - Negative Tests:
    - A runbook that omits one of the four target architectures fails a runbook-coverage test.
    - An offline test that requires a GPU is caught by the no-GPU CI pass.

- AC-19: Reproducibility & exclusion taxonomy — Every artifact carries manifest hashes, container digest, source commit, env vars, random seeds; every failure mode emits a typed exclusion. No silent default values anywhere in the toolchain.
  - Positive Tests:
    - Each run summary carries `manifest_sha256`, `container_digest`, `source_commit`, `env_var_snapshot`, `random_seed` (when stochastic).
    - The exclusion taxonomy includes (at minimum): `unmapped_memory_range`, `trace_volume_exceeded`, `modifier_optimized_away`, `nvml_alignment_gap`, `label_repeatability_failed`, `label_repeatability_missing`, `label_provenance_inconsistent`, `unsupported_memory_path`, `dual_run_manifest_missing`, `nvbit_slowdown_budget_exceeded`, `nvbit_drop_rate_exceeded`, `arch_feature_unavailable_*`, `arch_unsupported`, `cache_model_capacity_missing`, `cache_model_capacity_inconsistent`, `cache_model_underconfigured`, `nvml_cadence_mismatch`, `forbidden_surface_attempted`, `lane_address_length_invalid`, `no_lanes_active`, `projection_unavailable_sm_local`, `trace_convergence_not_reached_budget_cap`, `outstanding_load_density_unavailable_no_phase_records`.
    - A grep for the literal taxonomy strings appears in both the producer code and the verifier code (parity check).
  - Negative Tests:
    - A new failure mode introduced without a corresponding typed exclusion is caught by the exclusion-coverage test.
    - A silent default (e.g. zero-fill) on an unobservable signal is rejected by the regression suite.

- AC-20: Code style — Implementation code, tests, comments, and runbook prose contain no plan-specific terminology. The tokens `AC-`, `Milestone`, `Phase`, `Step`, and similar workflow markers do not appear in `tools/`, `workloads/simbench_bw/`, `workloads/portable_lcd/`, `data/local_labels/` (excluding markdown that is intentionally documentation-only and not consumed by code).
  - Positive Tests:
    - A scanner test under `tests/` greps the implementation tree for forbidden plan tokens; passes when none are found in scanned files.
    - Domain-appropriate naming is used (e.g. `cache_hit_rate_l2_sector` not `ac13_metric`).
  - Negative Tests:
    - Adding a comment containing `AC-` or `Milestone` to any in-scope file fails the scanner test.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The toolchain implements all twenty acceptance criteria across both workload directories with full coverage on all four server architectures (sm_120, sm_90a, sm_80, sm_89) and both local devices (sm_86, sm_120). All eight method components from the original draft are realized as data-collection or feature-extraction infrastructure. The microbenchmark suite covers all ten families enumerated in AC-14. The cache-tomography perturbation harness implements every perturbation family enumerated in AC-10 with per-architecture legality and SASS-confirmation. Every script is documented in `tools/cloud/RUNBOOK.md` and per-workload READMEs. The Condens-style adaptive sampler is parameterized and exposes per-quantile thresholds. NVML covariate cadence is preregistered per architecture. The exclusion taxonomy is exhaustive. Offline tests pass without GPU on all rosters. The contract amendment ships as a discrete change with its own audit-friendly diff in `CLAUDE.md`, `MANIFEST.md`, and the static guard test.

### Lower Bound (Minimum Acceptable Scope)

The toolchain implements: AC-1 contract amendment; AC-2 local label generation pipeline limited to L2 sector hit rate on the local 3090 Ti for kernels in `workloads/portable_lcd/`; AC-3 label repeatability gate; AC-4 NVBit per-instruction trace contract restricted to global memory ops; AC-5 warp coalescer & sector stream generator; AC-6 conditional HRD profiler with cache-model spec for L2 only; AC-7 grounded sharing-locality classes only (intra-lane / intra-warp / inter-warp same-CTA); AC-8 four projections (lane / warp-local / CTA-local / kernel-L2); AC-9 four divergence proxies (`unique_sectors_per_warp_inst`, `sectors_per_request`, `md_loads_per_kilo_inst`, `stride_entropy`); AC-10 perturbation harness with two families (`.ca`/`.cg` cache modifier + working-set sweep) on sm_120 only; AC-11 adaptive sampling with stationary stopping rule; AC-12 server collection script package targeting sm_120 only; AC-13 budgets enforced; AC-14 microbenchmark suite limited to pointer chase, working-set sweep, and `.ca`/`.cg` variants; AC-15 toolchain validated on at least one kernel from each of `workloads/simbench_bw/` and `workloads/portable_lcd/`; AC-16 NVML covariate collection on sm_120 only; AC-17 architecture portability layer with single supported sm_*; AC-18 runbook and offline tests passing without GPU; AC-19 reproducibility & exclusion taxonomy for the implemented surfaces; AC-20 code style.

### Allowed Choices

- Can use: NVBit 1.8 dynamic instrumentation; CUPTI Activity API (kernel/concurrent-kernel/memcpy/memset/runtime/driver/overhead/device/context records only); CUDA events; static SASS via `cuobjdump`/`nvdisasm`/`cu++filt`; static PTX via `ptxas -v`; read-only NVML covariates; `%clock64`/`%globaltimer` as phase-boundary diagnostics ONLY (not as inputs to cache prediction features); JSON or protobuf or parquet for sidecars; Python or C++ or CUDA for implementation; the existing dual-run manifest emitter/consumer; the existing trace-budget infrastructure; the existing schema validator; pytest for tests; `cuobjdump --dump-sass` for SASS-confirmation of perturbations; the chosen `provenance.profiler_family` identifier value (default `"nsight_compute"`; user may override via DEC-11 to any guard-safe string).
- Can use ONLY on local devices and ONLY into the quarantine path `data/local_labels/` (per AC-1 amendment): `ncu` / `nv-nsight-cu-cli`; `.ncu-rep*` artifacts; NCU section/metric IDs.
- Cannot use: privileged CUPTI surfaces (Events, Metrics, Range Profiling, PM Sampling, PerfWorks, PC Sampling, SASS Metrics, Instruction Execution, Instruction Correlation); `nvidia-smi` clock locking, persistence, MIG, power-limit, compute-mode, application-clock changes; kernel-module operations (DKMS, MOK, modprobe, NVreg_RestrictProfilingToAdminUsers); host-driver install; simulator revival (nvasim, Accel-Sim, GPGPU-Sim); architecture-conditional build targets (`sm_90a`, `sm_120a`, `sm_120f`) outside the explicit per-architecture variant slots permitted by `workloads/portable_lcd/common.mk`; `discard.global.L2` on live application data.

> **Note on Deterministic Designs**: The original draft's eight method components and five paper contributions are preserved verbatim. This plan narrows them to data-collection scope only (the predictor / training / paper portions are explicitly deferred to a follow-up plan). Within data-collection scope, the architecture, schemas, and exclusion taxonomy are fixed; ML model family, training methodology, and paper writing are out of scope and not constrained here.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

A reasonable execution path layers the work in dependency order:

1. Land the contract amendment first (AC-1). The static guard test must pass on master before any NCU code lands.
2. Stand up the local label pipeline (AC-2, AC-3). The pipeline is single-host, single-architecture per run, and can be exercised on the local 3090 Ti against a small `workloads/portable_lcd/` subset before scaling.
3. Extend the NVBit tracer (AC-4). The proto schema bumps; the `inst_trace_t` struct in `tools/nvbit_tracer/tracer_tool/common.h` already carries CTA coords, warp ids, SM id, and `vpc`. Add `memory_op_class`, allocation/range identity, and the unsupported-memory-path typed exclusion.
4. Build the feature-extractor library (AC-5 through AC-9). Each extractor consumes the new trace schema and emits a sidecar. Synthetic-trace unit tests gate correctness independent of any GPU run.
5. Build the perturbation harness (AC-10). The SASS-confirmation step is the gate; without it, no perturbation binary enters the dataset.
6. Build the adaptive sampler (AC-11) on top of the existing trace-budget infrastructure (`tools/cloud/trace_budgets.py`, `workloads/simbench_bw/trace_size_estimator.py`).
7. Implement the microbenchmark suite (AC-14) co-located with the existing portable_lcd discipline.
8. Wire everything into the server collection script package (AC-12, AC-13, AC-16). Bootstrap runs preflight, then CUPTI-Activity timing, NVBit trace, NVML covariate, microbench, and perturbation in matched-launch order. The dual-run manifest emitter pairs the producers as today.
9. Write the runbook (AC-18) and finalize the exclusion taxonomy (AC-19).
10. Validate on both workload directories (AC-15) end-to-end as the integration gate.

The local label pipeline can run in parallel with the NVBit tracer extension after AC-1 lands. Feature extractors can be built in parallel against synthetic-trace fixtures before the tracer extension is complete. The perturbation harness, adaptive sampler, microbenchmark suite, and server collection script package compose at the end.

### Relevant References

- `tools/nvbit_tracer/tracer_tool/common.h` — `inst_trace_t` struct already carries CTA coords, warp ids, SM id, vpc; extend with `memory_op_class` and allocation/range identity hooks.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto` — bump schema; add `memory_op_class`.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/address.proto` — already exposes `data_width`, `base_address`, `stride`, per-lane addrs.
- `workloads/simbench_bw/cupti_activity_collector.cu` — existing per-launch sidecar emitter; reuse the LD_PRELOAD wiring pattern.
- `workloads/simbench_bw/dual_run_manifest_emitter.py` — existing pair logic with `(slug, shape_point, repeat_index, launch_index, kernel_signature_sha256, preflight_manifest_sha256)` as the matching key; extend to consume the new feature-extractor sidecars.
- `workloads/simbench_bw/dual_run_manifest_consumer.py` — existing admission diagnostics; extend with the new exclusion taxonomy entries.
- `workloads/simbench_bw/schemas/` — existing schemas (CUPTI activity sidecar, phase cycle ledger, dual-run manifest, NVML covariate); add `sector_stream`, `hrd_profile`, `divergence_proxies`, `sharing_locality`, `perturbation_run_ledger`, `local_label_sidecar` schemas.
- `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` — `EXCLUDED_PREFIXES` tuple is the relaxation surface; add `data/local_labels/` and the new `tools/local_labels/` quarantine prefix.
- `workloads/simbench_bw/roster.toml`, `splits.toml`, `shapes/` — existing 74-kernel registry with arch-gated rows; the new toolchain queries this registry without modifying the splits.
- `workloads/portable_lcd/roster.toml`, `workloads/portable_lcd/common.mk` — second registry the toolchain must consume; LCD discipline forbids `sm_90a` / `sm_120a` / `sm_120f` build targets, which the microbenchmark suite must respect.
- `workloads/portable_lcd/verify_kernel_contracts.py` — existing kernel-contract verifier model; mirror for the new sector_stream and HRD schemas.
- `workloads/simbench_bw/nvml_covariate_join.py`, `nvml_covariate_acceptance.toml` — existing covariate alignment and acceptance threshold; extend to additional architectures.
- `tools/cloud/preflight.sh`, `tools/cloud/trace_budgets.py` — existing preflight and trace-budget enforcement; extend the trace-budget decision to honor the new 200x slowdown / 16 GiB / 5% drop budgets.
- `.claude/knowledge/measurement/nonroot-cloud-contract.md` — non-root contract description; informs the contract amendment wording.
- `.claude/knowledge/measurement/cupti-activity-api.md` — allowed Activity kinds.
- `.claude/knowledge/measurement/nvbit-injection-playbook.md` — invocation pattern, tool-combination constraints, sm_120 binary-version fallback.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — explicit guidance that `%clock64` and `%globaltimer` are diagnostic phase signals only and must NOT serve as cache-prediction inputs.
- `.claude/knowledge/cuda/cuda-memory-hierarchy.md`, `.claude/knowledge/best-practices/perf-coalescing-memory.md`, `.claude/knowledge/ptx/ptx-isa-cache-hints.md` — coalescing rules, sector geometry (32 B sectors / 128 B lines), PTX cache modifier semantics for the perturbation harness.
- `.claude/knowledge/blackwell/blackwell-tuning.md`, `.claude/knowledge/patents/arch-hopper.md`, `.claude/knowledge/patents/arch-ampere.md` — per-architecture cache geometry numbers used by the cache-model spec validator.
- Original draft Source Map — local primary target papers (`maeda2017.pdf` HRD, `jiang2022.pdf` Shalom) and prior-art comparison papers (Nugteren 2014, NVBit 2019, MDM 2020, microbenchmarking 2022/2024/2025).

## Dependencies and Sequence

### Milestones

1. Contract amendment: rewrite `CLAUDE.md` Rule 1 and the static guard relaxation.
   - Step 1: Draft the Rule 1 wording change naming the quarantine path.
   - Step 2: Add `data/local_labels/` to `EXCLUDED_PREFIXES` in the static guard test.
   - Step 3: Add a scoped allowlist for the new `tools/local_labels/` subtree.
   - Step 4: Update `MANIFEST.md` to declare the new directories as in-scope.
   - Step 5: Run the existing repo-wide static guard test; assert it still passes.

2. Local label generation pipeline.
   - Step 1: Create `tools/local_labels/` with a labeling driver that wraps NCU.
   - Step 2: Create `data/local_labels/` with a `.gitignore` for `.ncu-rep*` artifacts (commit only the per-launch JSON sidecars).
   - Step 3: Define `local_label_sidecar.schema.json` and the importer.
   - Step 4: Implement the repeatability aggregator.
   - Step 5: Smoke run on a `workloads/portable_lcd/` kernel on the local 3090 Ti.
   - Step 6: Smoke run on a `workloads/simbench_bw/` kernel on the local 5060 laptop.

3. NVBit tracer extension.
   - Step 1: Bump the proto schema and add `memory_op_class`.
   - Step 2: Add allocation/range identity hooks to NVBit `cudaMalloc`/`cuMemAlloc` family callbacks.
   - Step 3: Add typed exclusion `unsupported_memory_path` for TMA / async-copy / cluster-DSMEM / atomic / texture / surface paths where lane addresses cannot be reconstructed.
   - Step 4: Update the existing tracer build and tests; bump consumer-side schema versions.

4. Feature-extractor library.
   - Step 1: Warp coalescer & sector stream generator with synthetic-trace unit tests.
   - Step 2: Conditional HRD profiler with bootstrap CI gate.
   - Step 3: Cache-model spec validator (capacity / line-sector geometry / associativity / fill-write / L1-bypass).
   - Step 4: Six order projections.
   - Step 5: Sharing-locality classifier with grounded vs projection annotation.
   - Step 6: Eight divergence and MSHR-pressure proxies.
   - Step 7: New schemas committed under `workloads/simbench_bw/schemas/`.

5. Cache-tomography perturbation harness.
   - Step 1: Implement perturbation binaries for `.ca`/`.cg`/`.cs`, L2 prefetch hints, L2 persisting access window, working-set sweeps, stride/misalignment sweeps, warm/cold pairs, eviction-buffer sweeps.
   - Step 2: SASS-confirmation step parsing `cuobjdump --dump-sass`.
   - Step 3: Per-architecture legality gate.
   - Step 4: `perturbation_run_ledger.json` schema and emitter.

6. Microbenchmark suite.
   - Step 1: Pointer chase, working-set sweep, stride/misalignment sweep, warm/cold pair.
   - Step 2: `.ca`/`.cg`/`.cs` variants.
   - Step 3: L2 prefetch hint variants where supported.
   - Step 4: L2 persisting-window sweep where supported.
   - Step 5: Carveout sensitivity probe.
   - Step 6: Bandwidth saturation sweep.
   - Step 7: Resident-warp pressure sweep.
   - Step 8: Per-arch SASS slices respecting LCD discipline (sm_86 / sm_90 / sm_120 baseline).

7. Adaptive NVBit sampler.
   - Step 1: Trace-window iteration with K=3 stationary stopping rule.
   - Step 2: `trace_convergence_confidence.json` sidecar.
   - Step 3: Integration with existing `trace_volume_exceeded.json` budget.

8. Server collection script package.
   - Step 1: Bootstrap script `tools/cloud/collect_all.sh` (or chosen name).
   - Step 2: Architecture-portability layer with declared `supported_arches`.
   - Step 3: Admissibility budget enforcement (200x / 16 GiB / 5%).
   - Step 4: Run-summary emitter and verifier extension.
   - Step 5: NVML covariate collection on sm_80, sm_89, sm_90a, sm_120 paths.

9. Validation on both workload directories and operator runbook.
   - Step 1: Combined integration test discovers admissible kernels from both registries.
   - Step 2: `tools/cloud/RUNBOOK.md` documents the bootstrap on each of the four server architectures.
   - Step 3: README updates in both workload directories.
   - Step 4: Offline tests under `workloads/simbench_bw/tests/`, `workloads/portable_lcd/tests/`, and `tools/cloud/test_*.py` pass without GPU.

Milestone 1 must complete before Milestones 2 and 8 can land. Milestones 3 and 4 may proceed in parallel after Milestone 1. Milestones 5, 6, 7 depend on Milestones 3 and 4. Milestone 8 depends on Milestones 3-7. Milestone 9 depends on Milestone 8.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|------|------------|
| task1 | Rewrite CLAUDE.md Rule 1 to permit local NCU under `data/local_labels/`; update MANIFEST.md | AC-1 | coding | - |
| task2 | Update `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` to add `data/local_labels/` and `tools/local_labels/` to `EXCLUDED_PREFIXES`; add a scoped allowlist test | AC-1 | coding | task1 |
| task3 | Verify that the relaxation does not weaken the server contract by running the static guard test and confirming all server-context bans still fire (analyze the diff for unintended scope changes) | AC-1 | analyze | task2 |
| task4 | Define `local_label_sidecar.schema.json` (provenance, confidence_interval shape, target metric fields) | AC-2 | coding | task2 |
| task5 | Implement `tools/local_labels/label_driver.sh` (NCU wrapper) and `tools/local_labels/import_label.py` (sidecar importer + validator) | AC-2 | coding | task4 |
| task6 | Implement `tools/local_labels/repeatability_aggregator.py` (stddev gate; emits admitted/excluded indices) | AC-3 | coding | task5 |
| task7 | Smoke-run the labeling driver on the local 3090 Ti against a `workloads/portable_lcd/` kernel; commit only the JSON sidecar | AC-2, AC-3 | coding | task6 |
| task8 | Smoke-run the labeling driver on the local 5060 laptop against a `workloads/simbench_bw/` kernel whose arch_gate allows sm_120 | AC-2, AC-3 | coding | task7 |
| task9 | Bump `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto` schema; add `memory_op_class` enum | AC-4 | coding | task2 |
| task10 | Add NVBit `cudaMalloc`/`cuMemAlloc` family hooks for allocation/range identity in `tracer_tool.cu` | AC-4 | coding | task9 |
| task11 | Add typed exclusion `unsupported_memory_path` for TMA / async-copy / cluster-DSMEM / atomic / texture / surface where lane addresses cannot be reconstructed | AC-4 | coding | task10 |
| task12 | Implement warp coalescer & sector stream generator with synthetic-trace unit tests (broadcast, partial predicate, 32B crossing, 128B crossing, misaligned vector load, all-inactive lanes, scattered) | AC-5 | coding | task9 |
| task13 | Define `sector_stream.schema.json` and the validator | AC-5 | coding | task12 |
| task14 | Implement conditional HRD profiler with bootstrap CI gate (per-bin CI half-width <= 0.05 hit-rate-fraction) | AC-6 | coding | task12 |
| task15 | Implement cache-model spec validator (capacity / line-sector / associativity / fill-write / L1-bypass) and `hrd_profile.schema.json` | AC-6 | coding | task14 |
| task16 | Cross-check the cache-model spec capacity defaults against per-architecture device cards (sm_80, sm_86, sm_89, sm_90a, sm_120) | AC-6 | analyze | task15 |
| task17 | Implement six order projections (lane / warp-local / CTA-local / SM-local / kernel-L2 / sequential-allocation) with disagreement-variance scalar | AC-8 | coding | task14 |
| task18 | Implement sharing-locality classifier with grounded vs projection annotation; `sharing_locality.schema.json` | AC-7 | coding | task17 |
| task19 | Implement eight divergence and MSHR-pressure proxies; `divergence_proxies.schema.json` | AC-9 | coding | task12 |
| task20 | Implement perturbation harness for `.ca`/`.cg`/`.cs`, L2 prefetch hints, L2 persisting window, working-set sweep, stride/misalignment sweep, warm/cold pair, eviction-buffer sweep | AC-10 | coding | task15 |
| task21 | Implement SASS-confirmation step using `cuobjdump --dump-sass`; reject perturbations whose modifier was optimized away | AC-10 | coding | task20 |
| task22 | Implement per-architecture legality gate for perturbation families | AC-10 | coding | task20 |
| task23 | Define `perturbation_run_ledger.json` schema and emitter | AC-10 | coding | task20 |
| task24 | Implement Condens-style adaptive sampler with K=3 stationary stopping rule and `trace_convergence_confidence.json` sidecar | AC-11 | coding | task19 |
| task25 | Implement microbenchmark kernel set: pointer chase, working-set sweep, stride/misalignment sweep, warm/cold pair, `.ca`/`.cg`/`.cs` variants, L2 prefetch hint variants, L2 persisting-window sweep, carveout probe, bandwidth saturation, resident-warp pressure | AC-14 | coding | task22 |
| task26 | Add per-architecture SASS slices for the microbenchmark kernels respecting LCD discipline (sm_86 + sm_90 + sm_120 baseline) | AC-14, AC-17 | coding | task25 |
| task27 | Implement bootstrap `tools/cloud/collect_all.sh` (or chosen name); preflight, CUPTI Activity, NVBit trace, NVML covariate, microbench, perturbation in matched-launch order | AC-12 | coding | task24 |
| task28 | Add admissibility budget enforcement (200x slowdown / 16 GiB trace volume / 5% drop) to run controller and post-collection verifier | AC-13 | coding | task27 |
| task29 | Extend dual-run manifest emitter / consumer to consume new feature-extractor sidecars and exclusion entries | AC-12, AC-19 | coding | task19 |
| task30 | Add NVML covariate collection paths for sm_80, sm_89, sm_90a, sm_120 (sm_80, sm_89, sm_90a were schema-supported; sm_120 path added) | AC-16 | coding | task27 |
| task31 | Implement architecture-portability layer with declared `supported_arches` per script and a portability-coverage test | AC-17 | coding | task27 |
| task32 | Wire combined integration test that discovers admissible kernels from both `workloads/simbench_bw/` and `workloads/portable_lcd/` registries and runs the toolchain end-to-end on at least one kernel from each | AC-15 | coding | task27 |
| task33 | Write `tools/cloud/RUNBOOK.md` documenting bootstrap on each of sm_120, sm_90a, sm_80, sm_89; update README in both workload directories | AC-18 | coding | task27 |
| task34 | Implement exclusion-taxonomy parity check (producer code and verifier code agree on the typed exclusion strings) | AC-19 | coding | task29 |
| task35 | Implement code-style scanner test that greps the implementation tree for `AC-`, `Milestone`, `Phase`, `Step` tokens | AC-20 | coding | task27 |
| task36 | Independent design review of the bootstrap, budget enforcement, and exclusion taxonomy for completeness, ordering correctness, and cloud-cost rationality | AC-12, AC-13, AC-19 | analyze | task34 |
| task37 | Independent review of the cache-model spec defaults and the HRD bootstrap-CI threshold against the per-architecture cache geometry numbers in `.claude/knowledge/blackwell/blackwell-tuning.md`, `.claude/knowledge/patents/arch-hopper.md`, `.claude/knowledge/patents/arch-ampere.md` | AC-6 | analyze | task16 |
| task38 | Confirm sm_89 (L40S) cache hierarchy figures and L2 prefetch-hint support against NVIDIA Ada/L40S product documentation; record findings in the cache-model spec defaults | AC-6, AC-14 | analyze | task16 |

## Claude-Codex Deliberation

### Agreements

- External-label-only framing (Round 1) is contract-safe; user later overrode this to local NCU under quarantine, which preserves the server contract.
- Lower bound shape: external-label ingestion + warp coalescer + conditional HRD + grounded sharing + 4 baselines + L2-only.
- Positive and negative tests per AC are necessary for falsifiability.
- SASS-confirmed perturbations and typed exclusions are mandatory.
- Server validation must be label-free; perturbation direction, knees, stability, and timing-residual enrichment are reasonable label-free checks (now deferred to the follow-up plan).
- Cross-device splits and zero shape-identical leakage are required for credibility (now deferred to the follow-up plan).
- `%clock64` and `%globaltimer` are phase-boundary diagnostics only, never inputs to cache-prediction features.
- Schema-level assertion that `confidence_interval.lower <= confidence_interval.upper` is required.
- The exclusion taxonomy must include `unsupported_memory_path` for sm_120 TMA / async-copy / cluster-DSMEM where lane addresses cannot be reconstructed.

### Resolved Disagreements

- Topic: NCU label policy. Codex (Rounds 1-3) and Claude (Rounds 1-3) both held the position that local-NCU labels must be EXTERNAL imported artifacts to preserve the static guard, with the alternative being to amend Rule 1. The user's explicit decision overrode this: Rule 1 is amended to permit `ncu` only on local devices and only into a quarantined directory `data/local_labels/`. The static guard relaxation is scoped to that prefix; the server contract (Rules 2-5) is unchanged. Resolution: amendment proceeds as AC-1.

- Topic: L1/TEX scope. Codex (Rounds 1-3) recommended L2-only as the first defensible target because L1/TEX is policy-heavy, opcode-path-dependent, and noisy on Blackwell sm_120 (TMA / async-copy / cluster-DSMEM bypass per-lane address inference). Claude initially echoed Codex; the user explicitly overrode and selected "Both L1/TEX and L2 as primary targets". Resolution: the data-collection toolchain captures evidence sufficient for both targets; L1/TEX prediction risk is acknowledged but moves to the follow-up plan's domain; the lower-bound for THIS data-collection plan still scopes the local label pipeline to L2 only (with L1/TEX added in upper bound) because the pipeline's lower-bound surface is data sufficiency, not prediction accuracy.

- Topic: Architecture handling. Claude (Round 2) wrote "no architecture hard-coded", which Codex correctly flagged as unrealistic (compute capability and device attributes are inherently architecture-conditioned). Resolution: rephrased to "no server profiler labels and no server hardware-counter reads; architecture-conditioned device attributes (capacity, sm count, L2 size) are permitted; no fitted constants specific to a single sm_* without local validation evidence" — this lives in the cache-model spec validator (AC-6).

- Topic: Sharing-locality observability. Codex (Rounds 1-2) flagged that inter-CTA same-SM, cross-SM, and sequential-kernel reuse are NOT directly observable from NVBit traces and must be marked as projections. Claude (Round 2) accepted; Resolution: AC-7 splits the seven Shalom classes into three GROUNDED (intra-lane, intra-warp, inter-warp same-CTA) and four PROJECTED classes with mandatory provenance annotation and disagreement-variance.

- Topic: Server validation label-freedom. Codex (Round 2) flagged that "ECE" and "predicted/observed gap" presuppose labels and violate the no-server-label premise. Claude (Round 2) accepted; Resolution: AC-15 (in candidate v3) rewrote server validation as fully label-free observables (perturbation direction consistency vs sign-corrected wall-clock delta, capacity-knee alignment, repeated-run prediction stability, timing-residual enrichment, abstention-vs-residual enrichment). This validation is now deferred to the follow-up plan; the data collection it requires is implemented in this plan.

- Topic: Perturbation direction sign convention. Codex (Round 3) flagged that "predicted hit-rate delta sign matches observed wall-clock delta sign" is usually wrong because higher hit rate normally implies LOWER wall-clock time. Resolution applied at plan generation time: when this plan's collection toolchain feeds into the follow-up plan's validation, the sign convention is `sign(predicted_hit_rate_delta) == -sign(observed_runtime_delta)` for raw runtime delta, OR wall-clock delta is defined as a performance-improvement delta with explicit polarity. The data collected in this plan supports either convention; the follow-up plan picks one and documents it.

- Topic: Acceptance criteria with PENDING thresholds. Codex (Round 1-2) flagged that ACs without numeric gates are not falsifiable. Resolution: every AC in this plan with a quantitative gate has a preregistered numeric threshold (`<= 3 pp`, `<= 0.05`, `<= 200x`, `<= 16 GiB`, `<= 5%`, K=3, 100 ms cadence default, etc.). The user confirmed in Phase 6 that these are hard requirements (with starting values that may be tightened during DEC-3 follow-up). All thresholds carry DEC pointers for user override.

- Topic: Forbidden literal token in plan body. Codex (Round 2) flagged that the plan's title and body contained `ncu` and could leak into committed schema/code. Resolution: the static guard scans `.sh|.py|.cu|.cuh|.h|.cpp|.hpp|.c` only; `.md` and `.json` are out of scan scope. The plan title now uses "Counter-Free GPU Cache Observability" framing and the structured plan body uses neutral terminology, while the Source Map and original draft preserved at the bottom (markdown-only) retain the original terminology. Implementation field names use `cache_hit_rate_l2_sector`, `cache_hit_rate_l1tex_sector`, `provenance.profiler_family` for guard safety inside scanned file types.

- Topic: AC-6 negative test wording. Codex (Round 2) flagged that "misdeclared capacity produces predicted hit rate that fails AC-3 repeatability" was incoherent (AC-3 is about imported label repeatability, not model prediction correctness). Resolution: AC-6 negative tests rewritten to fail the cache-model spec validator and the per-target CI gate, never AC-3.

### Convergence Status

- Final Status: `converged`
- Rounds executed: 3 (max-rounds gate reached; final REQUIRED_CHANGES from Round 3 was a single sign-convention fix applied at generation time and now recorded under "Resolved Disagreements" above).
- All Codex REQUIRED_CHANGES through Round 3 were applied or explicitly resolved by user decision in Phase 6.

## Pending User Decisions

- DEC-3: Tightening of preregistered thresholds. Current values: label stddev <= 3 pp; HRD bootstrap CI half-width <= 0.05; NVBit slowdown <= 200x; trace volume <= 16 GiB; dropped records <= 5%; adaptive sampler K=3 windows; NVML cadence 100 ms.
  - Claude Position: Hard requirements; current values are reasonable starting points for a research toolchain.
  - Codex Position: Hard requirements with fixed numeric gates are required for falsifiability.
  - Tradeoff Summary: Tighter values increase rejection rate (fewer admissible cells, slower iteration); looser values increase admission rate but weaken downstream model fitting.
  - Decision Status: User chose "Hard requirements, with my values as starting point" in Phase 6; values may be tightened when starting RLCR.

- DEC-4: Server label-free validation thresholds (perturbation direction consistency, capacity-knee bin tolerance, repeated-run MAD, OOD enrichment lift, ECE on synthetic surrogate). Deferred to the follow-up validation plan because server validation is out of scope for this data-collection plan.
  - Claude Position: Defer; not in scope here.
  - Codex Position: Defer.
  - Tradeoff Summary: Locking thresholds now risks tying the follow-up plan to numbers the data has not yet supported; deferring lets the follow-up plan calibrate from real traces.
  - Decision Status: PENDING (resolved upon follow-up plan).

- DEC-10: OOD abstention rate target. Deferred to the follow-up validation plan.
  - Claude Position: Defer.
  - Codex Position: Defer.
  - Tradeoff Summary: Abstention rate is a model-design parameter, not a collection parameter.
  - Decision Status: PENDING (resolved upon follow-up plan).

- DEC-11: `provenance.profiler_family` identifier value (default `"nsight_compute"`).
  - Claude Position: `"nsight_compute"` as it is descriptive, matches NVIDIA's product name, contains no token currently forbidden by the static guard regex (the guard bans the bare three-letter token, not the full product name).
  - Codex Position: User must pick a guard-safe value; field name and identifier value both must satisfy the regex.
  - Tradeoff Summary: A descriptive value documents intent; a generic value future-proofs for non-NVIDIA label sources.
  - Decision Status: PENDING (default `"nsight_compute"` applied during implementation; user may change at any time without breaking the schema).

## Implementation Notes

### Code Style Requirements

- Implementation code, tests, comments, and runbook prose in `tools/`, `workloads/simbench_bw/`, `workloads/portable_lcd/`, `data/local_labels/`, and `tools/local_labels/` MUST NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Phase`, `Step`, or similar workflow markers.
- These terms exist for plan documentation only and do not belong in the resulting codebase.
- Use descriptive, domain-appropriate naming: `cache_hit_rate_l2_sector`, `warp_coalescer`, `hrd_profiler`, `perturbation_ledger`, `sharing_locality_classifier`, `adaptive_sampler`, `architecture_portability_layer`, `local_label_importer`, `repeatability_aggregator`, `bootstrap_collector`.
- The static guard at `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py` continues to enforce the server contract on `.sh|.py|.cu|.cuh|.h|.cpp|.hpp|.c` files; the new local-label code under `tools/local_labels/` and `data/local_labels/` is excluded from that scan via `EXCLUDED_PREFIXES` and gets its own scoped allowlist test.
- Schemas (`.json`) and documentation (`.md`) are not in the static guard scan scope but should still avoid plan tokens when describing the runtime contract that consumers will read.

## Knowledge Consulted

- `.claude/knowledge/measurement/nonroot-cloud-contract.md` — non-root cloud contract scope; informs the contract amendment wording and the server-side prohibition list.
- `.claude/knowledge/measurement/nvbit-injection-playbook.md` — NVBit invocation pattern, tool-combination constraint, sm_120 binary-version fallback note; informs AC-4 trace contract and AC-12 bootstrap.
- `.claude/knowledge/measurement/cupti-activity-api.md` — allowed CUPTI Activity kinds; informs AC-12 and AC-16.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — `%clock64` and `%globaltimer` are diagnostic phase signals, not cache-prediction inputs; informs the Allowed-Choices clause and DEC-9 resolution.
- `.claude/knowledge/measurement/nvarchsim-vs-this-project.md` — clarifies that this fork is non-root measurement and prediction, not simulator revival.
- `.claude/knowledge/cuda/cuda-memory-hierarchy.md` and `.claude/knowledge/best-practices/perf-coalescing-memory.md` — coalescing rules, 32 B sector, 128 B line; informs AC-5 warp coalescer.
- `.claude/knowledge/ptx/ptx-isa-cache-hints.md` — PTX cache modifier semantics for the perturbation harness (`.ca`/`.cg`/`.cs`/`.cv`, eviction priority hints, L2 prefetch hint sizes, L2 persisting access window); informs AC-10.
- `.claude/knowledge/blackwell/blackwell-tuning.md`, `.claude/knowledge/patents/arch-hopper.md`, `.claude/knowledge/patents/arch-ampere.md`, `.claude/knowledge/patents/patent-family-map.md` — per-architecture cache geometry numbers used by the cache-model spec validator default.



--- Original Design Draft Start ---

# Counter-Free GPU Cache Observability Research Plan

Date: 2026-05-03

## Purpose

This plan defines the research direction for predicting cache behavior on GPU
cloud/server environments where the process has CUDA access but does not have
admin control over the host GPU driver.

The target deployment environment has these hard constraints:

- No Nsight Compute / NCU profiling on server GPUs.
- No GPU hardware performance counters on server GPUs.
- No privileged CUPTI profiling, range profiling, PM sampling, events, or
  metrics on server GPUs.
- No `nvidia-smi` clock locking, power-limit changes, persistence-mode changes,
  MIG changes, or other host-driver mutation.
- Root inside a container does not count as host-driver privilege.

The local machines are the only labeled environments:

- Local RTX 3090 Ti: NCU is available and may be used for labels.
- Local RTX 5060 laptop: NCU is available and may be used for labels.

Server/cloud GPUs are permanently unlabeled for cache metrics. They can only be
evaluated through allowed timing, tracing, static, and read-only covariate
signals.

## Central Claim

The project should not claim that it measures cache hit ratio on server GPUs.
It should claim:

> We predict NCU-equivalent L1/TEX and L2 sector hit-rate metrics from
> non-root user-space evidence. Prediction accuracy is validated against local
> NCU labels on the 3090 Ti and 5060 laptop, then the same counter-free
> predictor is deployed on unlabeled server GPUs with label-free consistency
> checks.

This reframes the work as a non-privileged GPU cache-observability and metric
distillation method, not as another GPU cache simulator or direct cache-counter
measurement tool.

## Novelty Positioning

The crowded prior-art area is "reuse-distance modeling for GPU caches." That is
not enough for novelty by itself.

Known adjacent prior art includes:

- Nugteren et al. 2014, "A Detailed GPU Cache Model Based on Reuse Distance
  Theory": GPU reuse-distance model with warp/threadblock hierarchy, coalescing,
  associativity, MSHRs, latency, and divergence.
- RDGC 2019: reuse-distance-based GPU cache hierarchy hit-ratio and transaction
  analysis.
- PPT-GPU-Mem 2020: NVBit trace extraction and reuse profiles for L1/L2 GPU
  cache memory modeling.
- MDM 2020: NVBit-collected per-warp memory addresses, memory divergence, DPKI,
  MSHR pressure, request batching, and queueing effects.
- MLCRP 2025: ML-based GPU cache performance modeling with reuse profiles.
- Sieve, STEM, GCL-Sampler, and related sampled GPU simulation work: trace and
  kernel sampling, clustering, and learned kernel similarity.

The defensible novelty is the combination of the following:

1. NCU metric distillation under a deployment environment where server labels
   never exist.
2. Prediction of NCU-equivalent sector hit-rate metrics rather than an abstract
   cache hit ratio.
3. Warp-coalesced, conditional hierarchical reuse distance adapted from HRD.
4. GPU-specific sharing locality adapted from Shalom.
5. Condens-style adaptive NVBit trace acquisition for locality proxies.
6. Timing-only cache-behavior tomography on unlabeled server GPUs.
7. Explicit uncertainty and domain-shift reporting for cloud/server deployment.

## Research Questions

RQ1. Can local NCU L1/TEX and L2 sector hit-rate labels be accurately predicted
from only signals that are also available in non-root cloud environments?

RQ2. Does conditional hierarchical reuse distance improve prediction over flat
reuse-distance profiles and simple coalescing features?

RQ3. Does GPU sharing-locality decomposition improve L2 prediction over
per-warp or per-CTA reuse alone?

RQ4. Can cache-policy and working-set perturbations provide enough timing-only
constraints to retune the model on unlabeled server GPUs?

RQ5. Can uncertainty estimates identify when predictions on server GPUs are
out-of-domain relative to local 3090 Ti and 5060 laptop calibration?

RQ6. Does adding predicted cache behavior improve wall-clock prediction beyond
static SASS, launch metadata, CUDA event timing, and CUPTI Activity timing
alone?

## Allowed Evidence Surface

### Local and Server

These signals are allowed on both local and server GPUs:

- CUDA events for elapsed time.
- CUPTI Activity API records for kernel, concurrent-kernel, memcpy, memset,
  runtime, driver, overhead, device, and context activity where available.
- NVBit SASS instrumentation in separate matched runs.
- Static SASS/PTX from `cuobjdump`, `nvdisasm`, `nvcc`, and `ptxas -v`.
- Inline `%clock64` or `%globaltimer` only as diagnostic phase signals with
  caveats.
- Read-only NVML covariates such as GPU name, driver version, temperature,
  current clocks, power draw, power limits, P-state, and throttle reasons.
- Read-only `nvidia-smi` only as fallback/debug. Prefer NVML for production
  covariates because `nvidia-smi` text output is not a stable API.

### Local Only

NCU is allowed only on local GPUs to produce labels:

- RTX 3090 Ti.
- RTX 5060 laptop.

NCU outputs must be treated as labels for model validation and distillation,
not as part of the server deployment workflow.

### Forbidden in Server Deployment

- NCU / Nsight Compute.
- CUPTI Events and Metrics.
- CUPTI Profiling API.
- CUPTI Range Profiler.
- CUPTI PM Sampling.
- CUPTI SASS Metrics API.
- CUPTI PC Sampling as a required baseline. It may be explored only as an
  optional availability probe, but the core method must not depend on it.
- Nsys GPU Metrics.
- Clock locking or power-limit mutation.
- Any host driver or kernel-module setting change.

## Metric Target

The target is not "cache hit ratio" in the generic sense. The target is:

- NCU-equivalent L1/TEX sector hit rate.
- NCU-equivalent L2 sector hit rate.
- Optional supporting outputs:
  - sector/request pressure,
  - sector miss pressure,
  - coalescing efficiency,
  - memory-divergence class,
  - MSHR-pressure proxy,
  - prediction uncertainty.

Nsight Compute's memory workload model treats a sector as an aligned 32-byte
chunk. L1 and L2 cache lines are four sectors, i.e. 128 bytes. L1 sector hit
rate is not the same as tag hit rate. The model should therefore be trained and
reported in terms of NCU-equivalent sector accounting.

## High-Level Architecture

The system should be organized as a two-level model:

```text
NVBit/static features -> architecture-invariant cache evidence
timing perturbations + microbenchmarks -> device latent parameters
local NCU labels -> metric distillation target
```

At deployment time on server GPUs:

```text
server allowed evidence + local-trained model + timing retuning
    -> predicted NCU-equivalent sector hit rates + uncertainty
```

No server NCU labels are assumed or required.

## Core Method Components

### 1. NCU Metric Distillation

Treat NCU as a local teacher and train a student model using only features that
are available without privileged counters.

Teacher labels:

- L1/TEX sector hit-rate labels.
- L2 sector hit-rate labels.
- Optional memory workload metrics if available locally and stable across NCU
  versions.

Student inputs:

- Static SASS/PTX features.
- NVBit memory-reference features.
- Conditional HRD features.
- GPU sharing-locality features.
- Timing perturbation features.
- Timing-only microbenchmark retuning features.
- Read-only device covariates.

Student outputs:

- predicted L1/TEX sector hit rate,
- predicted L2 sector hit rate,
- confidence interval,
- out-of-domain score.

### 2. Warp-Coalesced Conditional HRD

Use Maeda's hierarchical reuse distance as a conditional hierarchy, not as
independent flat reuse-distance histograms.

The proposed hierarchy:

1. 32-byte sector.
2. 128-byte line/fill group.
3. allocation region or virtual page bucket.
4. optional larger region bucket for streaming/kernels with regular tiling.

The key rule is conditional:

- Profile reuse at the finest granularity first.
- Only when the finer layer sees infinity/cold reuse does the model consult the
  next coarser layer.

This preserves fine-grained sector locality while retaining larger locality
information useful for L2 and allocation-level behavior.

Before HRD, lane addresses must be converted into coalesced warp-level memory
requests:

- apply active/predicate mask,
- group touched 32-byte sectors,
- group sectors into 128-byte line/fill groups,
- record sectors per request,
- record uniform/broadcast/overlap cases,
- record strided/scattered cases.

### 3. Schedule-Marginalized Reuse Ordering

Reuse distance needs an ordered stream. GPU execution order is not uniquely
observable under non-root tracing and can vary by scheduling, residency, and
latency.

Instead of assuming one order, compute several plausible order projections:

- lane-order projection,
- warp-local projection,
- CTA-local projection,
- SM-local phase projection using NVBit CTA/warp/SM identity where available,
- kernel-level L2 projection,
- sequential-kernel allocation projection.

The model should learn from a distribution or feature bundle over these order
projections. The final prediction should include uncertainty when projections
disagree.

### 4. GPU-Shalom Sharing Locality

Adapt Jiang/Shalom conceptually, but do not reuse CPU thread-block-section
fractions.

GPU sharing classes:

- intra-lane or same-lane reuse,
- intra-warp reuse,
- inter-warp same-CTA reuse,
- inter-CTA same-SM reuse,
- inter-CTA cross-SM reuse visible mostly through L2,
- sequential-kernel reuse on persistent allocations,
- private/cold accesses.

For L1/TEX prediction, same-SM and same-CTA locality should matter most. For L2
prediction, inter-CTA and sequential-kernel allocation reuse should matter more.

Required trace metadata:

- kernel identity,
- launch index,
- CTA coordinates,
- warp id,
- SM id where available,
- function-relative SASS PC,
- memory op type,
- lane addresses,
- active/predicate masks,
- allocation/range identity.

### 5. Memory-Divergence And MSHR-Pressure Proxies

MSHR counts and replacement policy are not directly observable in server
deployment. Use proxies:

- unique 32-byte sectors per warp memory instruction,
- unique 128-byte lines per warp memory instruction,
- sectors/request,
- memory divergent loads per kilo instruction,
- outstanding load density by phase,
- PC-level repeated miss-likely streams,
- stride entropy,
- pointer-chase timing sensitivity,
- throughput saturation under increasing resident warps,
- timing response to thrash/working-set sweeps.

These proxies connect HRD hit-rate prediction to wall-clock impact. The same
sector hit rate can produce different wall-clock behavior depending on
divergence, request batching, and queueing.

### 6. Causal Cache Tomography

On server GPUs, no NCU labels exist. Use matched perturbations to constrain
latent cache behavior through timing deltas.

Candidate perturbations:

- `.ca` vs `.cg`: expose L1 sensitivity by bypassing/preventing normal L1 use
  where the compiler/SASS honors the policy.
- `.cs`: expose streaming/pollution behavior.
- `.cv`: use carefully; do not assume universal device-memory bypass semantics
  without local validation.
- L2 prefetch-size hints: `.L2::64B`, `.L2::128B`, `.L2::256B` where supported.
- L2 cache hints and eviction-priority policy where supported.
- L2 persisting access-policy window on compute capability 8.0+ where not
  disabled by MIG/MPS constraints.
- shared-memory/L1 carveout as a driver hint, not a deterministic control.
- working-set size sweeps for capacity knees.
- stride and misalignment sweeps for sector amplification.
- warm/cold paired runs.
- eviction-buffer sweeps.

Important interpretation rules:

- PTX cache operators are performance hints, not semantic controls.
- L2 `hitRatio` is not a measured hit rate. It is a policy probability/hint.
- `discard.global.L2` is destructive and must not be used as a benign eviction
  primitive for live application data.
- Perturbation deltas are constraints on latent cache behavior, not labels.

### 7. Condens-Style Adaptive NVBit Sampling

Full NVBit memory tracing can be very expensive and can perturb cache behavior.
Use adaptive trace acquisition:

1. Start with small trace windows.
2. Build HRD/MRC estimates for sector and line levels.
3. Continue tracing until the chosen locality proxy converges.
4. Stop when max change across selected MRC quantiles is below a threshold for
   several consecutive windows.
5. Report trace convergence confidence.

This adapts Condens from statistical simulation into evidence acquisition.
The convergence guarantee applies only to the chosen HRD/MRC proxy, not to
wall-clock or NCU labels directly.

### 8. Server Domain Adaptation

Server GPUs are unlabeled. Adapt only latent device parameters using allowed
signals:

- CUDA device attributes,
- compute capability,
- static SASS features,
- timing microbenchmark results,
- cache-perturbation deltas,
- read-only NVML covariates,
- observed noise level and thermal/clocks state.

Do not hard-code one architecture. Treat cache geometry, effective capacity,
replacement behavior, L1/shared carveout behavior, and L2 policy behavior as
latent or probed parameters.

The 5060 laptop should be probed at runtime for compute capability and
available features rather than assumed from product name.

## Experimental Design

### Local Labeled Experiments

Use both local devices as labeled domains:

- RTX 3090 Ti.
- RTX 5060 laptop.

Core splits:

1. Train on 3090 Ti, test on 5060.
2. Train on 5060, test on 3090 Ti.
3. Train on both, hold out kernels.
4. Train on both, hold out shapes.
5. Train on both, hold out perturbation families.
6. Train on both, hold out memory-pattern classes.

Local NCU is label-only. The deployment feature set must not include any NCU
metric. The model must be able to run without NCU.

### Server Unlabeled Experiments

On server/cloud GPUs:

- collect CUDA event timing,
- collect CUPTI Activity timing,
- collect NVBit evidence in separate matched runs,
- collect static SASS/PTX,
- collect read-only NVML covariates,
- run microbenchmark retuning,
- run cache-tomography perturbations.

No server NCU accuracy claim should be made because no server labels exist.

Server evaluation should report:

- prediction outputs,
- uncertainty intervals,
- domain-shift score,
- timing consistency,
- perturbation consistency,
- cache-feature contribution to wall-clock prediction,
- stability under repeated runs.

## Validation Metrics

### Local Labeled Metrics

For NCU-equivalent sector hit-rate prediction:

- MAE for L1/TEX sector hit rate.
- RMSE for L1/TEX sector hit rate.
- MAE for L2 sector hit rate.
- RMSE for L2 sector hit rate.
- calibration error for confidence intervals.
- Spearman/Pearson correlation across kernel variants.
- rank accuracy across optimization variants.
- delta-sign accuracy under perturbations.

For wall-clock linkage:

- timing prediction MAE/RMSE,
- improvement over a static-only baseline,
- improvement over a timing-only baseline,
- improvement over flat reuse-distance baseline,
- improvement over non-hierarchical coalescing baseline.

### Server Unlabeled Metrics

For unlabeled deployment:

- perturbation direction consistency,
- capacity-knee alignment with timing sweeps,
- `.ca`/`.cg` L1-sensitivity consistency,
- sector-pressure consistency with stride/misalignment,
- repeated-run stability,
- uncertainty-width response to noisy or out-of-domain settings,
- wall-clock prediction improvement from cache features.

## Baselines

Required baselines:

- Static SASS + launch metadata only.
- Timing-only microbenchmark features only.
- Coalescing features only.
- Flat reuse distance at 32B sector only.
- Flat reuse distance at 128B line only.
- Independent multi-granularity RD without conditional HRD.
- HRD without GPU-Shalom sharing classes.
- HRD without cache-tomography perturbation features.
- Full proposed model.

If feasible, compare conceptually or experimentally to:

- Nugteren-style GPU RD features.
- PPT-GPU-Mem-style reuse profile features.
- MDM-style memory-divergence features.
- ML model without analytical HRD structure.

## Artifacts To Build

### Data Schema

Create explicit artifact schemas for:

- local NCU label sidecar,
- NVBit cache evidence sidecar,
- HRD feature sidecar,
- GPU-Shalom sharing sidecar,
- perturbation-run ledger,
- timing microbenchmark ledger,
- prediction output with uncertainty,
- server deployment report.

### NVBit Enhancements

Needed fields:

- function-relative PC,
- unique function id,
- SASS opcode string,
- parsed memory op class,
- parsed cache policy if visible,
- active mask,
- predicate mask,
- lane address vector,
- CTA coordinates,
- warp id,
- SM id if available,
- launch id,
- phase id,
- allocation/range id.

Cache policy should be a derived parsed static-SASS/PTX feature, not assumed to
be a first-class NVBit output.

### Feature Extractors

Implement:

- warp coalescer,
- sector stream generator,
- conditional HRD profiler,
- memory reference curve generator,
- sharing-locality classifier,
- divergence/MSHR-pressure proxy extractor,
- perturbation feature extractor,
- domain-shift feature extractor.

### Microbenchmarks

Build or standardize:

- pointer chase for latency bands,
- working-set sweep for capacity knees,
- stride/misalignment sweep for sector amplification,
- warm/cold cache pair,
- `.ca`/`.cg`/`.cs` variants,
- L2 persisting-window sweep where supported,
- carveout sensitivity probe,
- bandwidth saturation sweep,
- resident-warp pressure sweep.

## Expected Risks And Mitigations

### Risk: Prior Art Crowding

Mitigation:

- Do not claim generic GPU reuse-distance novelty.
- Emphasize non-root, permanently unlabeled server deployment.
- Emphasize NCU-equivalent sector metric distillation.
- Emphasize cache tomography and uncertainty-aware adaptation.

### Risk: Local Devices Do Not Represent Server GPUs

Mitigation:

- Use 3090 Ti and 5060 as two labeled domains.
- Report cross-device transfer explicitly.
- Use server microbenchmarks to retune latent device parameters.
- Report uncertainty and domain-shift scores.

### Risk: NVBit Perturbs Cache Behavior

Mitigation:

- Separate NVBit trace runs from native timing runs.
- Use adaptive trace sampling.
- Never use NVBit-instrumented wall-clock as application ground truth.
- Join runs offline by kernel signature, launch id, binary hash, and covariates.

### Risk: Cache Hints Are Not Deterministic

Mitigation:

- Treat cache operators and L2 policies as perturbation levers.
- Validate perturbation direction locally against NCU labels.
- Use timing deltas as weak constraints, not labels.

### Risk: Clocks And Thermals Are Uncontrolled

Mitigation:

- Do not lock clocks in the deployment-like path.
- Use repeated runs, medians, quantiles, and robust regression.
- Record read-only NVML covariates.
- Include noise level in uncertainty model.

### Risk: Server Labels Never Exist

Mitigation:

- Be explicit: server evaluation is label-free.
- Validate accuracy locally.
- Validate deployment by metamorphic/perturbation consistency and wall-clock
  prediction improvement.

## Proposed Paper Contribution Wording

Contribution 1:

> A counter-free cache-observability formulation that predicts
> NCU-equivalent L1/TEX and L2 sector hit-rate metrics from signals available
> in non-root GPU cloud environments.

Contribution 2:

> A warp-coalesced conditional HRD model with GPU sharing-locality classes that
> captures sector, line, CTA, SM, L2, and sequential-kernel reuse without
> hardware counters.

Contribution 3:

> A cache-tomography protocol using timing-only cache-policy and working-set
> perturbations to adapt a locally calibrated model to unlabeled server GPUs.

Contribution 4:

> An adaptive NVBit acquisition method that stops tracing when locality proxies
> converge, reducing instrumentation overhead while reporting uncertainty.

Contribution 5:

> A validation methodology that separates local NCU label accuracy from
> server-GPU unlabeled deployment consistency.

## Near-Term Milestones

1. Define exact NCU label set on the local 3090 Ti and 5060 laptop.
2. Freeze the deployment feature contract: no feature may require NCU or
   hardware counters.
3. Implement local NCU label sidecar.
4. Implement NVBit cache evidence extraction or extend the current tracer.
5. Implement warp coalescing and sector stream generation.
6. Implement conditional HRD for sector/line/allocation hierarchy.
7. Implement GPU-Shalom sharing classification.
8. Implement first microbenchmark retuning suite.
9. Run a pilot on 3090 Ti: NCU labels vs counter-free predictions.
10. Run cross-device pilot: train 3090 Ti -> test 5060, and reverse.
11. Run a server unlabeled pilot with perturbation consistency checks.
12. Write a novelty comparison table against Nugteren, RDGC, PPT-GPU-Mem, MDM,
    MLCRP, Sieve/STEM/GCL-Sampler, and cuThermo.

## Source Map

Local primary target papers:

- `references/papers/knowledge_base/inbox/maeda2017.pdf`
- `references/papers/knowledge_base/inbox/jiang2022.pdf`

Local related papers:

- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`
- `references/papers/knowledge_base/active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf`
- `references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf`
- `references/papers/knowledge_base/active/2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking.pdf`
- `references/papers/knowledge_base/active/2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking.pdf`
- `references/papers/knowledge_base/active/2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture.pdf`
- `references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf`

Useful local knowledge cards:

- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/cupti-activity-api.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `.claude/knowledge/cuda/cuda-memory-hierarchy.md`
- `.claude/knowledge/best-practices/perf-coalescing-memory.md`
- `.claude/knowledge/ptx/ptx-isa-cache-hints.md`

External references:

- NVIDIA ERR_NVGPUCTRPERM:
  https://developer.nvidia.com/ERR_NVGPUCTRPERM
- CUPTI documentation:
  https://docs.nvidia.com/cupti/
- CUPTI Activity API:
  https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
- CUDA Runtime API events:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVBit GitHub:
  https://github.com/NVlabs/NVBit
- NVBit research page:
  https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus
- PPT-GPU-Mem:
  https://laro.lanl.gov/esploro/outputs/conferenceProceeding/Fast-accurate-and-scalable-memory-modeling/9916369267303761
- RDGC:
  https://www.cai.sk/ojs/index.php/cai/article/view/2019_2_421
- GCL-Sampler:
  https://www.researchgate.net/publication/401470700_GCL-Sampler_Discovering_Kernel_Similarity_for_Sampled_GPU_Simulation_via_Graph_Contrastive_Learning


--- Original Design Draft End ---
