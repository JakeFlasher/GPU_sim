# Comprehensive sm_86 Live Data-Collection Campaign (RTX 3090 Ti) — Contract-First

## Goal Description

Scale the sm_86 ground-truth data-collection campaign from the 17-cell `portable_lcd` pilot to the **full sm_86-runnable roster** on the RTX 3090 Ti, producing the frozen corpus the event-driven simulator (Phase 2 / M2 and beyond) consumes: NCU labels (full metric vector), NVBit traces (the sim input), roster-wide SASS control-bits, calibration cards (pchase + new additive microbenches), per-metric repeatability/determinism floors, and a defensible typed-exclusion ledger.

The organizing principle is **contract-transition-first**: land the tracer ASYNC_COPY+ATOM address-exclusion flip as ONE fully-lockstepped, capability-versioned trace-contract transition (both flips in the same transition), keep the frozen M1 bundle as the pre-flip regression anchor, then capture the whole roster **exactly once** under the new address-carrying contract in immutable one-commit / one-roster / one-protocol / one-manifest chunks.

This is a **DATA-COLLECTION campaign only**. Building the M2 eval-join, per-metric reporter, and baseline rigs is explicitly out of scope. No edits to existing validated code except (a) genuine bug fixes, (b) the sanctioned tracer flip plus its lockstep consumer widening, and (c) the sanctioned `freeze_label_bundle.py` widening (DEC-3); new files (roster JSONs, additive microbench probes, campaign-manifest + exclusion-ledger tools/schemas, SOL shape-adjust wrappers) are allowed.

## Capture readiness predicates (Q805)

Readiness is four distinct, non-promoting predicates. Each predicate is the
conjunction below; satisfying one never establishes the next one. In
particular, an offline-green byte set is not evidence that a clean capture
commit or a hardware-ready host exists.

- **P1 `offline-source-ready(bytes)`** is true only when the repository default
  verification commands and the capture-focused Python battery are green for
  the exact tested bytes; the battery explicitly deselects exactly
  `tools/local_labels/test_campaign_preflight.py::test_environment_probes_ok_on_reference_host`
  and no other test; and the frozen-v1, durable-v2, and matrix identity-closure
  checks all pass. P1 is byte-scoped and does not assert a clean commit, live
  hardware, or admissible capture output.
- **P2 `clean-S-ready(S)`** is true only when P1 holds for the bytes at a clean
  `HEAD == S`, the operator-clean-S producer emits live-v1 at the exact
  `inputs/r28_<S8>` path after verifying all 122 frozen runnable identities,
  producer start/authority/final HEAD all equal S, and the successor matrix is
  built from that live authority at S using the active registry. P1 alone does
  not imply P2.
- **P3 `live-canary-ready(host,S)`** is true only when P2 holds and the sole
  hardware probe passes live on that host, a capture-admitted preflight
  manifest is no more than 900 seconds old, the on-device injection smoke is
  nonempty and contract-valid, and all three P0 clock samples match the pinned
  persistence/1350 MHz SM/10501 MHz memory state. Collecting a probe artifact
  without a passing verdict does not satisfy P3; P2 alone does not imply P3.
- **P4 `campaign-ready`** is true only when P3 holds immediately before
  dispatch, the campaign lock is free, no storage-fault receipt exists, and
  retained/staging budget headroom is sufficient. Supplemental dispatch also
  requires the pinned live canary and closure of the launch-driver bridge and
  NCU-exporter authority amendments; until then it remains prohibited. P3
  alone does not imply P4.

The authority transition is immutable frozen v1 → host-independent durable v2
→ clean-S live v1 → noncanonical successor matrix. Later stages reverify their
predecessors and never rewrite frozen bytes. Missing evidence leaves the
corresponding predicate false/pending; it never inherits truth from an earlier
predicate.

Clock control remains an operator-only privilege boundary. Automation must
never run `sudo`; the operator commands are `sudo nvidia-smi -pm 1`,
`sudo nvidia-smi -lgc 1350`, and `sudo nvidia-smi -lmc 10501`, with mandatory
restore commands `sudo nvidia-smi -rgc` and `sudo nvidia-smi -rmc`.

**Status at this plan revision:** P1 is established only for exact tracked bytes
at commit `80aec8413a856587b24bedcdf3464d90c5d3f39f` (tree
`d538b0c265ffeb0a6bde4d1d341f5ae8772c2998`). The repository-default Rust
formatting and workspace-test gates passed, default Python reported 4241 passed,
54 skipped, 42 deselected, and 505 subtests passed, and SimBench validation
reported `ok`; the exact six-file focused battery reported 756 passed and
exactly the named live-host probe deselected; the independent hardware-marker
collection selected only that probe; and the frozen-v1, durable-v2, and
canonical-matrix schemas, authority hashes, 344-cell identity sets, 122/222
partition, and matrix seal closed in a read-only audit. This byte-scoped result does not transfer to later bytes. P2 remains false because no current
clean-S live-v1/successor output set has been admitted. P3 and P4 remain typed
pending external and therefore false. There is no new hardware credit, and no
hardware or campaign-readiness claim follows from P1.

## Acceptance Criteria

The `AC-*` items are the current RLCR completion gates. All accuracy/coverage claims are against real-GPU NCU under the pinned protocol v2.1; NCU values are targets/validation only and never enter model or card inputs (Rule 1).

- AC-1: **Preflight, provenance foundation, and roster authority are established before any capture.** A preflight emitter records host + toolchain + clock state and gates capture; a machine-checkable roster authority enumerates the universe, the runnable set, and the excluded set; the SOL-ExecBench environment is reconstituted enough to determine its runnable count on this host, with the 23 host-capacity cells shrunk to a fitting shape (DEC-2) so they are runnable, not deferred.
  - Positive Tests (expected to PASS):
    - Preflight manifest records git HEAD, `working_tree_clean`, NCU/driver/nvcc/NVBit versions matching the pins (NCU 2026.1.0 / driver 610.43.02 / nvcc 13.2 / NVBit 1.8), clocks == v2.1 (core 1350 / mem 10501, persistence on), a `LD_LIBRARY_PATH=.toolchain/pb/lib` injection smoke that yields a nonzero contract-valid trace, and declared free disk > the authored trace-retention budget.
    - The roster authority manifest resolves to a total universe count, a runnable-set count, and an excluded-set count with `universe == runnable + excluded`, each row carrying `roster_source`, binary, app-args, shape selector, and sha256 pins.
    - SOL-ExecBench reconstitution reproduces a recorded runnable/deferred split on this host (data present via `download_data.sh`, env via `uv sync`, CUTLASS via `CUTLASS_DIR` override) fingerprinted by a native-env hash; each of the 23 formerly-OOM cells is first re-checked under `--replay-mode application` (a cell that then fits natively is stamped `native_captured_via_replay_mode`, no shrink), else shrunk per the Fable-5 rubric (`plans/sm86_sol_shrink_to_fit_fable5_answer.md`) — the footprint/roofline/L2-working-set pre-check picks the largest regime-preserving shape, classified `shape_adjusted` / `reduced_representative` / `unavailable::oom_shape_gated`, stamped with the shrink vector, the L2-side + roofline-class verdicts, and a native-shape reference.
  - Negative Tests (expected to FAIL):
    - A dirty unrecorded worktree, unlocked host clocks for the reference config, a missing `.toolchain/pb` runtime, or a missing trace-budget file causes preflight to fail BEFORE any capture runs.
    - A roster authority where any runnable cell lacks a resolvable binary/shape/app-args, or where `universe != runnable + excluded`, fails.
    - A `shape_adjusted` SOL cell that does not record its native-shape reference, or that still OOMs at the reduced shape without a typed reason, fails; a shrink that cuts attention B·H below CTA saturation (dropping the K/V panel set under the 6 MiB L2 → silent `lts` hit-rate inflation) without a regime flag fails; a `shape_adjusted` cell whose post-capture `dram__bytes.sum` deviates > ±30% from its footprint model without demotion to `reduced_representative` fails.

- AC-2: **The frozen M1 bundle is a green pre-flip regression anchor.** `eval/frozen_labels/sm86_m1_bundle_v21.json` validates and replays under old-contract semantics, byte-identical, before the flip.
  - Positive Tests: replay/validate the frozen bundle → 15 admitted / 2 typed-excluded cells; `smsp__inst_executed.sum` EXACT 15/15; L1 op_ld sector-hit-rate label-exact; all bundle sha256 pins reproduce.
  - Negative Tests: any new validation failure, changed admitted/excluded count, or sha256 mismatch on the frozen bundle fails (guards against accidental pre-flip drift or a premature contract change).

- AC-3: **The tracer trace-contract transition lands as ONE lockstepped, capability-versioned commit with consumer widening and synthetic tests — semantically correct for atomics vs async-copy, with BOTH flips included.**
  - AC-3.1: The producer flip stops the host-side drop of the addressed classes and stamps a manifest-level `trace_contract_id` plus a producer capability list (`memory_payload_classes`); the SASS classifier is untouched; the proto is regenerated and re-pinned in the same commit.
  - AC-3.2: Consumers are widened in lockstep to ENFORCE identity, not merely decode the enum: `import_trace_contract` extends its global-class allocation-identity gate to address-bearing atomic/async-copy records; `tools/sector_stream/generator.py`, HRD, `order_projections`, and the cache-access policy are aligned to the new payload classes; the Rust `sim/trace` reader enforces the same identity the Python gate does.
  - AC-3.3: Synthetic producer fixtures cover ordinary global load, **global atomic**, **shared atomic (ATOMS)**, **RED**, **LDGSTS (cp.async)**, and predicated-off memory records. The ATOMIC flip is implemented and tested first WITHIN this single contract transition; the ASYNC_COPY (LDGSTS) flip is adjudicated (source/destination operand semantics proven), widened, tested, and included in the SAME contract transition before the transition is complete. "Atomic-first" is an internal implementation ordering only — never "atomic-only now, async later"; the loop does not complete and full-roster capture (AC-5) does not start until BOTH flips are in the contract.
  - Positive Tests: a new-contract trace with an active global atomic or LDGSTS-source record carries `allocation_id` + `memory_op_class` + `data_width`; a shared-atomic (ATOMS) record is classified without being forced through the global allocation-id rule; the frozen M1 traces still validate with the capability absent.
  - Negative Tests: a new-contract active global-atomic / LDGSTS-source record with `allocation_id` absent or zero is REJECTED; forcing a shared atomic through global allocation identity is REJECTED; a `.proto` change committed without its pin regeneration fails `cargo test`.

- AC-4: **A new-contract canary proves the flip unlocked scorability without regressing exact metrics.** A small canary set (ordinary load, global atomic, shared-atomic/RED, LDGSTS) is captured under the new contract.
  - Positive Tests: the canary manifest shows the new `trace_contract_id` + capability list, nonzero admitted atomic/async records where expected, and no unexpected `unsupported_memory_path` rows; a previously-`na` tensor-core op_ld cell (e.g. `gemm_hmma_bf16`) and the atomic histogram L2 cell become scorable; `smsp__inst_executed.sum` and L1 op_ld exactness on the pre-existing cells are unchanged.
  - Negative Tests: a canary cell that still emits address-stripped `unsupported_memory_path` for the flipped classes fails; any regression in `smsp__inst_executed.sum` or L1 op_ld exactness on the pre-existing cells fails.

- AC-5: **The full runnable roster is captured once under the new contract at the v2.1 reference protocol** — traces + full-metric-vector labels + control-bits — with resume/idempotence, per-chunk provenance, and typed sampling for budget-breaching cells.
  - Positive Tests: for every runnable cell, admitted outputs contain a full-registry NCU label sidecar (the sm_86-available metric set), a contract-valid NVBit trace, a control-bits sidecar, a binary-sha256 join, and a v2.1 protocol stamp; the roster-authority count equals admitted + typed-excluded; re-running a completed chunk skips cells whose sidecar hashes already match; each immutable chunk emits a DONE-manifest (command, source_commit, dirty-state, tool versions, protocol, roster hash, budget hash, output root, pass/fail counts, typed-failure rows) and passes a pre/post source-commit equality check; a cell that breaches the trace budget is handled per the AC-9 sampling ladder (compression-first, admissibility-gated, wave-segment/reduced-shape, typed) — never silently truncated.
  - Negative Tests: any runnable cell present in NEITHER admitted outputs NOR the exclusion ledger fails the phase; a cell whose repeats span differing `source_commit` (mid-sweep commit → `label_provenance_duplicate`) fails; a budget-breaching cell captured without the AC-9 sampling ladder and its typed provenance fails; an incomplete trace (block coverage < 100% vs the launch `gridDim`) accepted without a `trace_sampling` sidecar fails.

- AC-6: **Calibration cards are refreshed/extended from the Fable-5-designed six-probe clock64 suite, Rule-1 clean, overdetermined, and stored as additive evidence this loop** (designs: `plans/sm86_calibration_microbench_design_fable5_answer.md` for (a)–(e) + `plans/sm86_cp_async_timing_probe_fable5_answer.md` for (f)). pchase is re-run under the lock; the suite is built in **load-bearing order (b) port-width → (f) cp.async/LDGSTS timing → (a) L2↔DRAM fetch-granularity → (d) MSHR/MLP saturation → (c) achievable DRAM bandwidth → (e) L2 associativity/slice-hash**, at **execution order b, a, d, f, c, e** (f consumes (b)'s port + (d)'s MSHR verdicts, so it runs after both); (b)+(d) run as a Little's-law-coupled pair, and (f) = the F-L completion-latency + F-T streaming-plateau + F-B burst-issue sub-probes; control-bits are extended roster-wide. The sim `LDGSTS` event-model that consumes (f)'s constants (f1–f5) and the read-only attribution pass that confirms (f)'s provisional rank are downstream (FUT-6).
  - Positive Tests: each probe emits a measured value OR a typed bounded gap from the taxonomy {`RANGE`, `CENSORED`, `MIXTURE`, `UNRESOLVED_STRUCTURE`, `SURFACE_GAP`} carrying its full evidence curve + in-run positive/negative controls + confound verdicts, written as ADDITIVE evidence files (not transcribed into the byte-pinned `arch_profiles/cards/sm_86.json` this loop, per DEC-3); every card constant is **overdetermined by ≥ 2 independent probe readouts that must agree under the hardware hypothesis** (Little's-law triangulation (b)×(d), RTT-contrast, persistence/Δ-contrast); control-bits sidecars exist for every runnable binary (0 unknown_encoding, or typed); a Rule-1 guard proves no NCU metric from the same cell entered any card-input derivation.
  - Negative Tests: any card input traceable to an NCU metric of the same cell fails the Rule-1 guard; a microbench that produces neither a value nor a typed bounded gap (a silent guess) fails; a card constant derived from a SINGLE readout or tuned against the simulator's end-to-end output fails the overdetermination rule (cross-probe contradictions are recorded as typed contradictions, never averaged away); an edit to `arch_profiles/cards/sm_86.json` transcribing a microbench value this loop fails the DEC-3 boundary check.

- AC-7: **The full cross-protocol repeatability matrix is captured and analyzed by the Fable-5 uncensored determinism-study design** (full design: `plans/sm86_determinism_study_stats_fable5_answer.md`). Cells × {clock none/base/boost} × {cache all/none}, at **N=20 for the reference config (`clock=none`×`cache=all`) / N=10 for the five non-reference configs / escalate the reference to N=30** on a pre-registered trigger (modality/outlier flag, boundary-proximity, any out-of-range repeat, discrete-with-≥2-values); repeats collected as R shuffled sweeps, one fresh process per profile.
  - Positive Tests: the accounting identity `collected == classified + excluded_typed` holds and the study reproduces the historical admission set bit-for-bit via the imported `dispersion_gate` (the admission gate is demoted to a `would_have_been_admitted` reporting flag, not a filter); a NAMED uncensored `determinism_study` artifact (distinct from the admitted-only `matrix_summary.json`) classifies every `(cell,metric,config)` into the priority-ordered taxonomy {`out_of_range`/`vendor_defect`, `insufficient_evidence`, `multimodal`, `deterministic`, `quasi_stable`, `clock_noise`, `noisy`, `unstable`} via `sigma_rel` (d₂(N)-normalized for range rules, unit-asserted); pct-overflow rows are retained-typed (`ncu_pct_overflow`, never clamped/imputed) with per-(metric,config) OOR rate reported and the host-locked reference confirmed r=0; the two-stage small-N modality detector flags at tier-1 and certifies only at N=30; accuracy-gate denominators are the pooled P90 floor + per-cell override with the per-class rules; cross-protocol sensitivity is reported on the two SEPARATE axes (clock=instrument / cache=state) by equivalence testing; a cost/resume gate lets the sweep stop and resume without recomputing completed chunks.
  - Negative Tests: any admitted row produced by clipping or normalizing an out-of-range NCU percentage fails; an adaptive stop-when-CV-stable rule used for N (optional-stopping bias) fails; back-to-back same-process (non-shuffled) repeat collection fails the run-order design; a cell dropped without a typed exclusion reason (breaking the accounting identity) fails; a single-support outlier classified as `multimodal` fails; the classification statistic diverging from the imported `dispersion_gate` (a re-introduced scale bug) fails the lockstep property test.

- AC-8: **The full-roster corpus is frozen with a complete, defensible provenance and exclusion ledger.**
  - Positive Tests: the sanctioned widened freezer (DEC-3) emits, per admitted cell, every sm_86-available registered metric (mean + dispersion + repeatability rule), plus trace/control-bit/card-evidence manifests, source/protocol/tool hashes, and the `trace_contract_id`; a single typed-exclusion authority ledger classifies every non-captured cell with a reason ∈ {`arch_gate`, `semantic_mismatch`, `trace_unsupported`, `repeatability_failed`, `correctness_failed`, `calibration_inconclusive`, `host_capacity_deferred`} and satisfies `denominator == universe − Σ(typed exclusions)`; `shape_adjusted` is recorded as capture provenance on runnable cells, never as an exclusion reason; the freezer's `binary_sha256` check uses equality (the substring-idiom bug is fixed); the frozen bundle reports zero unresolved `problems`.
  - Negative Tests: a bundle with a cell in neither admitted outputs nor the ledger fails; a ledger row with a null/undefined `typed_reason` fails; a freezer that admits a cell whose label and trace `binary_sha256` differ (but one contains the other as a substring) fails.

- AC-9: **Trace-sampling for budget-breaching cells is compression-first, gate-routed, honest-by-construction, and golden-set-validated** (detailed design: `plans/sm86_trace_sampling_strategy_fable5_design.md`).
  - Positive Tests: lossless zstd compression + a compressibility probe are applied FIRST (contract hashes over decompressed bytes) and clear every cell that fits after compression; each still-breaching cell is routed by a deterministic admissibility-gate classifier (`coop_gate`/`wave_gate`/`atomic_gate`/`skew_gate`/`homogeneity`) to exactly one typed method with recorded gate verdicts; a wave-segment stratified-subset cell carries a valid closed-schema `trace_sampling.json` sidecar (method, gates, wave_model, strata, selected block IDs, estimator mode, `not_recoverable` list, `label_pairing`, tracer/stats-version pins) and HT calibrated-ratio estimates with stratified-cluster-bootstrap CIs meeting the stopping rule (rates ≤ 0.02 absolute half-width, counts ≤ 5% relative, ≥ 8 sampled segments, calibration self-check `X̂ ≈ X` within CI); the `sim/trace` reader fail-closes on an incomplete trace lacking a sidecar (`E_INCOMPLETE_TRACE`) and rejects a subset that rewrote `gridDim` (cross-checking the launch record against the sidecar `original_grid`); a golden-set fidelity run (8–12 small cells, full-vs-sampled at fractions × seeds) passes accuracy (p95 |R̂−R| ≤ 0.02 rates, ≤ 5% counts), CI coverage ≥ 90%, and sampling-induced error ≤ 0.3× the model MAE, WITH the histogram + few-wave negative controls correctly FAILING / gate-forbidden.
  - Negative Tests: a block-subset applied to a cell that fails `coop_gate`/`wave_gate`/`atomic_gate` is accepted (must FAIL — it must route to reduced-shape or `unavailable`); an incomplete trace accepted without a sidecar (must FAIL); cycles emitted as an exact full-shape prediction from a sampled trace (must FAIL — cycles are `not_recoverable`); a `dataset_shrunk` reduced-shape whose working set drops below the 6 MB L2 without the typed footprint flag (must FAIL); per-record or deterministic every-k decimation used as a sampling method (must FAIL — inadmissible); the golden-set negative controls PASSING, i.e. the estimator certifies a histogram-class cell (must FAIL the validation).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The full runnable roster — `portable_lcd` (20 kernel-cases), the 31 sm_86-gated `simbench_bw` kernels, and every SOL-ExecBench in-scope cell that reconstitutes and runs on this host (including the 23 shrunk-to-fit host-capacity cells) — captured once under the new address-carrying contract with the full sm_86-available NCU metric vector, the complete 6-config × N-repeat cross-protocol repeatability matrix, all three additive microbenches resolved (or typed-bounded), BOTH the ATOMIC and ASYNC_COPY flips landed and lockstep-verified in one contract transition, the pchase card refreshed as additive evidence, control-bits roster-wide, and a frozen corpus whose typed-exclusion authority ledger accounts for every non-captured cell. Budget-breaching cells are handled by the compression-first ladder (AC-9) — wave-segment subset only where the admissibility gates pass, reduced-shape or typed `unavailable` otherwise, with golden-set-validated estimators and negative controls.

### Lower Bound (Minimum Acceptable Scope)
The same corpus over the honestly-runnable set with typed records only where hardware genuinely blocks: both flips landed and canary-verified in the single contract transition; SOL cells shrunk-to-fit where host memory blocks the native shape (`shape_adjusted`, typed), and any cell that still cannot run recorded with a typed reason (not silently dropped); at least the reference-config full-metric capture banked for every runnable cell before the expensive protocol cross-product; microbenches that stay inconclusive recorded as typed bounded gaps; budget-breaching cells handled by the AC-9 compression-first ladder (at minimum: compression + the admissibility gate + the `sim/trace` reader fail-closed + typed `unavailable`/reduced-shape for gate-forbidden cells; the full wave-segment estimator + golden-set validation may narrow to the minority of cells that still breach after compression). At BOTH bounds the honest denominator, per-chunk provenance, the pre-flip regression anchor, and the typed-exclusion ledger are non-negotiable.

### Allowed Choices
- Can use: new roster-JSON generator + per-suite app-arg adapters; new additive microbench `.cu`/`.py` probes; new campaign-manifest emitter + typed-exclusion authority ledger + their schemas; SOL shape-adjust wrappers; the sanctioned tracer flip + lockstep consumer widening; the sanctioned `freeze_label_bundle.py` widening (DEC-3); genuine bug fixes (e.g. the freezer substring-vs-equality `binary_sha256` idiom); SOL env reconstitution via env overrides / shims (`CUTLASS_DIR`, path-remap) + `uv sync` + HF `download_data.sh`; resume/idempotence wrappers around the existing runners; the AC-9 trace-sampling ladder for budget-breaching cells (zstd compression tier + `.pb.zst` + the compressibility probe, the admissibility-gate classifier, two-pass counting/dump capture, the wave-segment HT estimator, and the closed-schema `trace_sampling.json` sidecar); the `sim/trace` reader completeness fail-closed (a genuine bug fix — closes the silent-incomplete-trace hole) + the never-rewrite-`gridDim` cross-check.
- Cannot use: NCU values as model or card inputs (Rule 1); edits to existing validated code beyond genuine bug fixes, the sanctioned flip, and the sanctioned freezer widening; editing the byte-pinned `arch_profiles/cards/sm_86.json` to transcribe calibration values this loop (DEC-3 — stays additive evidence, transcribed under a separate approved change); normalizing/clipping pct-overflow into admitted labels; committing while a capture sweep is running; capturing degenerate-fallback simbench kernels (arch-gated kernels forced to sm_86) as calibration or gate data; silent (untyped) trace truncation; per-record or deterministic every-k trace decimation (inadmissible — breaks ordered replay); block-subset sampling of `coop_gate`/`wave_gate`/`atomic_gate`-forbidden cells; emitting cycles as an exact full-shape prediction from a sampled trace; a `dataset_shrunk` reduced-shape that drops the working set below the 6 MB L2 without the typed footprint flag; building the M2 eval-join / reporter / baseline rigs and the typed scoring-side guard (out of scope; FUT-1 / FUT-5).

## Feasibility Hints and Suggestions

### Conceptual Approach
Sequence the campaign so the denominator is knowable before the expensive work and the contract is stable before capture:
1. Preflight + authorities: reassert clock locks; record host/toolchain; build the campaign-manifest emitter, the roster-authority generator, and the typed-exclusion ledger skeleton; author the trace-retention budget + the trace-sampling strategy; reconstitute SOL and determine its runnable count (the long pole), shrinking the 23 host-capacity cells to fit.
2. Anchor: replay the frozen M1 bundle under old-contract semantics; assert byte-identity.
3. Flip (under the RLCR/Codex loop): land the producer capability bump + `trace_contract_id`, widen the identity-enforcing consumers in lockstep, add synthetic fixtures; atomic implemented first, async-copy adjudicated and included in the SAME transition.
4. Canary: capture a handful of new-contract cells spanning load/atomic/RED/LDGSTS; prove scorability unlocked and exact metrics unchanged.
5. Reference capture: author the full roster JSON via the generator; run reference-config full-registry capture (traces + labels + control-bits) in immutable, resumable chunks with DONE-manifests; apply the AC-9 compression-first ladder to budget-breaching cells (compress + probe first; gate-classify; wave-segment subset / reduced-shape / typed unavailable).
6. Calibration: pchase re-run + three additive clock64 probes + MSHR/L2-assoc rerun-or-bound + roster-wide control-bits; store as additive evidence, Rule-1 guarded.
7. Matrix: run the full cross-protocol repeatability sweep with a cost/resume gate; emit the uncensored dispersion artifact + determinism classes; type pct-overflow.
8. Freeze: widen the freezer to the full metric vector; emit the frozen bundle + all manifests + the exclusion authority ledger; assert the denominator identity.

### Relevant References
- `tools/local_labels/label_driver.sh` — the only NCU invoker; `--clock-control` is a default, forwardable to none/base/boost.
- `tools/local_labels/repeatability_matrix_runner.py` — roster × protocol matrix runner; `--metrics-from-registry` injects the sm_86-available metric set.
- `tools/local_labels/roster_trace_capture.py` — NVBit capture driver; `LD_LIBRARY_PATH=.toolchain/pb/lib` + ldd preflight; portable_lcd routes through the identity path, others via direct injection.
- `tools/local_labels/freeze_label_bundle.py` — the bundle freezer (widen the HEADLINE tuple; fix the substring `binary_sha256` idiom).
- `tools/nvbit_tracer/tracer_tool/tracer_tool.cu` (`is_excluded_op_class`, the host-side drop, the `kGlobal`-only allocation-resolution gate) + `memory_op_class.h` (`LDGSTS→kAsyncCopy`, `ATOM*/RED*→kAtomic`); `tools/nvbit_tracer/import_trace_contract.py` (`validate_trace_dir`, the global-class identity gate); `tools/sector_stream/generator.py` (still excludes ATOMIC/ASYNC).
- `sim/trace/src/reader.rs` + `sim/trace/tests/proto_schema_pin.rs` (identity enforcement + proto pin).
- `tools/pchase/run_pchase_card.py` + `transcribe_latencies_to_arch_card.py`; `tools/sol_baselines_harness/calibration/cache_prediction/sm120_l2_associativity_probe/` (portable probe precedent); `tools/sass_control/emit_control_bits_sidecar.py`.
- `tools/adaptive_sampler/` (MRC convergence ledger), `tools/cloud/trace_budgets.py` (budget trigger + `trace_volume_exceeded` sidecar), `tools/perturbation_harness/`, `sim/trace/src/reader.rs` (the reader completeness fail-closed) — the AC-9 sampling machinery; full design in `plans/sm86_trace_sampling_strategy_fable5_design.md`.
- `tools/sol_execbench/scripts/download_data.sh` + `run_dataset.py`; `tools/sol_baselines_harness/` (`b200_preflight.py`, `native_env.py`, `run_ncu_harvest_batch.py`, `sm86_port_scope_v1.toml`, `deferred.json`).
- `arch_profiles/cards/sm_86.json`; `eval/frozen_labels/sm86_m1_bundle_v21.json`; `plans/micro2027_ncu_sim_execution_plan.md`; `plans/phase2_sim_core_progress.md`; `plans/host_state_and_migration_log.md`.

## Dependencies and Sequence

### Milestones
1. **M0 — Preflight & Authorities**: preflight/provenance emitter; roster-authority generator; exclusion-ledger skeleton; trace-retention budget + sampling strategy; SOL reconstitution + runnable-count determination (shrinking the 23 host-capacity cells).
   - Phase A: host/toolchain/clock reassert + record; injection smoke; disk budget.
   - Phase B: roster authority (universe/runnable/excluded) + SOL reconstitution feeding it.
2. **M1 — Pre-flip Anchor**: replay + byte-identity assert of the frozen M1 bundle. Depends on M0-A.
3. **M2 — Contract Transition**: producer flip (both classes) + capability/`trace_contract_id` + proto repin + lockstep consumer widening + synthetic fixtures; atomic implemented first, async-copy after operand adjudication, both in one transition. Depends on M1.
4. **M3 — Canary**: new-contract capture across load/atomic/RED/LDGSTS; scorability + no-regression proof. Depends on M2.
5. **M4 — Reference Capture**: full-roster generator + reference-config full-registry capture (traces+labels+control-bits) in resumable chunks; typed sampling for oversized cells. Depends on M0-B, M3.
6. **M5 — Calibration**: pchase re-run + 3 additive probes + MSHR/L2-assoc + roster-wide control-bits, Rule-1 guarded, additive-only. Depends on M0-A.
7. **M6 — Repeatability Matrix**: full cross-protocol sweep + uncensored dispersion + determinism classes + cost/resume gate. Depends on M4.
8. **M7 — Freeze**: widened freezer + frozen bundle + exclusion authority ledger + denominator identity. Depends on M4, M5, M6.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Build the preflight/provenance emitter (host/toolchain/clock/injection-smoke/disk-budget + per-chunk DONE-manifest), the roster-authority generator, and the typed-exclusion ledger schema/skeleton | AC-1 | coding | - |
| task2 | Reconstitute the SOL-ExecBench environment (HF data, `uv sync`, `CUTLASS_DIR` override, path shim) and determine the runnable/deferred split on this host; feed counts into the roster authority | AC-1, AC-5 | coding | task1 |
| task3 | Replay the frozen M1 bundle under old-contract semantics and assert byte-identity (pre-flip regression anchor) | AC-2 | coding | task1 |
| task4 | Land the tracer producer flip (ATOMIC + ASYNC_COPY in one transition, atomic implemented first) + `trace_contract_id`/`memory_payload_classes` capability + proto regen/repin + lockstep identity-enforcing consumer widening (import_trace_contract, sim/trace reader, sector_stream, HRD, order_projections, cache policy) + synthetic producer fixtures | AC-3 | coding | task3, task5 |
| task5 | Adjudicate LDGSTS source/destination operand semantics and the ATOMS shared-atomic vs global-atomic distinction before widening the allocation-resolution gate; resolvable by SASS/PTX/NVBit operand inspection + a known-address canary — escalate to an expert consult only if docs and the canary disagree | AC-3 | analyze | task3 |
| task6 | Capture the new-contract portable_lcd canary (load/atomic/RED/LDGSTS); verify scorability unlocked and `smsp__inst_executed`/L1-op_ld exactness unchanged | AC-4 | coding | task4 |
| task7 | Author the full-roster matrix roster JSON via the generator + per-suite adapters; run reference-config full-registry capture (traces+labels+control-bits) in immutable resumable chunks with DONE-manifests and pre/post source-commit checks | AC-5 | coding | task2, task6, task14, task15, task16 |
| task8 | Re-run pchase under the lock; build the Fable-5 clock64 probe suite in load-bearing order (b→a→d→c→e) with (b)+(d) coupled; rerun/port MSHR + L2-assoc; extend control-bits roster-wide; store as additive Rule-1-clean evidence (no card transcription this loop). Design: `plans/sm86_calibration_microbench_design_fable5_answer.md` | AC-6 | coding | task1 |
| task9 | Adversarially verify (Codex loop) that every microbench card input is clock64-derived, Rule-1 clean, overdetermined by ≥2 readouts (never tuned against sim output), and that inconclusive probes carry typed bounded gaps | AC-6 | analyze | task8 |
| task18 | Build the 6th calibration probe — cp.async/LDGSTS timing: the F-L (completion-latency), F-T (streaming-plateau), and F-B (burst-issue) sub-probes + the (b)/(d) port/pool sharing disentanglement; produce f1–f5 as additive Rule-1-clean evidence. Design: `plans/sm86_cp_async_timing_probe_fable5_answer.md` | AC-6 | coding | task8 |
| task10 | Run the full cross-protocol repeatability matrix (N=20 ref / N=10 non-ref / N=30 escalation, shuffled sweeps, fresh process per profile) with a cost/resume gate; emit the uncensored `determinism_study` artifact per the Fable-5 design (accounting identity, `dispersion_gate`+`sigma_rel` lockstep, 8-class taxonomy, two-stage modality, OOR retain-type, pooled-P90 denominators, two-axis cross-protocol); within-session floors only — no cross-day replication arm (DEC-4). Design: `plans/sm86_determinism_study_stats_fable5_answer.md` | AC-7 | coding | task7 |
| task11 | Widen the freezer to the full metric vector (sanctioned, DEC-3) + fix the substring `binary_sha256` idiom; freeze the full-roster bundle with all manifests + `trace_contract_id` + the typed-exclusion authority ledger; assert `denominator == universe − Σ exclusions` and zero unresolved problems | AC-8 | coding | task7, task8, task10 |
| task12 | Adversarially audit the frozen denominator + exclusion-ledger honesty (every non-captured cell typed; `shape_adjusted` is provenance not exclusion; SOL host-capacity handled per DEC-2) | AC-8 | analyze | task11 |
| task13 | Finalize + adjudicate the Fable-5 trace-sampling design (gate thresholds + estimator vs the sim's `issued == records_seen` replay semantics) via the Codex loop; design captured in `plans/sm86_trace_sampling_strategy_fable5_design.md` | AC-9 | analyze | task1 |
| task14 | Handle the 23 host-capacity SOL cells per the Fable-5 rubric: replay-mode-first check (`native_captured_via_replay_mode`), else shrink via the footprint/roofline/L2 pre-check to the largest regime-preserving shape (classified `shape_adjusted`/`reduced_representative`/`unavailable`), verify correctness + post-capture model agreement, stamp full provenance (DEC-2). Design: `plans/sm86_sol_shrink_to_fit_fable5_answer.md` | AC-1, AC-5 | coding | task2 |
| task15 | Implement the AC-9 sampling machinery: zstd compression tier + compressibility probe, the admissibility-gate classifier, two-pass counting/dump capture, the wave-segment HT calibrated estimator, and the closed-schema `trace_sampling.json` sidecar | AC-9 | coding | task13 |
| task16 | Add the `sim/trace` reader completeness fail-closed (`E_INCOMPLETE_TRACE`) + never-rewrite-`gridDim` cross-check + `records_seen == sidecar.captured` invariant + the calibration self-check assertion (bug-fix-adjacent) | AC-9 | coding | task4, task15 |
| task17 | Run the golden-set fidelity validation (8–12 small cells full-vs-sampled × seeds): accuracy + CI-coverage + error-budget gates + the histogram/few-wave negative controls + the wave-segment-vs-per-block ablation | AC-9 | coding | task15 |

## Future Work / Out of Scope

- FUT-1: Build the M2 eval-join / per-metric reporter (NRMSE + correlation + p90 tails) / baseline rigs (Accel-Sim, UPC MICRO'25, PPT-GPU, Nugteren RDH, SOLAR floor, frozen v5 heads). The campaign produces the corpus these consume.
  - Current-loop handoff: AC-8 (the frozen full-roster corpus is the reporter's input contract).
  - Promotion trigger: the next simulator increment that scores predictions against the frozen bundle.
- FUT-2: Transcribe the measured microbench values into the byte-pinned `arch_profiles/cards/sm_86.json` (updating its two-language lockstep tests) under a separate approved change.
  - Source DEC: DEC-3
  - Current-loop handoff: AC-6 (the values are carded as additive evidence files this loop).
  - Promotion trigger: an approved card-edit change window after the campaign.
- FUT-3: A sim fill-pairing knob to consume a measured 64B L2↔DRAM fetch granularity (a simulator increment; the campaign only measures + cards the value additively).
  - Current-loop handoff: AC-6 (the fetch-granularity value or typed gap is carded as evidence).
  - Promotion trigger: a confirmed non-32B granularity measurement.
- FUT-4: Cross-SKU / sm_120 capture (5060 laptop, cloud GB202) — blocked until M2 and hardware availability per the founding plan.
- FUT-5: The typed scoring-side sampled-input guard (`SimOutput::{Exact,Estimate,ReducedExact}` + `score_full_shape`/`score_estimator`/`score_reduced` that only compile for a matching `TraceScope`, with an explicit `EstimatorAck` opt-in) — belongs to the M2 eval/scoring layer.
  - Source DEC: DEC-1
  - Current-loop handoff: AC-9 (the campaign produces the `trace_sampling` sidecar, the reader completeness fail-closed, and the `full_shape_equivalent` flag the scoring guard consumes).
  - Promotion trigger: building the M2 eval-join (FUT-1).
- FUT-6: The sim `LDGSTS`/cp.async event-model that consumes the (f) card constants (f1 issue cost, f2 completion latency, f3 in-flight capacity, f4 stream bandwidth, f5 wait/bar delta) — replacing the current "bypass load to L2" pricing with issue-occupancy + pool-admission + `wait_group` backpressure — plus the read-only attribution pass that confirms probe (f)'s provisional rank #2 (swap only f's constants into the sim, measure the GEMM/conv/flash-attn per-cell delta; never reverse-fit).
  - Current-loop handoff: AC-6 (f1–f5 are carded as additive evidence with typed gaps).
  - Promotion trigger: the next simulator increment that models the async-copy pipeline.

## Claude-Codex Deliberation

### Agreements
- Contract-transition-first with capture-once (no double-capture) is the correct spine; the frozen M1 bundle is the pre-flip regression anchor; both flips land in one transition.
- The denominator must be machine-checkable and reproducible; a single typed-exclusion authority ledger defends it; SOL runnability (with shrink-to-fit) is established in M0 before full capture.
- Calibration card inputs must be clock64-derived and Rule-1 clean; NCU only cross-checks; values are additive evidence this loop (card transcription deferred to FUT-2).
- Per-chunk immutable provenance + no-mid-sweep-commit + injection-fail-closed preflight are load-bearing; the ~7k-capture matrix needs a cost/resume gate.

### Resolved Disagreements
- **"Lockstep widening" scope**: Codex showed enum-decode is insufficient — the identity-enforcing consumers (`import_trace_contract` global gate, `sector_stream` exclusion, HRD/order_projections/cache policy, Rust reader) must widen deliberately. Resolution: AC-3.2 enumerates them and requires identity enforcement, not decode.
- **Atomic vs async-copy semantics**: ATOMS (shared atomics) must not be forced through the global allocation rule and LDGSTS operand source must be adjudicated. Resolution (round 2): both flips land in ONE transition, atomic implemented first, async-copy adjudicated (task5) and included before completion — NOT deferred (the earlier FUT-2 async-deferral was removed as inconsistent with the operator scope and with AC-3/AC-4).
- **Denominator/SOL ordering**: denominator + SOL runnability precede full capture. Resolution: folded into M0 and M4 dependency.
- **NCU cost**: the ~7k-capture matrix needs a hard stop/resume + cost gate. Resolution: AC-7 requires it.
- **Card edit vs additive (DEC-3)**: freezer widening sanctioned; card transcription is FUT-2 (additive evidence this loop).
- **Sampling strategy (DEC-1, Fable-5 design)**: the detailed method was designed via a Fable-5 consultation and shifted the center of gravity to compression + reduced-shape — block-subset is admissible for a minority of breaching cells, the histogram cell fails the `wave_gate` structurally, per-block is replaced by wave-segment cluster sampling (the sim replays traced placement literally), cycles are `not_recoverable`, the reader gains a completeness fail-closed (bug-fix-adjacent), and golden-set validation with negative controls is mandatory. Full design: `plans/sm86_trace_sampling_strategy_fable5_design.md` (AC-9).

- **Calibration / SOL / determinism designs (Fable-5)**: three further Fable-5 consultations landed and are folded in — the calibration suite reordered to **(b) port-width first** (the 2.8× cycles wall) with a 6th **cp.async/LDGSTS** probe flagged (`plans/sm86_calibration_microbench_design_fable5_answer.md`, AC-6); the SOL shrink rubric with a **replay-mode-first** check and the **B·H-below-saturation → L2-inflation** hazard as the #1 corruption mode (`plans/sm86_sol_shrink_to_fit_fable5_answer.md`, AC-1); the determinism study's **anti-censoring architecture** (accounting identity, `dispersion_gate`+`sigma_rel` lockstep) with **N=20/10/30** fixed tiers (no adaptive stopping), the 8-class taxonomy, and two-axis cross-protocol equivalence testing (`plans/sm86_determinism_study_stats_fable5_answer.md`, AC-7). The 6th **cp.async/LDGSTS** probe was then fully designed (`plans/sm86_cp_async_timing_probe_fable5_answer.md`, task18), promoting it to **load-bearing rank #2** (order b → f → a → d → c → e); its #2 rank is provisional pending a downstream read-only attribution pass (FUT-6). No residual Fable-5-worthy open questions remain.

### Convergence Status
- Final Status: `converged` (two Codex passes + FIVE Fable-5 design consultations — sampling, calibration, SOL shrink, determinism, cp.async — all folded in; the four user decisions DEC-1/2/3/4 resolved; no `needs_user_decision` topics remain and no residual open design questions).

## Pending User Decisions

- DEC-1: Sampling for budget-breaching NVBit traces.
  - Decision Status: RESOLVED — sampling is the DEFAULT, designed by Fable 5 (full design: `plans/sm86_trace_sampling_strategy_fable5_design.md`; seed prompt: `plans/sm86_trace_sampling_strategy_fable5_prompt.md`). The center of gravity is COMPRESSION + REDUCED-SHAPE, not block-subset. Ladder: (T1) lossless zstd `-19 --long=27` per-block `.pb.zst` + a compressibility probe (contract hashes over decompressed bytes) — clears most streaming cells outright; (T1.5) a deterministic admissibility-gate classifier — `coop_gate` (grid-sync → forbid), `wave_gate` (< 8 reconstructable concurrency waves → forbid; this STRUCTURALLY excludes the histogram cell, which has ≈ 1.5–3 waves), `atomic_gate` (> 0.5% atomics + cross-block hot-lines → forbid for L2/cycles/atomic), `skew_gate`, `homogeneity`; (T2) WAVE-SEGMENT stratified cluster subset — NOT per-block (the sim replays traced placement literally + enforces `issued == records_seen`, so independent block thinning silently dilutes co-residency/L1/MSHR/L2 contention), two-pass capture (pass-1 counting-only → offline `blockIdx` force-includes + online pseudo-wave selection in pass-2, counting kept on for ALL blocks for same-run calibration + a bias alarm), a ghost warm-up wave for L2 warmth, HT calibrated-ratio estimators (never average per-block rates), cycles `not_recoverable`, stratified-cluster-bootstrap CIs (≥ 8 segments); (T3) matched reduced-shape (`grid_stride_footprint_preserving` preferred; `dataset_shrunk` FORBIDDEN when it drops the working set below the 6 MB L2) — a different point in launch space, valid for calibration coverage, not full-shape prediction; (T4) temporal window (proven-stationary only); (T5) typed `unavailable` / `budget_limited_wide_ci`. Per-record and deterministic-decimation sampling are INADMISSIBLE. Provenance: a closed-schema `trace_sampling.json` sidecar + version-pinned tracer/stats. Honesty guard (bug-fix-adjacent, in scope): the `sim/trace` reader fail-closes on an incomplete trace without a sidecar (`E_INCOMPLETE_TRACE`) and forbids `gridDim` rewrites. The typed SCORING-side guard (`SimOutput::{Exact,Estimate,ReducedExact}` + scope-matched score fns) belongs to the M2 eval (FUT-5, handoff AC-9). Fidelity validation with NEGATIVE CONTROLS certifies the estimator before use. See AC-9.
- DEC-2: The 23 SOL host-capacity cells.
  - Decision Status: RESOLVED — shrink each to a representative fitting shape and capture as runnable with `shape_adjusted` provenance + a native-shape reference (task14); `host_capacity_deferred` remains a ledger reason only for any cell still unfittable after shrink.
- DEC-3: Sanctioned-edit scope.
  - Decision Status: RESOLVED — the `freeze_label_bundle.py` widening is sanctioned this loop; calibration outputs stay additive evidence files, with transcription into `arch_profiles/cards/sm_86.json` deferred to FUT-2 (`Source DEC: DEC-3`).
- DEC-4: Cross-day replication arm for the determinism study (Fable-5's optional across-session strengthening, `plans/sm86_determinism_study_stats_fable5_answer.md` §5).
  - Decision Status: RESOLVED — NOT included. The simulator will NOT be validated against archived reference data, so the across-session noise floor is not the operative denominator; the determinism study uses **within-session floors only** (no 2–3-day reference replication arm). The Fable-5 design's within-session default is authoritative.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", or similar workflow markers. Use descriptive, domain-appropriate naming (e.g., `trace_contract_id`, `exclusion_ledger`, `preflight_manifest`, `shape_adjusted`).

### Knowledge Consulted
- The `validating-nonroot-gpu-measurement` skill and `.claude/knowledge/measurement/nonroot-cloud-contract.md` are RETIRED (superseded by the NCU-first charter, CLAUDE.md re-founded 2026-07-02) and must NOT be applied.
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`: its silent-injection guidance is honored via the `LD_LIBRARY_PATH=.toolchain/pb/lib` + ldd fail-closed preflight (AC-1); its "prefer bounded instrumentation / sampling" note + the two-pass counting/dump mode ground the AC-9 sampling design.
- `.claude/knowledge/simulator/rust-des-patterns.md`: the sim's tagged-enum dispatch + typed fail-closed error taxonomy (`SystemError::Livelock`, the unpriced-request bound) is the pattern the AC-9 reader completeness fail-closed and the FUT-5 typed `SimOutput` guard follow.
- `.claude/knowledge/patents/patent-family-map.md`: historical architecture context only; not load-bearing for this data-collection campaign.

--- Original Design Draft Start ---

# Contract-First sm_86 Live-Capture Campaign On The RTX 3090 Ti

## Original Idea

Comprehensive sm_86 / RTX 3090 Ti live data-collection campaign — scale Phase-1 capture from the 17-cell portable_lcd pilot to the FULL sm_86-runnable roster (portable_lcd 20 + simbench_bw 31 + SOL-ExecBench 201-in-scope ≈ 229–250 cells): collect ALL NVBit traces, run ALL runnable benchmarks, full NCU metric vector, full cross-protocol repeatability matrix, roster-wide control-bits, re-run/extend calibration cards (pchase + 3 NEW additive microbenches), and FLIP the tracer ASYNC_COPY+ATOM address-exclusion policy under an RLCR/Codex loop to unlock tensor-core/atomic scorability. DATA-COLLECTION ONLY — building the M2 eval-join/reporter/baseline rigs is OUT of scope. Constraint: no edits to existing validated code except (a) genuine bug fixes, (b) the sanctioned tracer flip + lockstep consumer widening; new files (roster JSONs, additive probes, exclusion ledger) are allowed. Full design synthesis (operator scope decisions, live host state, exact tooling entry points with file:line, Codex-revised phase order P0–P7, capability-bump containment for the tracer flip, top risks, bug fixes, typed-exclusion authority ledger) is in the scratchpad file: /tmp/claude-1000/-home-jakeshea-non-root-profiler-cross-arch-gpu-profiler-suite/cbe3a81e-11ee-41a7-b2ec-50423aa5d3cf/scratchpad/campaign_design_brief.md — read it as the primary seed. This is the founding plan context: plans/micro2027_ncu_sim_execution_plan.md (Phase 1) and plans/phase2_sim_core_progress.md (the walls this campaign unblocks).

## Primary Direction: Contract-Transition-First Pipeline

### Rationale

Organize the whole campaign around the tracer-flip trace-contract boundary: flip the ASYNC_COPY/ATOM exclusion FIRST, so the entire roster is captured exactly ONCE under the new address-carrying contract (no double-capture), with the frozen M1 bundle kept as the pre-flip regression anchor and every capture run as an immutable one-commit/one-roster/one-protocol/one-manifest chunk; this angle is distinct because it treats the contract transition (not roster breadth) as the campaign's spine.

### Approach Summary

Organize the campaign around the tracer's trace-contract boundary rather than roster breadth. Land the ASYNC_COPY+ATOM exclusion flip as a single, fully-lockstepped contract transition FIRST, then capture the entire sm_86 roster exactly once under the new address-carrying contract. Concretely:

1. Producer flip (the sanctioned edit), localized to 3 sites in `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`:
   - Remove `kAtomic` and `kAsyncCopy` from `is_excluded_op_class` (lines 720-726) — a ~2-member change to the 7-class exclusion set. TEXTURE/SURFACE/TMA/CLUSTER_DSMEM/TMEM stay excluded (correct on sm_86).
   - Widen the per-lane allocation-resolution gate (line 3275, `cls == kGlobal && mask.any()`) to also resolve global ATOMIC addresses (and the global source side of LDGSTS async-copy), so newly-admitted records get `allocation_id` stamping instead of falling through with `resolved_allocation_id == 0`.
   - The SASS classifier itself is untouched: `memory_op_class.h` already maps `LDGSTS→kAsyncCopy` (line 110) and `ATOM*/RED*→kAtomic` (lines 121-124); the flip only stops the host-side drop at lines 3218-3248 that currently routes these to the `unsupported_memory_path` sidecar.

2. Contract-transition capability declaration: add an additive `memory_payload_classes` field to the `Trace` message in `trace.proto` (currently 6 fields, natural proto3-additive attach point at tag 7). Old M1 traces leave it absent (= pre-flip contract); new traces carry `[GLOBAL,SHARED,LOCAL,ATOMIC,ASYNC_COPY]`. This makes old exclusion-semantics traces manifest-distinguishable from new address-carrying ones so they are never silently cross-compared.

3. Lockstep consumer widening (same commit as producer, per the repo's BL-20260507 multi-surface-lockstep rule):
   - `import_trace_contract.py`: extend the `global_class` gate (line ~1007) — which enforces `allocation_id_missing` / range identity — to also cover ATOMIC (and async-copy source). Critical: the validator currently only rejects UNSPECIFIED, so it already ADMITS ATOMIC/ASYNC_COPY records but WITHOUT identity enforcement; this transition must close that fail-open before any roster capture.
   - `sim/trace/src/reader.rs`: `known_enum::<MemoryOpClass>` already decodes ATOMIC=6/ASYNC_COPY=8 (they are committed enum values, not schema-skew). Downstream sim consumers (`L2OpClass{Read,WriteAtomic}`, HRD/coalescer/order_projections/cache-access policy) already model WriteAtomic but are starved of atomic addresses.
   - `sim/trace/tests/proto_schema_pin.rs`: re-pin the FNV-1a hash for the changed `.proto` in the same commit (existing enforced discipline).

4. Immutable one-commit/one-roster/one-protocol/one-manifest chunking, reusing existing freeze/provenance rigs: keep `eval/frozen_labels/sm86_m1_bundle_v21.json` as the frozen pre-flip regression anchor (replay on OLD traces must stay green), extend `freeze_label_bundle.py`'s manifest (already pins per-cell label+trace sha256 + protocol stamp + source_commit) as the per-chunk seal, and wire `tools/cloud/trace_budgets.py` as the retention/size-budget spine before the atomic-heavy sweeps.

### Objective Evidence

- `tools/nvbit_tracer/tracer_tool/tracer_tool.cu:718-727` — `is_excluded_op_class`, the exact 7-class exclusion set to edit; `:3218-3248` the host-side drop routing to `unsupported_memory_path`; `:3275-3313` the `kGlobal`-only alloc-resolution gate to widen; `:1395-1400` MREF address extraction (`nvbit_add_call_arg_mref_addr64`) proving addresses are already captured before being dropped.
- `tools/nvbit_tracer/tracer_tool/memory_op_class.h:110,121-124` — classifier already assigns `kAsyncCopy`/`kAtomic`; no classification work needed.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto:22-34` — `MemoryOpClass` enum already has `ATOMIC=6`, `ASYNC_COPY=8` as committed values; the flip is a producer-behavior change, NOT an enum bump. Comments cite the exact wire-extension/lockstep policy (`BL-20260507-multi-surface-lockstep-changes`).
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/trace.proto` — 6-field `Trace` message; the additive capability attach point for `memory_payload_classes`.
- `tools/nvbit_tracer/import_trace_contract.py:1007-1049` — `global_class` gate keyed on `MEMORY_OP_CLASS_GLOBAL` only; the fail-open on ATOMIC/ASYNC_COPY the transition must close. `:1108-1135` admits any present non-UNSPECIFIED class. `validate_trace_dir` is the fail-closed entry the capture harness already calls.
- `sim/trace/src/reader.rs:335-451` — `known_enum` + `visit_memory_access` already decode ATOMIC/ASYNC_COPY without schema skew; `MemoryAccessMeta.op_class` carries them downstream.
- `sim/trace/tests/proto_schema_pin.rs:15-25` — committed FNV pins per `.proto`, enforcing regen+repin in the same commit; `tools/nvbit_tracer/lane_reconstruction_golden.json` + the 3-surface cache-op lockstep test are the existing drift-guard precedent for "capture once under one contract."
- `tools/local_labels/roster_trace_capture.py` — existing fail-closed capture→`validate_trace_dir`→manifest with `binary_sha256` join keys, `host_clock_state`, per-cell typed reasons; already the one-manifest-per-roster mechanism.
- `tools/local_labels/freeze_label_bundle.py:1-70` — emits ONE manifest pinning per-cell label sha256 + trace identity + protocol stamp + registry/git provenance ("the manifest is the drift detector"); directly the immutable-chunk seal to extend.
- `eval/frozen_labels/sm86_m1_bundle_v21.json` — the concrete pre-flip anchor: `source_commit ba24b584`, protocol v2.1 (`--clock-control none --cache-control all`), 15 admitted + 2 typed-excluded cells, exact `smsp__inst_executed.sum` dispersion 0.0, per-cell `dynamic_trace_pb_sha256` + `records_admitted` (e.g. atomic histogram 411M records, L2 99.896%).
- `tools/cloud/trace_budgets.py:1-40,123-286` — ready per-cell trace-size/runtime budget validator with `schema_version=1` TOML, `trace_volume_exceeded.json` sidecar, and a `trace_budgets_sha256` downstream-reference validator: the retention spine the brief's risk #5 demands.
- `plans/phase2_sim_core_progress.md:501-524,716-741` — names the exact accuracy wall the flip unlocks: sim already routes "atomic/TEX/async-copy streams" through L2 with `L2OpClass{Read,WriteAtomic}` but the `lts__t_sector_hit_rate` cells and op_ld/LDGSTS scorability are starved because the addresses are dropped upstream.

### Known Risks

- Alloc-gate widening is the riskiest edit: LDGSTS async-copy source is global but destination is shared — deciding which address the wire carries and whether allocation resolution applies to the shared destination is a semantic call that can mis-stamp `allocation_id`. Global atomics are clean; async-copy is subtler and should canary first (P3).
- The validator currently fail-OPENS on ATOMIC/ASYNC_COPY (admits without identity enforcement); if consumer widening lands after captures, new traces bank without allocation-range identity. The transition MUST land validator + Rust widening before any P4 roster capture.
- Capability distinguishability leans on "field absent == old contract"; needs an explicit default/sentinel so absence is never confused with a truncated new-contract trace.
- Double-capture avoidance depends on the entire flip (producer + all consumers + repinned proto + golden fixtures green) landing in ONE commit BEFORE roster capture; a mid-roster flip splits provenance — exactly the brief's risk #2.
- Trace-volume growth: previously-dropped atomic/async records now carry 32-lane address vectors, materially inflating atomic-heavy cells (histogram already 411M records / multi-GB) — stresses the retention budget before the SOL sweep.
- Any `.proto` capability edit forces a `proto_schema_pin.rs` re-pin + `regen_pb.sh` + committed `dynamic_trace.rs` in the same commit, or `cargo test` fails loudly (by design).

## Alternative Directions Considered

### Alt-1: Provenance Spine & Exclusion Authority
- Gist: Build two additive, read-only-over-producers artifacts as the campaign's organizing spine — (1) an append-only campaign manifest that stamps every immutable capture chunk with `source_commit` + `working_tree_clean` + binary/`tracer_tool.so`/`metric_registry.json`/roster-JSON SHAs + the v2.1 protocol block + NCU/driver/nvcc/NVBit versions + `gpu_uuid_sha256` + post-capture clock/persistence/throttle state (closing the gap that the current trace `capture_manifest.json` stamps only `generated_by` + `host_clock_state` + per-cell `binary_sha256`), and (2) a single typed-exclusion authority ledger over the 6 reasons (arch_gate / semantic_mismatch / trace_unsupported / repeatability_failed / correctness_failed / calibration_inconclusive) that unifies ≥5 scattered producer vocabularies and a coverage test guaranteeing denominator = roster − Σ(authority rows). Plus the sanctioned freezer widening (3→61 metrics) and the substring-`sha256` bug fix.
- Objective Evidence:
  - `tools/local_labels/label_driver.sh` — per-run `source_commit`/`binary_sha256`/`kernel_signature_sha256` + `measurement_protocol` host-state sidecar (the per-artifact pattern to lift to manifest scope).
  - `tools/local_labels/evidence_manifest.py` (916 LOC, `pre_sweep_source_commit` + `working_tree_clean`) + `provenance_audit.py` (882 LOC, 7-field join-key audit) — direct precedent for the append-only commit-pinned spine.
  - `workloads/simbench_bw/schemas/sm86_cohort_regen_ledger.schema.json` (331 lines, closed status enum + if/then "excluded ⇒ typed_reason" arms) + `build_sm86_cohort_regen_ledger.py` + `test_per_cell_exclusions_typed_reason_consistency.py` — the exclusion-authority template + coverage test.
  - Scattered per-source typed exclusions to unify: `sm86_port_scope_v1.toml` (arch_gate), `import_nvbit_exclusion.py`/`unsupported_memory_path.json` (trace_unsupported), `label_supersession.py` (repeatability), `deferred.json` (correctness_failed).
  - `CLAUDE.md:83-85` — additive/capability-gated producer format + consumers widen in lockstep authorizes the "new schema, no producer edits" shape; registry confirmed at 61 metrics (not the stale 41).
- Why not primary: It is the honesty/reproducibility discipline that every phase depends on, but it is a cross-cutting layer rather than the campaign's sequencing backbone — it folds into the primary as the per-chunk seal + denominator defense.

### Alt-2: Card-Closure Calibration Campaign
- Gist: Make closing the five named sm_86 ArchProfile `research_gap`/documented-default knobs the primary deliverable — re-run pchase under the v2.1 lock, then build three NEW additive clock64-first standalone microbenches (L2 fetch-granularity 32B-vs-64B, per-SM L1↔XBAR port-width, achievable-DRAM-bandwidth), replace the single-thread MSHR probe with a multi-warp saturation variant, and port the sm_120 L2-associativity probe to sm_86 (clock64-first to stay Rule-1-clean); labels/traces become the validation substrate that scores whether closing each knob moved the sim's named accuracy walls (L2 unified MAE 13.96pp, cycles geomean 0.358).
- Objective Evidence:
  - `plans/phase2_sim_core_progress.md:854-863,930-940,1128-1133` — names each lever verbatim as a load-bearing, still-open accuracy wall ("close by microbench, never by fit").
  - `arch_profiles/cards/sm_86.json` — `mshr_outstanding_misses` research_gap; `dram_bandwidth` peak is a "documented-default … a bandwidth microbench under the locked protocol supersedes this."
  - `tools/pchase/run_pchase_card.py` reusable `ANALYSIS_CONFIG`/`nvml_sample`/`detect_step` harness + `pchase_kernels.cu` `launch_mshr` single-thread limitation (the fixable structural gap); `sm120_l2_associativity_probe/probe_v3.cu` (191-LOC portable precedent).
  - `workloads/simbench_bw/schemas/pchase_static_card.schema.json` OPEN `properties` dict — new probe records are additive, no schema edit.
- Why not primary: It is the sharpest attack on the cycles/L2 accuracy walls, but it prioritizes the sim's input card over the campaign's mandate to capture ALL labels/traces/benchmarks — it lands as the primary's P5 calibration phase.

### Alt-3: Unified Roster Fabric
- Gist: Build ONE additive roster-manifest layer + a deterministic `toml → matrix-roster-JSON` generator (`build_matrix_roster.py`, modeled on `build_sm120_tri_label_roster.py`) that emits the `{slug,binary,shape_point,app_args,roster_source}` contract both `repeatability_matrix_runner.py` and `roster_trace_capture.py` already consume, with per-suite app-arg adapters (portable_lcd `--timing_shape`, ac4 `--case_id`, simbench `_kernel_args_from_shape`, SOL routed distinctly) + a new `matrix_roster.schema.json`, so all corpora flow through the same capture machinery uniformly.
- Objective Evidence:
  - `tools/local_labels/repeatability_matrix_roster_full.json` — the shared `{slug,binary,shape_point,app_args}` contract, hand-authored (no generator exists).
  - `tools/local_labels/build_sm120_tri_label_roster.py` (~200 LOC, schema_versioned + source_commit + SHA pins) — strong generator prior art; `roster_path_resolver.py` `roster_source ∈ {portable_lcd,simbench_bw}` — the unified-routing precedent (SOL absent from `_KNOWN_ROSTER_SOURCES`).
  - `workloads/simbench_bw/run_suite.py:1000-1006,2870-2880` (`_find_kernel_executable`/`_kernel_args_from_shape`) — the simbench adapter to mirror; `tools/sol_execbench/scripts/run_dataset.py:204-253` — SOL has no native `<slug>_run` (the structural mismatch).
- Why not primary: It is the concrete P4 authoring mechanism, but SOL-ExecBench breaks its single-contract uniformity for ~200 of ~250 cells and widening the resolver/routing touches validated code — so it is a component the primary invokes, not the spine.

### Alt-4: Repeatability Determinism-Class Study
- Gist: Make the per-metric determinism-class / noise-floor characterization the primary tracked artifact (the missing Weaver/McKee GPU-counter-determinism analog + the M2/M3 accuracy-gate denominators), scaling the unchanged matrix runner at full roster and adding a NEW uncensored study layer that fixes the load-bearing right-censoring defect (the current classifier only sees `label_admitted.json`, so `noisy` is a truncated `(2%, admission-gate]` band) by reading the raw per-repeat sidecars — which persist even for excluded cells — via the imported `_dispersion_for_rule`, plus a `vendor_defect`/`out_of_range` class for the S7 pct-overflow and a modality axis for genuinely bimodal cells.
- Objective Evidence:
  - `tools/local_labels/repeatability_matrix_runner.py:159-210` + `determinism_class_report.py:158-165` — both self-document the right-censoring ("rows reflect ADMITTED evidence only"); `repeatability_aggregator.py:643-676` `_dispersion_for_rule` is the importable per-rule statistic.
  - `data/annex/repeatability_matrix_full_20260706/` — the 17-cell pilot: top `reference_max_cv` values pile up at 0.049–0.052 against the 0.05 gate (truncation is empirically real); raw `label_<r>_<l>.json` sidecars readable even for excluded cells (`flash_attn_bwd` L2 = [0.877,0.869,0.851,0.987,0.878]).
  - `plans/*` name "the missing Weaver/McKee GPU counter-determinism analog" as load-bearing novelty + a standalone paper section; `vendor_escalation/ncu_sm120_lts_pct_out_of_range/REPORT.md` — the S7 115–217% class with no current bucket.
- Why not primary: It is the campaign's most publishable scientific product and exposes a real defect, but it is the P6 analysis over the full cross-protocol matrix rather than the capture backbone — it consumes what the primary produces.

### Alt-5: Long-Pole-First: SOL Reconstitution
- Gist: Front-load the biggest unknown — reconstitute the absent SOL-ExecBench environment (HF data via `download_data.sh`, the `uv sync` venv, the CUTLASS path via `CUTLASS_DIR` override of `third_party/cutlass/`, a path-remap shim for pre-migration constants) behind a NEW `sm86_reconstitution_preflight` (modeled on `b200_preflight.py`), reproduce the 178/201-correct split on THIS host, then generalize the only-2-kernel NCU-harvest surface to the full 201 via `run_ncu_harvest_batch.py` + hand-authored single-target probe `.cu` wrappers.
- Objective Evidence:
  - `tools/sol_execbench/scripts/download_data.sh` (HF `nvidia/SOL-ExecBench`) — ABSENT `data/benchmark`; `tools/sol_execbench/.venv` ABSENT; `third_party/cutlass/` populated but `native_env.py:89 default_cutlass_dir()` points at absent `REPO_ROOT/cutlass` (`CUTLASS_DIR` override = the no-edit seam).
  - `run_ncu_harvest_batch.py` (~842 LOC, `discover_landed_baselines` + typed `probe_binary_missing_for_slug` folding + 14-cell drift sentinel) scales; `L1_direct_ncu_binaries/` has exactly 2 single-target `.cu` (the pattern to generalize, hand-written per kernel).
  - `deferred.json` — the 23 deferred are mostly 24 GiB host-OOM shape-gating, "NOT an sm_86 compile/portability defect" (reduces reconstitution risk); `sm86_port_scope_v1.toml`/`sm86_coverage_ledger_v1.toml` already author the denominators.
- Why not primary: SOL is the correct highest-risk sub-campaign to de-risk early, but its per-kernel single-target probe labor (~140 hand-written wrappers) and version-skew reconstitution make it a scoped P4 chunk, not the organizing principle for the whole roster.

## Synthesis Notes

The primary (Contract-Transition-First) is the campaign's sequencing backbone; the five alternatives are not rival campaigns but the components it sequences, and each contributes an element the primary should absorb. From Alt-1 (Provenance Spine), fold the append-only campaign manifest as the per-immutable-chunk seal and the single typed-exclusion authority ledger as the denominator defense — both are prerequisites for the primary's "one-commit/one-roster/one-manifest" discipline to be auditable, and the manifest must carry the `memory_payload_classes` capability so pre/post-flip chunks are never joined. From Alt-3 (Unified Roster Fabric), adopt the `build_matrix_roster.py` generator + per-suite adapters as the P4 authoring mechanism that turns the three corpora into the shared `{slug,binary,shape_point,app_args}` contract (accepting that SOL routes distinctly). From Alt-5 (SOL Reconstitution), take the `sm86_reconstitution_preflight` + NCU-harvest generalization as the P4 SOL sub-chunk, front-loaded because it is the highest-uncertainty leg. From Alt-2 (Card-Closure), slot the pchase re-run + three additive clock64 microbenches as the P5 calibration phase that attacks the named cycles/L2 walls. From Alt-4 (Repeatability Study), run the full cross-protocol matrix as P6 and — importantly — adopt its uncensored raw-sidecar reader + modality/`out_of_range` taxonomy so the frozen determinism-class denominators are scientifically honest rather than right-censored. If the operator preferred a different primary, Alt-2 (Card-Closure) is the strongest pivot when the goal is narrowly to move the sim's cycles/L2 accuracy, and Alt-4 (Repeatability Study) is the strongest pivot when the deliverable is the standalone methodological paper section; both, however, still require the primary's contract flip to have happened first so the tensor-core/atomic cells they analyze actually carry addresses.

--- Original Design Draft End ---
