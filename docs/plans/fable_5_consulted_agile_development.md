# Agile Trace-Driven GPU Simulator Redesign Plan

- File: `plans/agile_trace_simulator_redesign.md`
- Status: PROPOSED (offline redesign; no live NCU/NVBit/CUDA/hardware collection in any phase; no `sudo`)
- Target: RTX 3090 Ti / GA102 / `sm_86` only. `sm_120` deferred (embedded fact §2).
- Facts marked "repository-recorded" (embedded §4.7) are prior observations, not re-measured by this plan.

---

## 1. Mission, success metrics, and non-goals

**Mission.** Convert the repository into a small trace-driven GPU simulator prototype with exactly two product-level outcomes:

1. **ACCURACY** — per-metric, per-kernel-family comparison of simulator predictions against already-collected, independently matched NCU labels (labels score predictions only; they never enter simulator inputs).
2. **SPEED-UP** — externally measure this Rust simulator's release-mode process runtime, then compare it through an exact, read-only join to the existing RTX 3090 Ti baseline-runtime database.

**Success metrics (raw, never blended).**

| Outcome | Metric | Current repository-recorded state (§4.7) |
|---|---|---|
| Accuracy | `smsp_inst_executed_sum` exact-match rate | 15/15 exact |
| Accuracy | L1 eligible-load hit-rate MAE (pp) | 0.0003 pp on 10/10 scorable cells |
| Accuracy | Unified L2 hit-rate MAE (pp) | 13.96 pp |
| Accuracy | Remaining 4 implemented metrics: MAE/bias per family | reported, thresholds set post-baseline |
| Accuracy | `gpc__cycles_elapsed.max` | `RequiredUnimplemented(AggregationSemanticsStudy)`; directional ratio geomean ≈ 0.358 (underprediction; not admissible) |
| Speed | Current simulator wall time (raw samples; min/median/MAD/mean±sd, N≥5) | informal: ~30 s @134M records; ~13.9 s @27M records; no formal benchmark exists |
| Speed | Stored-baseline/current-simulator ratio | **PENDING** — requires current runtime samples and exactly one admitted row from the existing RTX 3090 Ti baseline-runtime database |

**Corrections to the prompt's assumptions.**
- A thousands-times speed-up is a **hypothesis (H-SPEED)**, not an acceptance predicate: the authored prior-art record reports 40–250x over GPGPU-Sim 4.0 and does not establish the stronger claim (§4.10).
- The prompt's claim that only an unmatched stub exists is superseded by the user's stated existence of a large RTX 3090 Ti baseline-runtime database. That database is the sole comparison source: this plan never recreates its measurements or acquires, builds, configures, or executes an external simulator.
- `Flat` mode cannot serve as MVP accuracy mode: it yields only instruction count; `L1SharedL2` is the smallest route to all seven implemented metrics (§4.3).

**Non-goals.** Publication/M5–M7 completion; campaign governance; `sm_120`; live capture or NCU collection; history rewrite; backups/archives/worktrees; any blended quality score; claiming unmeasured speed; mutating or regenerating the existing baseline database; cloning, downloading, building, configuring, implementing, or running any external simulator or its repository.

**Blocker closure map.** B001→§4/§5/§11; B002→§7; B003→§9; B004→§6; B005 is superseded by the existing-database fact and becomes the admission/join contract in §8; B006→§10/§4 predicate; B007→§4 (PROTECTED class) + §12; B008→§1/§8; B009→§12.

---

## 2. Minimum end-to-end architecture

Five ownership planes, four process boundaries. NCU and NVBit remain in separate processes and separate matched executions (constraint §3.2); in this redesign all traces and labels are **pre-collected**.

```
[Capture plane — separate process, PRE-COLLECTED]
  NVBit tracer_tool (C++/CUDA) ──> traces_enhanced protobuf files
  tools/sass_control ──> static control-bit JSON sidecar

[Label plane — separate matched executions, PRE-COLLECTED]
  NCU label rows (name, unit, value, dispersion, repeat rule/count)
  └── readable ONLY by the evaluator; no simulator code path can read them

[Simulation plane — ONE Rust process: `sim-run <manifest.v2>`]
  trace decode/admission (sim/trace)
    -> static/dynamic join (sim/sm program.rs)
    -> deterministic (time, sequence) event engine (sim/engine)
    -> SM replay + memory hierarchy (sim/sm, sim/mem; mode L1SharedL2)
    -> metric emission + aggregation + result JSON v2 (sim/run)

[Scoring plane — Python: `eval/score.py`]
  result JSON v2 + label rows -> launch crosswalk -> accuracy_report.json

[Timing/comparison plane — Python: `eval/bench.py`]
  external monotonic wall clock around the release `sim-run` process
    -> `sim_runtime_record.v1.json`
  read-only exact join to existing RTX 3090 Ti baseline-runtime database
    -> `speedup_report.v1.json` or a typed non-comparable outcome
```

Boundary rules: the simulator binary has **no CLI argument, import, or code path** for label files or baseline-runtime data (structural anti-leakage); capture and NCU time are never inside simulator runtime; the evaluator never mutates simulator inputs; the benchmark path never mutates the baseline database or launches a baseline producer.

---

## 3. Irreducible trace / static-SASS / architecture / label contracts

**Dynamic enhanced-trace contract (from §4.6).**

| Field group | Status |
|---|---|
| kernel/device/stream identity; binary architecture; function identity + name | REQUIRED |
| grid dims; block dims | REQUIRED |
| block, warp, SM, warp-on-SM placement/order | REQUIRED |
| `sm_id`, `warpid_sm` | protobuf-optional on the wire; **admission-required** — absence is a typed admission failure, never a default |
| per-instruction PC/function id, active mask, predicate mask, memory op, width, address space | REQUIRED |
| compressed per-lane effective addresses | REQUIRED |
| `memory_op_access_kind` | OPTIONAL (additive) |
| `instruction_cache_op` | **DROPPED from prototype contract** — producer never populates it; replay derives cache role from static opcode text (§4.6) |
| allocation identity | admission integrity only; excluded from cache physics |
| embedded execution-info control bits | **IGNORED** — not authoritative; sidecar is the authority (§4.6) |

**Static-SASS contract.** The 8-field `EXPECTED_LAYOUT` in `sim/sm/src/control_bits.rs` (source, stall, yield_flag, write_barrier, read_barrier, wait_mask, reuse, reserved_must_be_zero) is frozen. The `build_schedules` join keeps its full strictness: `unique_function_id` match, architecture equality, PC-set equality in **both** directions; attaches control bits, opcode class, memory role, opcode text. The Python-producer↔JSON-sidecar↔Rust-consumer parity test is retained.

**Architecture contract.** One `sm_86` / GA102 / RTX 3090 Ti card consumed by replay: SM count and SM→GPC topology, L1 config, L2 config, DRAM bandwidth channel parameters, scheduler counts. Provenance must be microbenchmark (`data/pchase/`) or vendor documentation — **never NCU labels** (constraints §3.1, §3.5). Card carries a provenance field the evaluator checks.

**Label contract.** Generic unit-tagged rows: NCU metric name, unit, value, dispersion, repeatability rule, repeat count (§4.8), plus launch-identity fields sufficient for the crosswalk below. Metric registry retained rows: exactly the seven implemented metrics plus `gpc__cycles_elapsed.max`. The rest of the campaign factory is not retained for label reading.

**Launch identity (replaces insufficient fixture signature of name+grid+block, §4.8).** Composite key, all fields required, join must be bijective:

`(workload_id, matched_execution_id, process_launch_ordinal, kernel_mangled_name, grid xyz, block xyz, device_id, stream_id, binary_architecture)`

Any ambiguity or missing field → typed `JOIN_AMBIGUOUS` / `MISSING_IDENTITY`; never a fuzzy match.

**Result wire v2 (replaces the large result; permitted by §4.4).** Fields: `schema_version="sim_result.v2"`; build identity (sim git commit and dirty flag, binary SHA-256, rustc, profile); trace identity (path, SHA-256, record count, instruction count, byte size, wire version); arch-card identity (id, SHA-256, SM arch, SKU); resolved replay configuration; per-launch `{launch_key, metrics: name → EmitOutcome}`; aggregate metrics; diagnostics block explicitly labeled non-admissible (`diagnostic_last_issue_cycles`, `terminal_modeled_event_time`, and optional internal phase timings for trace open/read/decode, initialization, replay/event-memory processing, emission, and result serialization); typed exit status. Internal timers are diagnostic and may not replace the external spawn-to-exit wall time in §8. **Removed:** scientific-authority status, hardware-readiness status, campaign provenance blobs. `EmitOutcome` keeps its three arms (`Predicted`/`Unavailable`/`RequiredUnimplemented`) unchanged.

---

## 4. KEEP / MERGE / DELETE / PROTECTED / DEFER disposition table

Every DELETE/MERGE is **provisional** until the pre-delete predicate below passes locally (constraint §3.10).

| Path family | Action | Destination / retained responsibility |
|---|---|---|
| `sim/arch` | KEEP | arch card loading; sm_86 facts |
| `sim/engine` | KEEP | deterministic `(time, sequence)` queue; idle-skip; flow-controlled queues |
| `sim/trace` | KEEP | decode, admission typing, lane-address decompression; owns B003 work |
| `sim/sm` | KEEP | replay, `control_bits.rs`, `program.rs` join; **SIMPLIFY:** remove `ReplayMemMode::L1Const` variant if predicate shows no retained consumer/test |
| `sim/mem` | KEEP | L1/L2 sector model + DRAM bandwidth channel |
| `sim/emit` | MERGE | → new `sim/run` crate (emission table for 7 metrics + cycles gate) |
| `sim/runner` | MERGE | → `sim/run` lib (manifest → replay → aggregate → emit → result v2) |
| `sim/cli` | MERGE | → `sim/run` binary `sim-run` (one manifest arg, canonical JSON out) |
| `sim/calib` | DELETE | constants must already live in arch cards; predicate must confirm no retained crate imports it |
| `tools/nvbit_tracer/` + enhanced-trace proto contracts/bindings | KEEP | sole trace producer; wire-contract co-owner with `sim/trace` |
| `tools/sass_control/` | KEEP | sidecar producer; parity fixtures |
| `arch_profiles/` — sm_86 cards consumed by replay | KEEP | runtime architecture input |
| `arch_profiles/` — non-`sm_86` (e.g., `sm_120`) cards | DEFER | retained in place, inactive; never substituted (constraint §3.5); no copies |
| `arch_profiles/` — cards consumed by nothing and duplicating no unique measurement | DELETE | after predicate |
| `data/pchase/` | PROTECTED | unique measured microbenchmark evidence; read-only source for arch cards; untouched, uncopied |
| label schema + metric-registry rows (7+1) + reader | MERGE | → `eval/labels.py`, `eval/registry.py` |
| existing RTX 3090 Ti baseline-runtime database | PROTECTED | read-only comparison evidence consumed by `eval/bench.py`; retain database/snapshot identity and content hash; no mutation, copy, regeneration, or live producer path |
| `eval/production_evaluator/` blocked package + fixture scaffolding | DELETE | replaced by one real scorer (§6); untracked scaffolding first requires `STOP_DIRTY_OR_UNTRACKED_AUTHORITY` |
| `tools/local_labels/` campaign/capture/orchestration factory | DELETE | after reader/registry extraction; subtree-by-subtree predicate |
| Publication/M5/M6/M7 plans, schemas, controller code | DELETE | product direction changed (embedded §2) |
| Historical ledger rows; `data/local_labels/campaign/`; `data/annex/`; external capture / preparation-failure archives | PROTECTED | frozen evidence (constraint §3.9); classified out of active machinery; no copies, no deletion |
| Cache predictor / replay-proxy / calibration tools not consumed at runtime | DELETE | after predicate |
| Docs, ADRs, references, paper corpora, generated reports, old plans, old fixtures | DELETE | except survival set: this plan; the GPC-cycles policy ADR; the NVArchSim prior-art record (grounds §4.10 speed claims); minimal build/run README |
| Broad Python/Rust test suites | DELETE | after fixture/dependency review; §10 set retained |
| CI workflows | DELETE | replaced by the local verification command (§10) |
| `IISWC_conference_latex/` | PROTECTED | untouched, no delegation (constraint §3.7) |
| `.claude/worktrees/` | PROTECTED | untouched (constraint §3.7) |

**Mandatory local pre-delete predicate (P-DEL).** A path may be deleted at HEAD only when all pass, with the transcript pasted into the deletion commit message:

1. `git status --porcelain -uall -- <path>` is clean; any dirty/untracked entry → `STOP_DIRTY_OR_UNTRACKED_AUTHORITY`.
2. Reverse-dependency scan over retained families returns zero hits: Rust `use`/`Cargo.toml` deps, Python imports, protobuf includes, build scripts, retained-doc links, path literals.
3. Producer/consumer + schema check: not referenced by the retained wire, sidecar, registry, manifest, or result schemas.
4. Test-fixture check: not loaded by any retained test (§10).
5. Protected-root check: path is not under, and deletion of its parents cannot touch, any PROTECTED root.
6. Post-deletion gate: build + retained tests green in the same commit series.

No backup, copy, archive, externalization, branch, or worktree may be created for removed material; ordinary Git history retention suffices; no history rewrite (constraints §2, §3.8).

---

## 5. Target repository shape after simplification

```
sim/                      # workspace: 6 crates (was 9)
  Cargo.toml              # members = ["arch","engine","sm","mem","trace","run"]
  arch/ engine/ sm/ mem/ trace/ run/
tools/
  nvbit_tracer/           # producer + enhanced-trace .proto + bindings
  sass_control/           # sidecar producer + parity fixtures
arch_profiles/
  sm_86/...               # consumed cards (KEEP); non-sm_86 cards remain as DEFER
eval/
  labels.py registry.py score.py bench.py
  tests/  fixtures/golden/   # tiny trace, sidecar, labels subset, expected result+report
plans/agile_trace_simulator_redesign.md
docs/  adr-gpc-cycles.md  nvarchsim-record.md  README.md
data/                     # PROTECTED roots untouched: pchase/, local_labels/campaign/, annex/
IISWC_conference_latex/   # PROTECTED, untouched
```

**Smallest survival set (files/interfaces that must exist for the loop to run):** the six crates; the enhanced-trace `.proto` + producer; the control-bit sidecar producer + `EXPECTED_LAYOUT` + `build_schedules` join; one consumed `sm_86` arch card; `manifest.v2` and `sim_result.v2` schemas; `eval/{labels,registry,score,bench}.py`; golden fixtures; the 7+1 registry rows; this plan and the cycles ADR.

---

## 6. Accuracy evaluator contract

One small module set + CLI (`python eval/score.py --result R.json --labels L --out report.json`). No framework, no scaffolding layers (§4.8).

1. **Admission.** Validate result schema version, arch card SKU/arch = RTX 3090 Ti/`sm_86`; label arch equality. Mismatch → fail closed `ARCH_MISMATCH` (constraint §3.3/§3.5).
2. **Crosswalk.** Bijective join on the §3 composite launch key. Unmatched or multiply-matched rows → typed `JOIN_AMBIGUOUS`/`MISSING_LABEL`/`MISSING_PREDICTION`; counted, never scored, never a pass.
3. **Units.** Exact registry unit match; only registry-declared conversions; else `UNIT_MISMATCH`.
4. **Scoring formulas.**
   - Count metrics (`smsp_inst_executed_sum`, ld/st sectors, dram read sectors): exact-match count; relative error `(pred − label)/label` for `label > 0`; `label == 0 && pred == 0` → exact; `label == 0 && pred != 0` → flagged failure cell with absolute value reported.
   - Rate metrics (three hit rates): signed error in percentage points; report MAE, max |err|, signed bias.
   - Cycles (only when admissible per §7): pred/label ratio, geometric-mean ratio, MAE%.
   - Informational flag `WITHIN_LABEL_NOISE` when |err| ≤ label dispersion (dispersion/repeat fields exist per §4.8).
5. **Reporting.** Per metric × kernel family (family from label metadata): N labeled, N scored, N unavailable **by type**, exact count, MAE, max, bias. Plus coverage totals. No aggregate blend; failures are never hidden (constraint §3.4).
6. **Anti-leakage checks.** (a) `sim-run` exposes no label input; (b) evaluator validates the result against a field allowlist (no NCU/label-derived fields); (c) arch-card provenance must not reference NCU; (d) any constant change motivated by label disagreement must be global, documented, and re-scored on all cells — per-cell/same-cell calibration is forbidden (constraint §3.1).
7. **Outputs.** `accuracy_report.json` (schema-versioned) + human-readable table on stdout.

---

## 7. Cycle-metric decision and anti-leakage boundary

**Decision: retain the accepted boundary.** `emit_exec_cycles_gpc_max` keeps returning `RequiredUnimplemented(AggregationSemanticsStudy)` until a checked-in study exists (§4.5). Neither `diagnostic_last_issue_cycles` nor `terminal_modeled_event_time` may be renamed or promoted to `gpc__cycles_elapsed.max`; both stay in the explicitly non-admissible diagnostics block. The alternative — policy supersession promoting a diagnostic without the study — is **not recommended** and would require explicit user approval (Approval A2).

**Smallest target-free single-launch study (deliverable `docs/cycles_semantics_study.md`).** Inputs limited to: documented NCU semantics of `gpc__cycles_elapsed.max`, the arch card's SM→GPC topology, and simulator-internal event times. NCU label *values* are excluded from study inputs. The study must define:

- D1 — per-SM elapsed-cycle window inside replay (start boundary; terminal event inclusion, reconciling the two diagnostics' definitions);
- D2 — GPC aggregation: elapsed cycles per GPC from its member SMs, headline = max over GPCs;
- D3 — multi-launch composition: none exists today (§4.5), so MVP cycle scoring is restricted to single-launch label cells; a serialized-launch sum rule may be proposed in the study for later review;
- D4 — admission rule: emission flips to `Predicted` only after the study is checked in and user-reviewed; the evaluator then scores it like any metric.

Expectation management: the repository-recorded directional geomean ≈ 0.358 implies substantial initial underprediction; report it per family honestly. Remediation is global model work (e.g., DRAM latency from `data/pchase/`-derived card values), never label-fitted constants.

---

## 8. Runtime recording and existing 3090 Ti baseline-database protocol

**Authoritative runtime boundary.** `T_sim` is the externally measured monotonic wall time from immediately before spawning the release `sim-run` executable until that exact child exits. The release build occurs before timing. `T_sim` includes process/loader startup, trace open/read/decode, simulator initialization, replay, memory/event processing, metric emission, result serialization, output flush, and process teardown. It excludes compilation, trace capture, NCU profiling, accuracy scoring, database access, the baseline join, and all baseline-data production. Internal Rust timers for trace read/decode, initialization, replay/event-memory processing, emission, and result serialization are optional diagnostics; they must identify overlap or unmeasured gaps and may never replace, reconstruct, or override external `T_sim`.

**Runtime artifact.** `eval/bench.py record` writes one append-only `sim_runtime_record.v1.json` containing:

- record/schema identity and UTC creation time;
- simulator Git commit and dirty-state flag, release-binary SHA-256, rustc version, Cargo profile, canonical argv, an allowlisted environment snapshot/hash, manifest SHA-256, resolved replay configuration, and arch-card id/SHA-256;
- host identity: hostname hash, CPU model, physical/logical core counts, RAM, OS/kernel, process affinity, CPU governor, load averages, and target GPU SKU/architecture/driver identity when available; unavailable optional host fields are typed rather than omitted;
- workload/family/input identity, launch-set digest and launch count, every launch key, trace SHA-256/wire version/byte size/record count/instruction count;
- one record per warm-up and timed repeat: repeat index, warm-up flag, monotonic start/stop nanoseconds, wall nanoseconds, user/system CPU nanoseconds, peak RSS, exit code or signal, result path/size/SHA-256, and stderr SHA-256; optional OS counters use typed `UNAVAILABLE`, never zero defaults;
- summary over timed repeats only: `N`, min, median, median absolute deviation, mean, sample standard deviation, coefficient of variation, records/s, and instructions/s; and
- the exact `sim_result.v2` identity plus the matched `accuracy_report.json` id/SHA-256 when scoring is complete, so speed and accuracy refer to the same simulator result and launch set.

Raw samples are authoritative and retained in the record; summaries never replace them.

**Recording procedure and gates.** Build once with `cargo build --release` outside the timed region, hash the executable, run one discarded warm-up, then run at least five timed repeats with an identical manifest, argv, environment, affinity, and warm page-cache policy. Do not drop caches or alter governors. Start the external monotonic timer immediately before child spawn and stop it only after `wait` returns. Every repeat must exit zero and produce byte-identical result JSON; otherwise emit `RUNTIME_EXECUTION_FAILED` or `NONDETERMINISTIC_RESULT` and compute no summary or speed-up. Record background load before every repeat and run no concurrent heavy job. If coefficient of variation exceeds 5%, collect four additional repeats; if it still exceeds 5%, retain all samples and emit `RUNTIME_NOISY`, with no speed-up claim. A dirty build may produce an exploratory record but is labeled `DIRTY_BUILD_UNCLAIMABLE`; a dashboard speed-up requires a clean committed build.

**Read-only baseline admission and exact join.** `eval/bench.py compare --runtime <record> --baseline-db <authority> --baseline-config-id <id>` opens the existing RTX 3090 Ti baseline-runtime database read-only and never invokes a producer. It records the database schema version, immutable snapshot/database id, whole-database or snapshot-manifest SHA-256, selected configuration id/hash, selected row id, and canonical selected-row SHA-256. Unsupported schema, missing hash authority, or a changed snapshot fails before query results are consumed.

The comparison consumer—not merely a helper test—must validate that exactly one nonempty row from that one admitted database snapshot matches all of:

`(workload_id, workload_version_or_input_digest, kernel_or_launch_set_digest, launch_count, target_gpu_sku=RTX_3090_Ti, target_gpu_arch=GA102, sm_arch=sm_86, baseline_config_id, cache_state, launch_scope, timing_boundary_id, host_fingerprint, runtime_unit)`.

No fuzzy name match, cross-row aggregation, cross-database fill, nearest configuration, architecture substitution, or silent unit conversion is allowed. A versioned explicit workload-id crosswalk may be used only when its id and SHA-256 are recorded and it still yields one row. The baseline timing boundary must cover the same workload/launch scope and the same end-to-end simulator phases as `T_sim`; capture, profiling, compilation, and evaluation time must be absent from both. The host fingerprint must prove the database row and current runtime were recorded on the same host or an explicitly predeclared identical host class; otherwise raw current runtime remains reportable but the ratio is unavailable.

**Speed-up computation and output.** For one admitted row, convert only a registry-declared runtime unit, select the database row's declared canonical median (or recompute the median from that row's raw samples under its declared repeat policy), and compute:

`speed_up = stored_baseline_median_seconds / current_simulator_median_seconds`.

`speedup_report.v1.json` stores both raw sample sets when present, both summaries, numerator and denominator units, the exact join key, database/row/config hashes, runtime-record hash, simulator-result hash, accuracy-report hash, the ratio, and the descriptive spread `[baseline_min/current_max, baseline_max/current_min]` when both raw ranges exist. It never reports that range as a confidence interval.

Typed non-comparable outcomes include `BASELINE_DB_UNAVAILABLE`, `BASELINE_SCHEMA_UNSUPPORTED`, `BASELINE_AUTHORITY_MISSING`, `BASELINE_NO_MATCH`, `BASELINE_MULTIPLE_MATCHES`, `BASELINE_ROW_EMPTY`, `WORKLOAD_MISMATCH`, `LAUNCH_SCOPE_MISMATCH`, `ARCH_MISMATCH`, `CACHE_STATE_MISMATCH`, `TIMING_BOUNDARY_MISMATCH`, `HOST_MISMATCH`, `UNIT_MISMATCH`, `RUNTIME_NOISY`, and `MISSING_ACCURACY_LINK`. Each preserves the current simulator's raw runtime record and computes no ratio. An empty query result can never satisfy a gate.

**Hard scope rule.** No phase may clone, download, check out, build, configure, implement, patch, execute, or benchmark an external simulator or its repository. No new baseline runs are authorized. The existing RTX 3090 Ti database is immutable comparison evidence only. The thousands-times result remains hypothesis H-SPEED until the exact join above yields measured numerator and denominator values (§1).

---

## 9. Performance optimization order

Profile-before-optimize; after every step, the golden replay and full accuracy report must be regenerated and be identical (or the accuracy delta measured and reviewed). Optimizations that weaken metric semantics without a measured accuracy effect are forbidden.

1. **O0 — Runtime recorder.** Implement `eval/bench.py record` + phase timers; establish this project's records/s and instructions/s on T-mid/T-big; admit and join the existing database only after the current raw samples pass §8. (Closes "no formal benchmark", §4.7.)
2. **O1 — Decode (B003, largest lever: 3.19 s of 3.49 s trace walk is Prost decode, §4.7).** In order: (a) allocation reduction — intern function-name/opcode strings to ids on first decode, reuse buffers; (b) lazy/zero-copy field access for hot per-instruction records instead of full `prost` message materialization; (c) only with Approval A4: wire v2 (packed/columnar batches) — requires producer change, wire version bump, and parity/fixture updates on both sides; (d) optional pipelined decode thread feeding a bounded channel — permitted only if replay event order stays byte-identical (determinism gate), else rejected.
3. **O2 — Batched per-SM actor work** (stated design principle, §4.1); measure event-queue overhead before/after.
4. **O3 — Queue/wake audit** — extend the measured wake-deduplication direction (15 s → 13.9 s, §4.7); verify no retry-event loops remain.
5. **O4 — Memory-model micro-costs** — dense structures over hash maps in sector paths, guided by profile only.

---

## 10. Minimal tests and local verification command

**Retained/created tests.**

| ID | Contract protected |
|---|---|
| T1 | Wire/lane decode: protobuf round-trip + compressed lane-address golden |
| T2 | Static-control parity: Python producer ↔ sidecar ↔ Rust consumer (exists; retained per §4.6) |
| T3 | `EXPECTED_LAYOUT` invariants incl. reserved-bits-zero |
| T4 | `build_schedules` join failures: function-id mismatch, arch inequality, PC-set asymmetry |
| T5 | Memory-model parity: synthetic access patterns with hand-computed L1/L2 sector hit rates |
| T6 | Determinism: same trace twice → byte-identical result JSON |
| T7 | End-to-end golden: tiny checked-in trace → exact expected `sim_result.v2` (7 metrics + cycles `RequiredUnimplemented`) |
| T8 | Evaluator: join-failure typing, unit mismatch, zero-label handling, per-family algebra, anti-leakage allowlist |
| T9 | Runtime/database harness: golden release process emits schema-valid `sim_runtime_record.v1`; a tiny read-only fixture database yields one exact ratio; empty, duplicate, row-empty, unit, host, scope, cache-state, and hash mismatches fail on the real `record`/`compare` consumer path; subprocess allowlist proves the harness can launch only this project's `sim-run` binary |

Broad campaign/orchestration/publication-schema test families and CI workflows are deleted only after P-DEL (notably its fixture check), since blind deletion could sever producer/consumer or parity contracts (B006).

**Local verification command (replaces CI):**

```
cargo fmt --all --check \
 && cargo clippy --workspace --all-targets -- -D warnings \
 && cargo test --workspace \
 && cargo build --release \
 && target/release/sim-run eval/fixtures/golden/manifest.v2.json > /tmp/g.json \
 && diff /tmp/g.json eval/fixtures/golden/expected_result.v2.json \
 && python -m pytest eval/tests -q \
 && python eval/score.py --result /tmp/g.json --labels eval/fixtures/golden/labels.jsonl --out /tmp/r.json \
 && diff /tmp/r.json eval/fixtures/golden/expected_report.json
```

---

## 11. Phased deletion and implementation sequence

All phases are offline: pre-collected `sm_86` traces and labels only; no `sudo`, no live NCU/NVBit/CUDA (constraint §3.6).

| Phase | Path families / actions | Prerequisites | Outputs | Exit gate |
|---|---|---|---|---|
| **P0 — Inventory & authority (no deletions)** | `git status --porcelain -uall` full classification of every dirty/untracked path into OWNER_KEEP_COMMIT / OWNER_DISCARD / UNKNOWN; reconcile `master` vs `main` by inspection (no rewrite, no force-push) | none | dirty-path adjudication list for user; branch decision | zero UNKNOWN paths; branch base fixed; else `STOP_DIRTY_OR_UNTRACKED_AUTHORITY` / `STOP_BRANCH_DIVERGENCE` |
| **P1 — Contract freeze** | write `manifest.v2` + `sim_result.v2` + launch-key + report schemas; list 7+1 registry rows to extract | P0 | schema docs in `docs/` | schemas reviewed; T2 parity still green |
| **P2 — Evaluator MVP** | implement `eval/{labels,registry,score}.py` + T8; run on existing stored results + labels | P1; label-reader extraction | first `accuracy_report.json`, dashboard | reproduces repository-recorded facts (§4.7): inst 15/15 exact; L1 MAE ≤ 0.001 pp; L2 MAE = 13.96 ± 0.05 pp — proves crosswalk correctness |
| **P3 — Consolidation** | merge `emit`+`runner`+`cli` → `sim/run`; slim result to v2; drop `L1Const` if predicate passes; update golden T7 | P2 | 6-crate workspace | full §10 command green; evaluator rerun byte-identical report |
| **P4 — Staged deletion** | one family per commit with P-DEL transcript, in order: publication/M5–M7 controllers+schemas+plans → `tools/local_labels/` factory → `eval` scaffolding (post-P0 authority) → predictor/proxy/calib tools incl. `sim/calib` → docs/references/reports/old plans/old fixtures (minus survival set) → broad tests + CI → unconsumed arch cards (non-sm_86 → DEFER, not deleted) | P3; P-DEL per family | target tree of §5 | build+tests green after every commit; PROTECTED roots untouched |
| **P5 — Current-simulator runtime recorder** | implement `eval/bench.py record`, `sim_runtime_record.v1`, external monotonic timing, raw-sample/rusage capture, result-hash determinism gate, internal phase diagnostics, and T9 record-path smoke; measure T-mid/T-big with one warm-up and N≥5 | P3 | immutable runtime records, formal records/s and instructions/s, runtime dashboard rows | all identities and raw samples present; zero exits; byte-identical results; clean committed build for claimable rows; noise gate passes or is typed |
| **P6 — Existing-database admission and speed-up join** | implement read-only database adapter and `compare` consumer; validate schema/snapshot hash on the actual consumer path; exact one-row join to the existing RTX 3090 Ti database; add T9 empty/duplicate/mismatch/adversarial fixture cases; never invoke a baseline producer | P5; authoritative database location/snapshot available | `speedup_report.v1.json` with raw numerator/denominator evidence and database-row hash, or a typed non-comparable result | exactly one nonempty same-snapshot row; matching architecture/host/scope/cache/boundary/units; linked accuracy hash; ratio formula replay-checks exactly |
| **P7 — Decode optimization round 1** | O1a, then O1b; O1c/O1d only with approvals/gates (§9) | P5 profile | measured Δ + unchanged goldens | T6/T7 byte-identical; accuracy report identical; speed Δ reported; P6 join rerun against the same immutable database row when available |
| **P8 — Cycle study** | write target-free study (D1–D4); after user review, flip emission gate; score single-launch cells | P2 | study doc; cycles rows in dashboard | no diagnostic renamed; emission only post-review; honest per-family ratios reported |

First useful accuracy milestone = P2 (achievable quickly on existing `sm_86` traces and labels). First complete two-outcome milestone = P6. `sm_120`, live collection, publication work, and all new baseline execution stay deferred or prohibited until the dashboard has measured results.

---

## 12. Commit boundaries and dirty-worktree stop rules

- One concern per commit; deletion commits never mixed with refactors; every deletion commit embeds its P-DEL transcript.
- Stage per explicit path only (`git add <path>`). **Forbidden:** `git add -A`/`-u` repo-wide, `git checkout -- .`, `git clean` without the P0 adjudication list, history rewrite, force-push, backup branches, worktrees, archives/copies of removed or protected material.
- `STOP_DIRTY_OR_UNTRACKED_AUTHORITY`: no deletion, overwrite, staging, or commit of any path whose current diff, dependencies, and ownership were not first inspected locally (embedded §2); unresolved untracked evaluator/protocol work is user-adjudicated in P0 before `eval/` scaffolding deletion.
- Branch rule: session starts on `master` while the main branch is `main` (embedded §2); P0 resolves relationship by fast-forward/normal merge only; divergence plus dirty overlap → stop and report.
- PROTECTED prefixes are hard-guarded in any scripted deletion helper.

---

## 13. Acceptance dashboard and mechanical success predicate

**Dashboard (`dashboard.md` + machine-readable `dashboard.json`), no blended score.**

| Section | Required cells |
|---|---|
| Accuracy | per metric × kernel family: N labeled, N scored, N unavailable by type, exact count, MAE, max, bias; cycles row shows `RequiredUnimplemented` or scored ratios |
| Coverage | total cells; silent-absence count (must be 0); unavailable-type histogram |
| Speed | per reference trace: every raw wall/CPU/RSS sample; min/median/MAD/mean±sd/CV (N, host fingerprint, binary/build id, manifest/config/arch hashes, trace SHA-256, records, instructions, bytes); records/s, instructions/s; internal phase diagnostics |
| Comparison | existing-database snapshot/schema/hash, selected configuration id/hash, exactly one selected row id/hash, stored raw times/summary, current raw times/summary, exact join key, timing-boundary id, and stored-median/current-median ratio; otherwise one typed non-comparable status with no ratio |
| Identity | sim commit/dirty flag/binary hash, arch-card hash, replay mode, wire version, label-set id, accuracy-report hash, runtime-record hash, schema versions, baseline-database snapshot hash, selected row/config hashes |

**PROTOTYPE_SUCCEEDS** iff all hold:

- A1: every (metric, launch) cell in the full `sm_86` labeled set is Predicted-and-scored or typed-unavailable; silent absences = 0.
- A2: `smsp_inst_executed_sum` exact on 100% of scorable cells.
- A3: L1 eligible-load hit-rate MAE ≤ 0.01 pp.
- A4: all seven metrics report MAE/max/bias per family (L2 improvement toward ≤ 5 pp is goal G-L2, not a gate).
- C1: cycles cell is either `RequiredUnimplemented(AggregationSemanticsStudy)` or Predicted with a checked-in study reference — never a renamed diagnostic.
- S1: current simulator wall time is recorded per §8 for T-mid and T-big with full identity, raw samples, zero-exit and deterministic-result gates, and acceptable noise.
- S2: each claimed speed-up joins to exactly one nonempty row in one hash-admitted RTX 3090 Ti database snapshot; architecture, host, workload/launch scope, cache state, timing boundary, configuration, and units all match.
- S3: `speed_up` is finite, positive, mechanically replayable as stored baseline median divided by current simulator median, and links the exact accuracy report for the same simulator result. Typed non-comparable outcomes are honest outputs but do **not** satisfy `PROTOTYPE_SUCCEEDS`.
- S4: static and runtime guards show that the benchmark path can spawn only this project's release `sim-run` binary and cannot mutate the database or invoke any baseline producer.

---

## 14. Risks, explicit stop conditions, and decisions requiring user approval

**Risks and mitigations.** R1 decode optimization drifts semantics → T6/T7 byte gates + accuracy re-score per step. R2 crosswalk ambiguity for repeated same-signature launches → ordinal-bearing composite key, bijection requirement, typed failure. R3 L2 error (13.96 pp) propagates into cycles → per-family reporting; global-only model fixes. R4 benchmark noise → raw samples, warm-up/repeats, load capture, MAD/CV gate. R5 deleting a needed fixture/contract → P-DEL steps 2–4. R6 `master`/`main` divergence over dirty work → P0 stop. R7 protected-root adjacency during parent-directory deletions → prefix guard. R8 wrong or empty database selection yields a plausible ratio → snapshot/row/config hashes, exact one-row/nonempty consumer-path validation, adversarial T9 cases. R9 timing-boundary drift invalidates the ratio → named boundary id and exact include/exclude validation on both operands. R10 database mutation or accidental external execution expands scope → read-only database opening, subprocess allowlist, and no network/repository acquisition path.

**Stop conditions.** `STOP_DIRTY_OR_UNTRACKED_AUTHORITY`; `STOP_BRANCH_DIVERGENCE`; `STOP_PROTECTED_PATH`; any typed baseline admission/join failure from §8; `STOP_POLICY_SUPERSESSION` (any cycles promotion without the study); `STOP_ARCH_MISMATCH` (labels/trace/card disagree on `sm_86`/SKU); `RUNTIME_EXECUTION_FAILED`; `NONDETERMINISTIC_RESULT`; `RUNTIME_NOISY`; P-DEL failure on any deletion candidate. A stop preserves raw current-project runtime evidence but never emits a speed-up ratio.

**Decisions requiring explicit user approval.** A1: disposition of every dirty/untracked path from P0. A2: any cycle-policy supersession (recommendation: retain boundary + study). A3: any proposal to replace the existing RTX 3090 Ti database authority, permit a proxy/cross-host/ambiguous ratio, mutate the database, or perform new baseline execution (recommendation: reject and keep the result typed non-comparable). A4: trace wire v2 (producer+consumer+parity change). A5: any deletion whose predicate outcome is ambiguous. A6: branch reconciliation choice if `master`/`main` have diverged.
