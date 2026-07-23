# FABLE-5 BINDING SUPPLEMENTAL CORRECTION — Round 28 V2 Reproduction Contract

**Session:** `b41e0614-a371-4041-b2ed-0d0aecfda548` · **Round:** 28
**Supplements and narrowly amends:** `ANNEX_C_AMEND_R28_PREP070_RECON_V2` (which remains the live amendment ID — no V3 is minted, no live schema const changes its amendment-ID value)
**Correction ID:** `R28_PREP070_RECON_V2_CORR1`
**Save verbatim to:** `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_prep070_reproduction_contract_correction.md`, mode `0444`, before any implementation resumes; record its SHA-256 per Section 8.

---

## 1. DECISION

1. **Option A is binding**, with the refinements below. Option B is rejected (it contradicts the retained requirement that the tool itself emit the complete reproduction record). Option C is rejected (the no-subprocess boundary of the comparator tool is retained verbatim; `subprocess` never enters the tool).
2. **Contradiction A is resolved** by (a) making the permitted import list exhaustive and adding exactly one module, `datetime`; (b) making `executed_at` self-clocked inside the tool, with caller-supplied timestamps structurally impossible; and (c) adding a required `--tool-head` argument whose value the **builder** obtains and git-verifies. The record is produced at Section 10 step 6 — **after H3 exists** — so the "H3 does not exist" paradox dissolves: the tool is implemented placeholder-free, and the truthful value is injected at run time by a component that is already authorized to run git.
3. **Contradiction B is resolved** by striking the false "journal files" sentence. The source scratch copy comes from the **external file pinned by the sealed historical prepare manifest** (absolute path and SHA `3e0aceb3…` recorded at `source_inputs.definition` in the journal member `capture_prepare_manifest.json`, manifest SHA `d9c2fd5e…`); the staged scratch copy comes from the **journal member** `sol_stage/definition.json` (`e7526755…`). Both are rehashed before and after copying; the journal is untouched.
4. `command_argv` becomes a **JSON array of the exact `sys.argv`** observed by the tool. The string form is retired.
5. The reproduction record gains a schema: `workloads/simbench_bw/schemas/legacy_comparator_reproduction.schema.json` — the sixth affected schema file. The five stopped drafts are retained and amended in place.
6. Ledger compatibility is bound as an **entry-level `oneOf`**: Arm A is the H2-implemented contemporaneous shape verbatim (the sole path for all future incidents); Arm B is the single allowlisted reconstructed shape for this incident. `prior_preparation_failures` never appears in any ledger entry; it is carried only by the replacement journal.
7. The RLCR session log is bound to a new dedicated append-only file `.humanize/rlcr/2026-07-08_11-39-13/session_log.jsonl`. The Markdown `## Session Log` in `round-28-summary.md` is narrative, not the log, and is never rewritten.
8. Everything not amended here remains binding verbatim: Annexes A–D; the V2 factual correction and authority rebinding (`SOL_DEF_LEGACY_COMPARE_9B8E0E40`, conjunctive dual-basis, the replacement literal `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`); frozen V1 and both immutable audits; H2 `823f2d60c9bc5c5e78d21f0…` (full: `823f2d60c9bc5c5e78d21f` — exact: `823f2d60c9bc5c5e78d21ff0c74f3e48ceaa1abc`); all cardinalities (344/122/222/0; budgets 122; chunks 18; runnable SOL 74; bootstrap target 2/120/0/0/0/true; final 122); AC-9; the required H3 subject `fix(local_labels): rebind prep070 refusal authority to legacy 9b8e0e40 comparator`; and the pinned `gpt-5.5:high` review path with wait-and-retry, no model substitution. Codex stays on its pinned model.

## 2. EXACT AMENDMENTS TO V2 SECTIONS 5–7

Exactly four loci of the V2 adjudication are amended; everything else in V2 stands verbatim.

**Amendment 1 — V2 §6, tool constraints sentence.** Struck: the parenthetical list `(json, hashlib, argparse, pathlib, sys)`. Replaced by:

> Python stdlib only, and exhaustively: `argparse`, `datetime`, `hashlib`, `json`, `pathlib`, `sys`. Importing any other module — explicitly including `os`, `subprocess`, `shlex`, `time`, `platform`, and any capture, runner, or CUDA module — is forbidden and is a static-test failure.

**Amendment 2 — V2 §6, Inputs sentence.** Struck: "Codex copies, read-only, the journal files whose SHA-256 equal the sealed source/staged definition SHAs into `prep070_reconstruction/scratch/source_definition.json` and `scratch/staged_definition.json`." Replaced by:

> The record builder creates `scratch/` inside the construction site and populates it read-only: (a) `scratch/source_definition.json` is copied from the external pinned source file named by the sealed historical prepare manifest — the builder first rehashes the journal member `cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation/sol_stage/capture_prepare_manifest.json` against `d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7`, parses it, requires `source_inputs.definition.sha256` to equal `3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa`, requires `source_inputs.definition.path` to equal `/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/tools/sol_execbench/data/benchmark/L1/070_mamba2_fused_intra_chunk_diagonal_computation/definition.json`, and verifies the on-disk file at that path hashes to `3e0aceb3…` both before and after copying; (b) `scratch/staged_definition.json` is copied from the historical journal member `cells/…/sol_stage/definition.json`, verified to hash to `e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387` before and after copying. The journal is never modified; its inventory-domain SHA (`ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`, 22 files, 3,855,797 bytes) is re-verified after the copies. The source definition is not a journal member; its being gitignored (`tools/sol_execbench/.gitignore:35:/data/*`) is immaterial because identity is carried by the manifest pin and the hash, not by Git tracking.

**Amendment 3 — V2 §6, bound command and record shape.** Replaced in full by Section 5 below (adds `--tool-head`, binds CWD and path spellings, converts `command_argv` to an array, adds the two origin objects).

**Amendment 4 — V2 §5.6, ledger rules.** Replaced in full by Section 7 below (entry-level `oneOf`; `authority` object extended with correction fields; five-element `reconstruction_records`; `supersedes` const null on the reconstructed arm only; `campaign_rows` and `capture_supersession_attempts_consumed` live inside the embedded attestation object, not as top-level duplicates; no `prior_preparation_failures` field in any ledger entry).

## 3. PERMITTED IMPORTS AND TIME/HEAD SOURCES

**Imports (exhaustive):** `argparse`, `datetime`, `hashlib`, `json`, `pathlib`, `sys`.

**`executed_at`:** read exactly once inside the tool, after argument parsing and before any hashing, as `datetime.datetime.now(datetime.timezone.utc)`, serialized `isoformat(timespec="microseconds")` with `+00:00` replaced by `Z`. No `--executed-at` flag may exist; argparse must define exactly the six flags of Section 5 and no others.

**`python_version`:** `f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"`.

**`tool.sha256`:** self-computed by the tool as SHA-256 of `pathlib.Path(__file__).resolve().read_bytes()`.

**`tool.head` (H3):** supplied by the builder via required `--tool-head`. The tool validates only the format `^[0-9a-f]{40}$` (refusal `SOL_LEGACY_REPRO_HEAD_INVALID`, nonzero exit) and embeds the value. The **builder**, before invoking the tool, must:

1. verify `git status --porcelain -uall` is empty;
2. verify `git rev-parse HEAD` equals the value it will pass;
3. verify `git rev-parse HEAD^` equals H2 `823f2d60c9bc5c5e78d21ff0c74f3e48ceaa1abc` (H3 is the single commit atop H2);
4. verify `git show -s --format=%s HEAD` equals exactly `fix(local_labels): rebind prep070 refusal authority to legacy 9b8e0e40 comparator`;
5. verify the committer instant of HEAD is strictly after `2026-07-10T20:15:42Z`;
6. record `t_pre` (wall clock) immediately before spawning the tool and `t_post` immediately after exit;

and after the tool exits, must:

7. validate the emitted record against the Section 6 schema;
8. verify `tool.head` equals the passed value and equals current `git rev-parse HEAD` (refusal `SOL_LEGACY_REPRO_HEAD_MISMATCH` for any failure of checks 1–5 or this one);
9. verify `tool.sha256` equals both the SHA-256 of the on-disk tool file and the SHA-256 of the blob bytes from `git show HEAD:tools/local_labels/legacy_compare_9b8e0e40.py` (refusal `SOL_LEGACY_REPRO_TOOL_DRIFT`);
10. verify `t_pre − 2s ≤ executed_at ≤ t_post + 2s` and `executed_at > 2026-07-10T20:15:42Z` (refusal `SOL_LEGACY_REPRO_TIME_INVALID`);
11. verify `command_argv` per Section 5 (refusal `SOL_LEGACY_REPRO_ARGV_MISMATCH`).

The builder invokes the tool with the same interpreter as itself (`sys.executable`) via `-m`, CWD = repository root. Interpreter spelling never enters the sealed record; identity is carried by `python_version` and the argv module path.

## 4. CORRECTED SOURCE-BYTE PROVENANCE

As in Amendment 2. Additionally, the reproduction record seals the provenance explicitly (Question 8 — **yes**) via two closed const objects inside `inputs`:

```json
"source_origin": {
  "kind": "external_prepare_manifest_pinned",
  "prepare_manifest_journal_relative_path": "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation/sol_stage/capture_prepare_manifest.json",
  "prepare_manifest_sha256": "d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7",
  "source_path": "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/tools/sol_execbench/data/benchmark/L1/070_mamba2_fused_intra_chunk_diagonal_computation/definition.json"
},
"staged_origin": {
  "kind": "historical_journal_member",
  "journal_relative_path": "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation/sol_stage/definition.json"
}
```

These are schema-pinned constants describing builder-verified provenance; the builder must complete the Amendment-2 verification **before** invoking the tool, else refuse with `SOL_LEGACY_REPRO_INPUT_DRIFT` (detail distinguishes manifest-hash, pin-mismatch, missing-file, and hash-drift cases).

## 5. EXACT REPRODUCTION COMMAND AND RECORD SHAPE

**Bound command** (CWD = `/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite`; `<H3>` is the verified 40-hex head):

```
python3 -m tools.local_labels.legacy_compare_9b8e0e40 \
  --source .humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/scratch/source_definition.json \
  --staged .humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/scratch/staged_definition.json \
  --expect-source-sha 3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa \
  --expect-staged-sha e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387 \
  --tool-head <H3> \
  --emit .humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/legacy_comparator_reproduction.json
```

**Tool behavior, in order:** parse args (six flags exactly) → format-check `--tool-head` → read `executed_at` → read and SHA both scratch files, refuse drift (`SOL_LEGACY_REPRO_INPUT_DRIFT`) → `json.loads` both (plain loads, faithful to legacy parse; non-object root or parse failure → `SOL_LEGACY_REPRO_PARSE_ERROR`, unreachable on sealed bytes) → apply legacy normalization exactly (for each of `hf_id`, `op_type`, `custom_inputs_entrypoint`: a **present** value equal to `""` is set to `None`; missing keys are never inserted) → strict dict equality → compute sorted `differing_top_level_keys` (symmetric key difference plus unequal shared keys) and RFC 6901 pointers → emit truthfully. The tool emits whatever it computed; the **schema consts** enforce the expected verdict (`equal: false`, keys `["op_type"]`, pointers `["/op_type"]`, `maps_to_refusal_code: "sol_stage_source_mismatch"`), so an unexpected verdict fails validation at the builder. Emission is create-exclusive (`open('xb')`; existing file → `SOL_LEGACY_REPRO_EMIT_EXISTS`). Serialization: UTF-8, `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` plus one trailing LF. On success: exit 0 and print exactly one stdout line `legacy_compare_9b8e0e40: EMITTED <emit path>`.

**Record shape (complete):**

```json
{
  "record_type": "legacy_comparator_reproduction",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "rule_id": "SOL_DEF_LEGACY_COMPARE_9B8E0E40",
  "executed_at": "<self-clocked RFC 3339 UTC Z, microseconds>",
  "no_gpu": true,
  "pure_cpu": true,
  "capture_code_imported": false,
  "tool": {
    "path": "tools/local_labels/legacy_compare_9b8e0e40.py",
    "sha256": "<self-computed sha256 of the tool file>",
    "head": "<H3 from --tool-head>"
  },
  "inputs": {
    "source_scratch_path": ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/scratch/source_definition.json",
    "source_sha256_expected": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
    "source_sha256_observed": "<computed; must equal expected>",
    "staged_scratch_path": ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/scratch/staged_definition.json",
    "staged_sha256_expected": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
    "staged_sha256_observed": "<computed; must equal expected>",
    "source_origin": { "…exactly as Section 4…" },
    "staged_origin": { "…exactly as Section 4…" }
  },
  "verdict": {
    "equal": false,
    "differing_top_level_keys": ["op_type"],
    "difference_pointers": ["/op_type"],
    "maps_to_refusal_code": "sol_stage_source_mismatch"
  },
  "python_version": "<MAJOR.MINOR.MICRO from sys.version_info>",
  "command_argv": ["<argv[0]>", "--source", "…", "--staged", "…", "--expect-source-sha", "…", "--expect-staged-sha", "…", "--tool-head", "<H3>", "--emit", "…"]
}
```

**`command_argv` canonical representation (Question 6):** the JSON array of `sys.argv` verbatim, exactly 13 elements. Element 0 is the module file path as Python reports it under `-m` (schema: string matching `tools/local_labels/legacy_compare_9b8e0e40\.py$`); elements 1–9 and 11–12 are consts equal to the bound flag/value strings above; element 10 matches `^[0-9a-f]{40}$`. The builder additionally asserts `command_argv[10] == tool.head` and `command_argv[1:] ==` the bound 12-element tail with H3 substituted. No `shlex`, no joined string, no whitespace rules needed.

## 6. EXACT SCHEMA FILE SET

Six schema files, all Draft 2020-12, `additionalProperties: false` throughout, all purged of `deterministic_v1_comparator_over_sealed_bytes` and `ANNEX_C_AMEND_R28_PREP070_RECON_V1` outside `not`-guards:

1. `workloads/simbench_bw/schemas/runner_refusal_reconstruction.schema.json` — amended in place to the V2 §5.1 shape and §5.6 rules (unchanged by this correction beyond the purge).
2. `workloads/simbench_bw/schemas/phase_reconstruction.schema.json` — amended in place to V2 §5.2/§5.6 (`interval_type` const `non_certifying_reconstruction`; anchors; `gpu_execution.occurred` const false).
3. `workloads/simbench_bw/schemas/gpu_process_observation_late.schema.json` — amended in place to V2 §5.5 (`certifying` const false, `late` const true, `raw_output_file` const `gpu_process_observation_late.raw.txt`).
4. `workloads/simbench_bw/schemas/preparation_failure_no_gpu_attestation_v2.schema.json` — amended in place to the V2 §5.3 shape: `record_version` const 2; `amendment_id` const `ANNEX_C_AMEND_R28_PREP070_RECON_V2`; `certifying_elements` const exactly `["annex_c_adjudication", "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes", "journal_forensic_absence_scan"]`; `journal_forensic_absence_scan` consts (`[]`, `false`, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, `true`, `22`, `3855797`, `ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`); `non_certifying_elements` const `["gpu_process_observation_late"]`; `late_observation_reference` `{file: const "gpu_process_observation_late.json", sha256: 64-hex, certifying: const false}`; `reproduction_reference` `{file: const "legacy_comparator_reproduction.json", sha256: 64-hex}` **required**; `campaign_rows` const 0; `capture_supersession_attempts_consumed` const 0.
5. `workloads/simbench_bw/schemas/preparation_failure_ledger_entry.schema.json` — amended per Section 7.
6. **New:** `workloads/simbench_bw/schemas/legacy_comparator_reproduction.schema.json` — validates Section 5's record: consts for every constant field; `^[0-9a-f]{64}$` for hashes; `^[0-9a-f]{40}$` for `tool.head`; RFC 3339 `Z$` pattern for `executed_at`; `^\d+\.\d+\.\d+$` for `python_version`; the 13-element `prefixItems` array for `command_argv` (`items: false`, `minItems: 13`, `maxItems: 13`); `verdict` consts as in Section 5. Hex/timestamp/const patterns make every `TO_BE_SEALED` placeholder structurally unshippable.

The five drafts are retained and amended in place — none is discarded (Question 14).

## 7. EXACT LEDGER COMPATIBILITY AND FIELD SHAPES

**Entry-level `oneOf` with two complete arms:**

**Arm A — contemporaneous (all future incidents, re-enacted verbatim):** exactly the H2-committed 28-field entry shape with `no_gpu_attestation` = the existing `no_gpu_attestation_contemporaneous` definition, unchanged. Contains no `authority`, no `reconstruction_records`, no `supersedes`. The H2 runtime validator's closed field set remains correct for this arm.

**Arm B — reconstructed (allowlisted problem-070 incident, exactly once):** the same 28 fields **plus** exactly three required additions:

- `"supersedes": null` (const);
- `"reconstruction_records"`: order-fixed five-element array, each `{path: const, sha256: 64-hex}`, closed, in this exact order:
  1. `chunk_context/reconstructed/legacy_comparator_reproduction.json`
  2. `chunk_context/reconstructed/runner_refusal_reconstruction.json`
  3. `chunk_context/reconstructed/phase_reconstruction.json`
  4. `chunk_context/reconstructed/gpu_process_observation_late.json`
  5. `chunk_context/reconstructed/preparation_failure_no_gpu_attestation_v2.json`
- `"authority"` (closed object):

```json
"authority": {
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "supersession_status": "superseded_by_v2",
  "corrected_adjudication_sha256": "<sha256 of adjudication_r28_prep070_reconstruction_v2.md>",
  "superseded_adjudication_sha256": "<sha256 of adjudication_r28_prep070_reconstruction_superseded_v1.md>",
  "correction_id": "R28_PREP070_RECON_V2_CORR1",
  "correction_adjudication_sha256": "<sha256 of this correction as saved>"
}
```

`amendment_id`, `supersession_status`, and `correction_id` are consts; the three SHAs are required 64-hex and the builder refuses if any two are equal.

Arm B's `no_gpu_attestation` field embeds, value-for-value, the same JSON object as the standalone `preparation_failure_no_gpu_attestation_v2.json` record (Section 6 item 4 shape). The builder parses both and refuses on any tree inequality (`PREPARATION_LEDGER_INCONSISTENT`). `campaign_rows: 0` and `capture_supersession_attempts_consumed: 0` therefore travel inside the attestation and are **not** duplicated at entry top level. The interval/`historical_event_interval` `$defs` remaining from the V1-era draft are deleted from the ledger schema.

**`prior_preparation_failures` (Question 12):** absent from every ledger entry, both arms — a row cannot reference itself before its `entry_id` exists. The **replacement bootstrap journal** (`journal_start.json` of the new attempt) carries `"supersedes": null` and `"prior_preparation_failures": [{"ledger_entry_id": "<uuid4 of this row>", "ledger_path": "<resolved ledger path>", "archive_path": "<resolved archive path>"}]` — exactly the output of the H2-implemented `preparation_failure_backrefs()`, a one-element array for problem-070. DONE/accounting closure per Annex C §5(6) is unchanged.

The chunk-context copy step of the archive places into `chunk_context/reconstructed/`: the five records, `legacy_comparator_source_excerpt.py.txt`, `gpu_process_observation_late.raw.txt`, and the `scratch/` directory (copies; the construction-site originals are retained). All are hashed into the archive `inventory.json` under the existing inventory discipline; the excerpt, raw output, and scratch copies are additionally hash-sealed inside the records that cite them, so bidirectional closure holds without listing them in `reconstruction_records`.

## 8. EXACT RLCR SESSION-LOG PATH

**Path:** `.humanize/rlcr/2026-07-08_11-39-13/session_log.jsonl`. Format: one JSON object per line, serialized `sort_keys=True, separators=(",", ":")`, UTF-8, LF, appended with fsync of file and parent directory. Entries are never edited or reordered. All V2 references to "the RLCR session log" mean this file. `round-28-summary.md` remains narrative; its existing `## Session Log` bullets are historical and are never rewritten.

**Required entries, in order:**

1. On saving this correction (before implementation):
```json
{"event":"adjudication_saved","artifact":"adjudication_r28_prep070_reproduction_contract_correction.md","sha256":"<computed>","status":"binding_supplement","supplements_amendment_id":"ANNEX_C_AMEND_R28_PREP070_RECON_V2","correction_id":"R28_PREP070_RECON_V2_CORR1"}
```
2–3. After H3, the two V2 §10-step-5 objects verbatim (superseded-V1 event, then binding-V2 event with `"h3":"<h3_hash>"`).
4–10. One `{"event":"reconstruction_record_created","artifact":"<construction-site-relative path>","sha256":"<64-hex>"}` per artifact in creation order: `legacy_comparator_source_excerpt.py.txt`, `legacy_comparator_reproduction.json`, `runner_refusal_reconstruction.json`, `phase_reconstruction.json`, `gpu_process_observation_late.raw.txt`, `gpu_process_observation_late.json`, `preparation_failure_no_gpu_attestation_v2.json`.
11. After the archive: `{"event":"preparation_failure_archived","entry_id":"<uuid4>","archive_path":"<absolute>","inventory_sha256":"<64-hex>"}`.

The archive builder, before appending entry 11, re-parses the log and refuses `RLCR_SESSION_LOG_INCONSISTENT` if any prior required entry is missing or unparseable.

## 9. REQUIRED TEST DELTA

V2 §8/§9 remain binding with these amendments and additions (no-GPU throughout):

**Amended:** V2 P5 (`test_reproduction_record_expected_verdict`) now also validates the emitted record against the new schema and exercises the builder assertions of Section 3 items 7–11 on a fixture repo. V2 P6/P7 and N13 extend to the new schema file. V2 N6 (static import scan) now asserts the exhaustive six-module set — any other import fails, explicitly including `os`, `subprocess`, `shlex`, `time`.

**New positive:** P16 self-clock: two tool runs yield distinct, parseable `executed_at`, both inside the builder sandwich. P17 argv: emitted `command_argv` equals observed `sys.argv`; schema rejects a permuted or 12/14-element array. P18 tool-blob binding: `tool.sha256` equals on-disk and `git show H3:` blob hashes in a fixture repo. P19 source-origin: fixture prepare manifest pinning an external file; builder verification and copy succeed; origin consts validate. P20 ledger Arm A regression: a synthetic future contemporaneous entry (H2 shape) validates against schema and runtime validator with no reconstructed-only fields. P21 ledger Arm B golden fixture: embedded attestation value-equals the standalone file; five records in bound order; authority block with three distinct SHAs. P22 session log: append/fsync round-trip; all required event shapes parse.

**New negative:** N14 `--executed-at` (or any seventh flag) → argparse error. N15 malformed `--tool-head` → `SOL_LEGACY_REPRO_HEAD_INVALID`; dirty tree / HEAD mismatch / parent≠H2 / wrong subject → `SOL_LEGACY_REPRO_HEAD_MISMATCH`. N16 tool-file vs blob drift → `SOL_LEGACY_REPRO_TOOL_DRIFT`. N17 `executed_at` outside the sandwich or ≤ `2026-07-10T20:15:42Z` → `SOL_LEGACY_REPRO_TIME_INVALID`. N18 argv tail mismatch or `command_argv[10] ≠ tool.head` → `SOL_LEGACY_REPRO_ARGV_MISMATCH`. N19 string-typed `command_argv` → schema invalid. N20 manifest hash drift, pin mismatch, or missing/drifted external source file → `SOL_LEGACY_REPRO_INPUT_DRIFT` before any tool run. N21 contemporaneous entry carrying `authority`/`reconstruction_records`/`supersedes`, or reconstructed entry missing any of them → schema invalid; embedded-vs-file attestation inequality → `PREPARATION_LEDGER_INCONSISTENT`. N22 pre-existing `--emit` target → `SOL_LEGACY_REPRO_EMIT_EXISTS`, no overwrite. N23 corrupted/truncated `session_log.jsonl` at archive time → `RLCR_SESSION_LOG_INCONSISTENT`.

## 10. UPDATED SEQUENCING

Binding order; no step begins before the prior completes:

0. Save this correction verbatim (path and mode per header); create `session_log.jsonl`; append entry 1.
1. Amend the five draft schemas per Section 6; add the sixth schema; purge withdrawn literals outside `not`-guards.
2. Add `tools/local_labels/legacy_compare_9b8e0e40.py` per Sections 3 and 5.
3. Extend the archive/record builders and validators per V2 §7 item 3 as corrected here (allowlist, placeholder rejection, context-mode, plus Sections 3–4 and 7–8 checks and refusal codes).
4. Retain V2 §7 item 4 unchanged (equivalence-attestation writers refuse `SOL_DEF_LEGACY_COMPARE_9B8E0E40` and `SOL_DEF_EQUIV_V1`).
5. Implement Section 9 tests; run the complete no-GPU battery green.
6. Commit **H3**: single commit atop H2, subject exactly as bound, clean tree after.
7. Save the superseded-V1 and binding-V2 adjudication artifacts, mode 0444; append session-log entries 2–3.
8. Construct records in the construction site, in order: scratch copies + verification (Amendment 2) → source-identity verification + excerpt → **reproduction run** (builder invokes the Section 5 command with `--tool-head` H3 after Section 3 checks) → `runner_refusal_reconstruction.json` → `phase_reconstruction.json` → late GPU observation + raw (the sole authorized GPU-adjacent read-only action before bootstrap) → attestation last; validate all against the six schemas; append entries 4–10.
9. Re-verify journal byte-identity (22 / 3,855,797 / `ff0ace5c…`); perform the Annex-C atomic archive with the Arm-B ledger entry; verify bidirectional closure; append entry 11.
10. Matrix regeneration at H3 → certifying H3 V2 audit → fresh preflight → one-cell SOL bootstrap → accounting exactly 2/120/0/0/0, integrity true → pinned `gpt-5.5:high` review (wait-and-retry on any safety/capacity check) → remaining captures toward exactly 122 terminal published outcomes.

## 11. ANSWERS TO QUESTIONS 1–15

1. **Now exhaustive:** exactly `argparse`, `datetime`, `hashlib`, `json`, `pathlib`, `sys`; nothing else, ever.
2. Self-clocked inside the tool per Section 3; caller-supplied timestamps structurally impossible; builder sandwich + post-H2 checks.
3. Via required `--tool-head`; the tool format-checks only; the builder git-verifies per Section 3 and post-verifies the record.
4. **Yes.** Exact builder checks: Section 3 items 1–11.
5. Reissued in Section 5, including CWD binding and `--tool-head`.
6. JSON array of `sys.argv` verbatim, 13 elements, schema-pinned per Section 5.
7. **Confirmed:** source from the external manifest-pinned file after manifest-hash, pin-equality, and byte-hash verification; staged from the journal member; the false sentence is struck and replaced by Amendment 2's exact text.
8. **Yes:** `inputs.source_origin` and `inputs.staged_origin`, exact fields in Section 4, builder-verified before invocation.
9. `workloads/simbench_bw/schemas/legacy_comparator_reproduction.schema.json`; six affected schema files total, five drafts retained.
10. **Yes, entry-level `oneOf`** per Section 7; Arm A re-enacts the H2 contemporaneous shape verbatim as the sole future path.
11. The exact ordered five-element array in Section 7.
12. Ledger Arm B: `"supersedes": null` (const); **no** `prior_preparation_failures` in any ledger entry; the replacement journal carries `supersedes: null` plus the one-element `prior_preparation_failures` back-reference exactly as in Section 7.
13. `.humanize/rlcr/2026-07-08_11-39-13/session_log.jsonl`, append-only fsync'd JSONL per Section 8.
14. **Confirmed:** five drafts retained and amended in place; the reproduction schema is added; nothing is silently replaced or discarded.
15. **Confirmed:** H3, archive, matrix regeneration, certifying audit, preflight, bootstrap, captures, and all GPU work (except the step-8 read-only late observation) remain blocked until this correction is saved, implemented, tested green, and H3 is committed and logged — in the Section 10 order.

## 12. FORBIDDEN SHORTCUTS

All V2 §12 prohibitions remain in force verbatim, and additionally: do not add `--executed-at` or accept any caller-supplied timestamp; do not import `os`, `subprocess`, `shlex`, `time`, or anything beyond the six permitted modules in the tool; do not have the tool run git or read environment variables; do not describe the source definition as a journal member anywhere; do not serialize `command_argv` as a string or reconstruct shell quoting; do not run the reproduction before H3 exists or pass any `--tool-head` other than verified current HEAD; do not overwrite an existing emit target or edit an emitted record; do not put `prior_preparation_failures` in a ledger row or make reconstructed-only fields required on the contemporaneous arm; do not treat `round-28-summary.md` as the session log or rewrite any appended log line; do not mint a V3 amendment ID or change any live `ANNEX_C_AMEND_R28_PREP070_RECON_V2` const — this correction is cited only through the `authority.correction_id`/`correction_adjudication_sha256` fields and the session log; do not skip the tool-blob equality check; and do not switch, downgrade, or substitute the pinned `gpt-5.5:high` review path — on any safety or capacity check, wait and retry the identical path.

**End of binding supplemental correction.** Codex may resume at Section 10 step 0 and must stop and re-escalate on any further material contradiction rather than guessing or weakening a check.
