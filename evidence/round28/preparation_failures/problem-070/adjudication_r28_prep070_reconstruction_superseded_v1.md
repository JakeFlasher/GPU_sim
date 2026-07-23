# FABLE-5 BINDING ADJUDICATION — Round 28: Missing Historical Context Records Before Preparation-Failure Archive

Session `b41e0614-a371-4041-b2ed-0d0aecfda548`. This adjudication is binding and self-contained. All Annex A, B, C, and D clauses not expressly amended below remain in full force, including the pinned `gpt-5.5:high` review path with wait-and-retry on any safety/capacity check and no model substitution.

---

## 1. DECISION

The archive is **not** permanently blocked. Codex must implement **Option A′**, a precisely specified alternative defined as:

- **Option A** (honest late reconstruction with explicit provenance) as the operative mechanism;
- merged with **Option D**'s typed-interval timestamp semantics, with the upper-bound problem resolved by binding the interval to the H2 committer timestamp;
- incorporating **Option B**'s substitution principle *only* for the two historically unrecoverable proof elements (Annex C elements 4 and 6), narrowly allowlisted to this exact incident identity;
- **Option C** (old-HEAD reproduction) is *permitted as optional, non-authoritative corroboration* under the constraints in §12-Q6, and is **not required**;
- **Option E** is rejected;
- all fabrication paths (backdated timestamps, mtimes presented as event times, late observations presented as contemporaneous, silent omission) are rejected and enumerated as forbidden in §11.

A new clean commit **H3** is required, because the implemented archive tool's expected contemporaneous shapes cannot be truthfully populated. `replacement_head` becomes H3. All post-archive checklist steps that Annex D bound to H2 are re-bound to H3.

## 2. RATIONALE

The purpose of Annex C's six-element no-GPU proof is to establish two facts: (i) the runner refused at `prepare_validate` with `sol_stage_source_mismatch`, and (ii) no GPU-phase work occurred. Both facts are independently established by evidence that already exists and is sealed:

- The refusal classification is a deterministic function of sealed bytes: the source definition (`3e0aceb3…`) and staged definition (`e7526755…`) differ, and the frozen V1 comparator maps that divergence to `sol_stage_source_mismatch`. The frozen V1 blocking artifact and the H1 V2 audit at the same identities confirm the divergence class (axis-description null-vs-missing, normalized only under `SOL_DEF_EQUIV_V2`). Annex C additionally bound the observed refusal as an adjudicated fact.
- The no-GPU fact is established structurally: sealed `capture_started` journal with no `journal_complete.json`; prepare stdout parsing to exactly the preparation manifest; empty stderr (SHA `e3b0c442…`, the empty-input SHA-256); zero forbidden-artifact matches; a 22-file inventory containing no eval, NCU, NVBit, decoded, control, accepted-launch, or launch records; and the static old-HEAD `ProblemPackager.execute()` proof that the driver is written but never executed.

Elements 4 and 6 of Annex C's proof were *recording-format* requirements presupposing files that were never persisted. The underlying facts they were meant to record are provable without them. Fabricating the files would destroy the campaign's evidentiary integrity to satisfy a formality; permanent blockage would sacrifice 121 captures to a formality that adds no evidentiary value beyond what sealed bytes already prove. Honest, explicitly-labeled, non-certifying reconstruction — with the certifying weight carried by the sealed structural evidence — preserves both integrity and progress. The exception is bound to a single-incident allowlist so it can never become a general weakening: every future preparation failure must persist contemporaneous records via the shapes already implemented in H2.

## 3. PRIOR-ADJUDICATION CLAUSES AMENDED

Amendment ID: `ANNEX_C_AMEND_R28_PREP070_RECON_V1`. Scope: exactly one incident (allowlist in §12-Q11). Amended clauses:

1. **Annex C proof element 4** (runner-refusal record with timestamp + phase log): replaced, for this incident only, by `runner_refusal_reconstruction.json` and `phase_reconstruction.json` with interval semantics per §4–§5. The certifying basis becomes the deterministic comparator relation over sealed SHAs plus the Annex C adjudicated classification.
2. **Annex C proof element 6** (recorded post-refusal GPU observation): replaced, for this incident only, by the structural no-GPU proof (elements 1, 2, 3, 5, unchanged and still mandatory) plus an honestly late `gpu_process_observation_late.json` marked `contemporaneous: false`.
3. **Annex C archive protocol**: extended to place reconstruction records exclusively under `chunk_context/reconstructed/`, and to extend the ledger's `no_gpu_attestation` object per §5.
4. **Annex C "archive after H2 exists"** and **Annex D post-H2 sequencing**: re-bound from H2 to H3. H2 remains a valid historical preparatory commit in ancestry; authority for the remaining checklist transfers to H3.

Not amended: Annex A in its entirety (g++-14 restoration, fingerprint `0d7c673c…`, `NVCC_CCBIN` pin, ambient-conflict refusal, AC-9 routing); Annex B in its entirety (single-preflight identity, 900.000 s inclusive freshness with 2 s skew guard, supersession rules, wrapper-based shell mitigation, no edits to `/etc/profile.d/cuda.sh`); Annex C's classification, counts (preparation-failure count 1, zero campaign rows, no supersession consumption, `supersedes: null`, `prior_preparation_failures` carried by the replacement journal), inventory/atomicity/fsync/no-rollback requirements; Annex D's `SOL_DEF_EQUIV_V2` rule, V1 freeze, byte-immutability of source and staged definitions; authority totals 344/122/222/0, budgets 122, chunks 18, runnable SOL 74; bootstrap accounting target 2/120/0/0/0 with integrity true; pinned `gpt-5.5:high` review.

## 4. EXACT HISTORICAL-EVIDENCE CONTRACT

1. **Certifying elements** (carry the archive's evidentiary weight): sealed `journal_start.json` (`6b6dcce8…`) with no `journal_complete.json`; prepare stdout (`ca3249b3…`, 4,084 bytes) parsing to exactly the preparation manifest (`d9c2fd5e…`); empty stderr (`e3b0c442…`); forbidden-artifact scan `[]`; 22-file inventory, 3,855,797 bytes, domain SHA `ff0ace5c…`; static old-HEAD `ProblemPackager.execute()` no-execution proof; deterministic V1 comparator divergence over sealed source/staged SHAs; Annex C adjudication (`b9bc56d0…`) binding the observed refusal.
2. **Non-certifying elements** (explicitly labeled, additive only): the historical-event interval; the late GPU observation; any optional Option C reproduction.
3. **Timestamp truth rules**: every `recorded_at_utc`/`observed_at_utc` is the actual wall-clock instant of record creation/observation, taken at creation time. The historical refusal instant is represented *only* as a typed non-certifying interval: lower bound = the exact stat-derived mtime of the sealed prepare stdout file, read at reconstruction time (expected ≈ `2026-07-10T13:34:52.699Z`; the approximation in this request is not authoritative — read the actual stat); upper bound = the committer timestamp of H2 `823f2d60…` (hash-bound via git, indisputably after the incident because H2 postdates Annexes C/D which adjudicated it). `exact_time_known: false` always. No field named `refused_at_utc` may exist in any reconstruction record; the schemas forbid it.
4. **Journal immutability**: the 22-file journal is moved byte-identical. Inventory count/bytes/SHA verified immediately pre-move and post-move against 22 / 3,855,797 / `ff0ace5c…`. Reconstruction records never enter the journal.
5. **One-shot scope**: reconstructed-mode archival is valid only for the allowlist tuple in §12-Q11 and only once. All future preparation failures use the contemporaneous shapes already implemented at H2, with exact event-time `refused_at_utc` recorded at event time.

## 5. EXACT JSON/SCHEMA SHAPES

All three records are validated against Draft 2020-12 schemas with `additionalProperties: false` and the closed field sets shown. Placeholders in angle brackets are the only fields Codex fills, from the exact sources stated; nothing may be inferred beyond them.

`chunk_context/reconstructed/runner_refusal_reconstruction.json`:

```json
{
  "schema_version": 1,
  "record_kind": "runner_refusal_reconstruction",
  "context_origin": "reconstructed_after_event",
  "certifying": false,
  "recorded_at_utc": "<ACTUAL creation instant, RFC3339 UTC>",
  "runner_refusal_code": "sol_stage_source_mismatch",
  "reason_code": "STAGE_SEMANTIC_VALIDATION_FAILED",
  "refusal_phase": "prepare_validate",
  "historical_event_interval_utc": {
    "lower_bound_utc": "<EXACT stat mtime of sealed prepare stdout>",
    "lower_bound_basis": "prepare_stdout_mtime",
    "upper_bound_utc": "<H2 committer timestamp from git show -s --format=%cI 823f2d60...>",
    "upper_bound_basis": "h2_commit_timestamp",
    "exact_time_known": false,
    "certifying": false
  },
  "authority_basis": ["annex_c_adjudication", "deterministic_v1_comparator_over_sealed_bytes"],
  "annex_c_adjudication_sha256": "b9bc56d06ddbed14ca28df02012077a533c9b846a90f2f08ffdcfc9c6c52b568",
  "source_definition_sha256": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
  "staged_definition_sha256": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
  "cell_id": "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation",
  "original_head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
  "session_id": "f10de1cc-612e-43e3-b51f-ce1c9985e281",
  "admission_run_id": "0ff22c451824e77fa4ec20392866c50a7356b6de4e1e986ba788bd1b5ee1a658"
}
```

`chunk_context/reconstructed/phase_reconstruction.json`:

```json
{
  "schema_version": 1,
  "record_kind": "phase_reconstruction",
  "context_origin": "reconstructed_after_event",
  "certifying": false,
  "recorded_at_utc": "<ACTUAL creation instant>",
  "phases": [
    {
      "phase": "prepare",
      "status": "complete",
      "evidence_basis": [
        "prepare_stdout_sha256:ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc",
        "preparation_manifest_sha256:d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7",
        "staged_module_sha256:017ec9c8c977078c141e8d7d4fb58a67501550aa5540c220dbd97a94b8e1ee96"
      ]
    },
    {
      "phase": "prepare_validate",
      "status": "refused",
      "evidence_basis": ["annex_c_adjudication", "deterministic_v1_comparator_over_sealed_bytes"]
    }
  ],
  "phases_after_refusal": [],
  "no_subsequent_phase_basis": [
    "journal_complete_absent",
    "forbidden_artifact_scan_zero_matches",
    "inventory_domain_sha256:ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36",
    "empty_prepare_stderr_sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "static_problempackager_execute_no_execution_proof"
  ],
  "historical_event_interval_utc": { "<identical object to the refusal record>": "..." }
}
```

(The interval object must be byte-identical across both records; the schema repeats the full closed shape.)

`chunk_context/reconstructed/gpu_process_observation_late.json`:

```json
{
  "schema_version": 1,
  "record_kind": "gpu_process_observation_late",
  "context_origin": "late_observation",
  "contemporaneous": false,
  "certifying": false,
  "observed_at_utc": "<ACTUAL observation instant>",
  "observation_method": "<truthful method actually used, e.g. nvidia-smi compute-apps query>",
  "remaining_gpu_processes": ["<truthful current result; empty or not>"],
  "historical_no_gpu_basis": [
    "journal_complete_absent",
    "prepare_stdout_equals_preparation_manifest",
    "empty_prepare_stderr",
    "forbidden_artifact_scan_zero_matches",
    "inventory_contains_no_execution_artifacts",
    "static_problempackager_execute_no_execution_proof"
  ],
  "limitation": "Observation postdates the historical refusal and does not certify process state at the historical event time."
}
```

`remaining_gpu_processes` must be the truthful current result even if non-empty; the historical no-GPU conclusion does not depend on it.

Ledger `no_gpu_attestation` object (embedded in the fsynced append-only UUID4 ledger entry, alongside all existing Annex C ledger fields):

```json
{
  "schema_version": 2,
  "attestation_kind": "pre_gpu_preparation_failure",
  "context_origin": "reconstructed_after_event",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V1",
  "amendment_adjudication_sha256": "<SHA-256 of this adjudication as saved to the session dir>",
  "certifying_elements": [
    "sealed_capture_started_journal_no_completion",
    "prepare_stdout_equals_manifest",
    "empty_prepare_stderr",
    "forbidden_artifact_scan_zero_matches",
    "static_problempackager_execute_no_execution_proof",
    "deterministic_v1_comparator_over_sealed_bytes",
    "annex_c_adjudicated_refusal_classification"
  ],
  "non_certifying_elements": ["historical_event_interval", "late_gpu_process_observation"],
  "reconstruction_records": [
    {"path": "chunk_context/reconstructed/runner_refusal_reconstruction.json", "sha256": "<computed>"},
    {"path": "chunk_context/reconstructed/phase_reconstruction.json", "sha256": "<computed>"},
    {"path": "chunk_context/reconstructed/gpu_process_observation_late.json", "sha256": "<computed>"}
  ],
  "historical_event_interval_utc": { "<same interval object>": "..." },
  "journal_inventory": {
    "file_count": 22,
    "total_bytes": 3855797,
    "inventory_domain_sha256": "ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36"
  }
}
```

If the optional Option C reproduction is performed, add `{"path": "chunk_context/reconstructed/comparator_reproduction.json", "sha256": "<computed>"}` to `reconstruction_records` and `"comparator_reproduction"` to `non_certifying_elements`; its record carries `record_kind: "non_authoritative_reproduction"`, `historical: false`, `certifying: false`, current timestamps, the exact command line, environment guard evidence, verdict, and input SHAs verified before and after.

## 6. REQUIRED CODE CHANGES — ORDERED

1. Add Draft 2020-12 schemas (`additionalProperties: false`) for the three reconstruction record kinds and the `no_gpu_attestation` v2 object. Schemas must structurally forbid `refused_at_utc`, forbid `contemporaneous: true` in `gpu_process_observation_late`, forbid `exact_time_known: true`, and forbid `certifying: true`.
2. Extend the archive tool with `--context-mode {contemporaneous, reconstructed}`. Contemporaneous mode is unchanged and remains the sole path for all future incidents. Reconstructed mode: (a) gated by a single embedded allowlist entry equal to the tuple in §12-Q11, compared field-for-field; (b) validates all three records against the new schemas; (c) independently recomputes and verifies the interval bounds (stat of sealed prepare stdout; `git` committer timestamp of `823f2d60…`) and refuses on mismatch; (d) refuses if any legacy filename (`runner_refusal_record.json`, `phase_log.json`, `gpu_process_observation.json`) exists in or would be written to the journal or archive; (e) copies the reconstruction records into `chunk_context/reconstructed/` with SHAs; (f) writes the v2 `no_gpu_attestation` into the ledger entry with `replacement_head` = H3. Refusal codes: `PREP_ARCHIVE_RECON_ALLOWLIST_MISMATCH`, `PREP_ARCHIVE_RECON_SCHEMA_INVALID`, `PREP_ARCHIVE_RECON_INTERVAL_INVALID`, `PREP_ARCHIVE_RECON_JOURNAL_CONTAMINATION`.
3. Enforce one-shot semantics: after a sealed reconstructed-mode ledger entry exists for this incident, any further reconstructed-mode invocation refuses (no-replace publication and ledger closure already implemented at H2 apply).
4. Add the positive and negative tests of §7–§8.
5. Rerun the complete no-GPU battery from §3 of the request (all suites, schema checks, shell syntax, `git diff --check`, compilation).
6. Commit exactly one clean commit H3 (suggested subject: `feat(prep_archive): reconstructed historical-context mode, ANNEX_C_AMEND_R28_PREP070_RECON_V1`). Git status must be empty afterward.

No changes to AC-9, comparator V1/V2, adapters, runners, DONE, accounting semantics, or any Annex A/B machinery.

## 7. REQUIRED POSITIVE TESTS

- Golden fixtures for all three reconstruction records validate against the new schemas.
- End-to-end reconstructed-mode archive on a fixture replicating the exact 22-file inventory (temp filesystem): pre-move verification, same-device check, atomic rename, post-move verification, `inventory.json` with per-file origins, chunk-context copy and hashing, fsynced UUID4 ledger append containing the v2 attestation, `replacement_head` = fixture H3, `supersedes: null`, `prior_preparation_failures` propagation readable by the replacement-journal precheck.
- Allowlist exact-tuple acceptance.
- Interval verification: lower bound equals stat mtime of the fixture stdout; upper bound equals fixture-H2 committer timestamp; identical interval object across both records.
- One-shot: second invocation after success refuses.
- Contemporaneous mode regression: a synthetic future incident with exact event-time `refused_at_utc` still archives correctly.
- Both new schemas pass Draft 2020-12 meta-validation.

## 8. REQUIRED NEGATIVE TESTS

- Parameterized allowlist mutation over every tuple member (cell ID, HEADs, matrix SHA/seal, preflight SHA, session/admission IDs, inventory SHA/count/bytes, source/staged/module SHAs) → `PREP_ARCHIVE_RECON_ALLOWLIST_MISMATCH`.
- `refused_at_utc` present in any reconstruction record → schema invalid.
- `contemporaneous: true`, `certifying: true`, or `exact_time_known: true` anywhere → schema invalid.
- Upper bound earlier than lower bound; lower bound not equal to observed stdout mtime; interval objects differing between the two records → `PREP_ARCHIVE_RECON_INTERVAL_INVALID`.
- Reconstruction file inside the journal directory, or inventory drift (21/23 files, byte drift, SHA drift) → refuse before any move.
- Legacy filenames written or present in reconstructed mode → `PREP_ARCHIVE_RECON_JOURNAL_CONTAMINATION`.
- `journal_complete.json` present → refuse (not a preparation failure).
- Reconstructed mode invoked for any other cell/session/incident → refuse.
- Post-move verification failure → no rollback occurs, dedicated failure state recorded (Annex C no-silent-rollback preserved).
- Anything under `preparation_failures/` or `superseded/` offered for publication/accounting → rejected (re-assert existing closure tests).

## 9. COMMIT/SEQUENCING IMPACT

H2 remains a valid historical preparatory commit in ancestry; it is not superseded as evidence, but checklist authority transfers to H3. The H1 non-certifying V2 audit (`ea51b0c6…`) remains valid and must **not** be regenerated — it binds H1 identities, which are unchanged. The frozen V1 blocking artifact (`b6dd22ad…`, mode 0444) remains untouched. The full no-GPU battery must be rerun at H3. `replacement_head` in the ledger is H3. Binding order:

1. Implement §6 changes; run §7–§8 tests plus the full battery; commit clean H3.
2. Save this adjudication into `.humanize/rlcr/2026-07-08_11-39-13/`, mode 0444, record its SHA-256 for the ledger field.
3. Execute the archive (checklist in §10).
4. Regenerate authority/matrix at H3; verify exact cardinalities 344/122/222/0, budgets 122, chunks 18, runnable SOL 74.
5. Produce the clean-tree certifying `SOL_DEF_EQUIV_V2` audit at H3.
6. Create a fresh preflight (Annex B freshness rules unchanged) and run the one-cell SOL bootstrap; accounting must read exactly 2 accounted / 120 missing / 0 duplicates / 0 unexpected / 0 leaks, integrity true.
7. Pinned `gpt-5.5:high` review; on any safety/capacity check, wait and retry the identical path.

## 10. UPDATED ARCHIVE CHECKLIST

1. Confirm clean tree at H3; confirm no GPU work has occurred this session.
2. Create the three reconstruction records per §5 with actual current `recorded_at_utc`/`observed_at_utc`; derive the interval bounds from the exact stdout stat and the H2 committer timestamp; validate against schemas.
3. (Optional) Perform the Option C reproduction under §12-Q6 constraints.
4. Perform the late GPU observation truthfully.
5. Re-verify the source journal read-only: exists; `journal_complete.json` absent; 22 files; 3,855,797 bytes; inventory SHA `ff0ace5c…`; forbidden-artifact scan `[]`.
6. Same-device check for the atomic rename.
7. Run the archive tool in reconstructed mode: allowlist match → pre-move verification → atomic rename of the journals parent per Annex C → post-move re-verification (22 / 3,855,797 / `ff0ace5c…`) → `inventory.json` → copied and hashed chunk context including `chunk_context/reconstructed/` → fsynced append-only UUID4 ledger entry with v2 `no_gpu_attestation`, `replacement_head` = H3, `supersedes: null`.
8. On any post-move failure: no silent rollback; stop and re-adjudicate.
9. Verify bidirectional ledger/archive closure and that nothing under `preparation_failures/` is publishable or countable.
10. Only then proceed to §9 steps 4–7.

## 11. FORBIDDEN SHORTCUTS

- Backdating any timestamp, or writing any `refused_at_utc` field in reconstruction context.
- Using file mtimes or birth times as exact event timestamps anywhere; they may appear only as interval bounds with explicit basis labels.
- Presenting the late GPU observation as contemporaneous, or omitting `contemporaneous: false`.
- Creating the three originally-expected files in their §5 (request) shapes with any timestamp value; those shapes are retired for this incident.
- Inserting, renaming, or modifying anything inside the 22-file journal; creating `journal_complete.json`; any inventory drift.
- Loading or executing the staged module `017ec9c8…`; any eval driver, warmup, candidate execution, NCU, NVBit, or control capture; any GPU work at all during archival.
- Mutating the immutable source/staged definitions or stage directory; Option C may use only verified copies in a scratch area.
- Proceeding to matrix regeneration, certifying audit, preflight, or bootstrap before the ledger entry is fsynced and verified.
- Publishing or counting anything under `preparation_failures/` or `superseded/`.
- Reusing the reconstruction allowlist for any other incident, cell, or session; treating any reconstruction record or interval as certifying.
- Editing `/etc/profile.d/cuda.sh`; weakening any check without adjudication; silent rollback after post-move failure.
- Switching or downgrading models; review remains pinned to `gpt-5.5:high` with wait-and-retry.

## 12. EXPLICIT ANSWERS TO QUESTIONS 1–12

**Q1.** Not permanently impossible. Annex C is amended narrowly per §3: a single-incident reconstruction/substitution rule is authorized because both underlying facts (refusal classification, no GPU work) are provable from sealed evidence plus prior adjudication; only the recording format is unrecoverable.

**Q2.** Option A′ as defined in §1 is binding: A merged with D's interval semantics, B's substitution principle limited to elements 4 and 6 under the exact allowlist, C optional and non-authoritative, E rejected.

**Q3.** Exact shapes and closed field sets are given in §5 and are binding as written.

**Q4.** `refused_at_utc` is removed entirely from this incident's reconstruction records — not required, and not permitted as null; the schemas forbid the field so no reader can mistake a value for an event time. The historical instant is expressed only as the typed interval: lower bound = exact stat-derived prepare-stdout mtime (basis `prepare_stdout_mtime`; permitted precisely because it is labeled a bound, never an event time); upper bound = H2 committer timestamp (basis `h2_commit_timestamp`, hash-bound). `exact_time_known: false` and `certifying: false` are mandatory constants. Current observation time is distinguished by dedicated fields (`recorded_at_utc`/`observed_at_utc`) that by schema definition mean creation/observation instants, plus mandatory `context_origin`, `contemporaneous: false`, and the fixed `limitation` sentence. For future incidents, contemporaneous `refused_at_utc` remains required and exact.

**Q5.** Annex C may serve as authority for the refusal code, and is corroborated by the deterministic V1-comparator relation over the sealed source/staged SHAs — a pure function of pinned bytes. Independent reproduction is not required; it is permitted as additive corroboration only.

**Q6.** If performed: a temporary detached read-only worktree at `9b8e0e40…` is permitted; copy (never move) the source and staged definition bytes to scratch, verifying `3e0aceb3…`/`e7526755…` before and after; run only the V1 prepare-validate stage-vs-source comparison entry point — never `ProblemPackager.execute()`, never the runner capture path; run with `CUDA_VISIBLE_DEVICES=""` and no NVCC-related environment; required output: verdict mapping to `sol_stage_source_mismatch`, exit status, stdout/stderr SHAs, command line, input SHAs; no-GPU proof: the path is pure CPU JSON comparison plus the recorded environment guard; label per §5 (`record_kind: "non_authoritative_reproduction"`, `historical: false`, `certifying: false`, current timestamps); store only under `chunk_context/reconstructed/`; remove the worktree afterward; the journal inventory SHA must be unaffected.

**Q7.** Yes. The archive is bound to inventory SHA `ff0ace5c…` with count 22 and bytes 3,855,797, verified pre-move and post-move. All reconstruction records live solely in copied `chunk_context/reconstructed/`; nothing is ever inserted into the old journal.

**Q8.** A new clean H3 commit is required. `replacement_head` = H3. The full no-GPU battery must be rerun at H3. The H1 non-certifying V2 audit remains unchanged and must not be regenerated. H2 remains a historical preparatory commit in ancestry; it is not evidence-superseded, but every remaining checklist step Annex D bound to H2 now binds H3.

**Q9.** The premise is rejected: no truthful population of the existing contemporaneous shapes is possible, because their `refused_at_utc` semantics encode an exact historical event time that was never recorded. A code change is therefore mandatory. Nevertheless, §4–§5 fully specify every truthful value and source so nothing is left to inference.

**Q10.** Confirmed. Matrix regeneration, the certifying audit, preflight creation, and the bootstrap remain strictly blocked until the archive ledger entry is fsynced and verified — and additionally until H3 exists with the full battery passed. Order is §9.

**Q11.** The allowlist tuple is binding and must match exactly: cell `sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation`; chunk `sol_execbench-000-bootstrap`; original HEAD `9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a`; matrix SHA `fd3a1868…` and seal `5449383…` (full values as pinned in Annex C); preflight SHA `3810d2b2…`; session `f10de1cc-612e-43e3-b51f-ce1c9985e281`; admission run `0ff22c45…`; inventory SHA `ff0ace5c…`, count 22, bytes 3,855,797; source `3e0aceb3…`, staged `e7526755…`, module `017ec9c8…`; journal-start SHA `6b6dcce8…`; prepare stdout SHA `ca3249b3…`; stderr SHA `e3b0c442…`; manifest SHA `d9c2fd5e…`; replacement head = H3 (must equal current git HEAD at archive time). Any mismatch refuses.

**Q12.** All items in §11 are forbidden, explicitly including: mtime-derived *exact* timestamps (bounds-with-basis only), backdating of any kind, silent omission of any proof element outside this amendment's explicit substitution, and presenting current process observations as historical or contemporaneous.

---

This adjudication is complete and implementation-ready. Codex should proceed with §6 step 1 and stop for a fresh Fable-5 adjudication if any material ambiguity arises during implementation, without guessing, weakening any check, or leaving the pinned `gpt-5.5:high` review path.
