# ADJUDICATION — Round 28 Staged-Definition Equivalence and Pre-GPU Attempt Preservation

## 1. DECISION

**Ambiguity A: Option A is binding** — narrow, closed canonicalization of optional-field representation, tightened into a fully specified type-strict comparison rule (`SOL_DEF_EQUIV_V1`, §3), **plus** two additive elements borrowed from Option C's spirit at near-zero cost: the pinned source bytes are copied into the stage as `definition.source.json`, and a definition-equivalence attestation (both SHAs, rule id, per-field representation states) is sealed into stage evidence. Option B is rejected: post-hoc replacement of a packager-produced artifact is in-place mutation of staged evidence and creates dual representations between compile-time and validation-time consumers. Full Option C is rejected as disproportionate; its useful residue is absorbed additively.

**Ambiguity B: Option 2 is binding** — the preserved attempt is classified as a **pre-GPU preparation attempt**, preserved through a new, precisely bounded **preparation-failure archive** protocol (§4–§5). Cross-HEAD supersession of *capture* journals (Option 1) remains **not permitted**: Annex B's C3 stays single-HEAD, single-matrix, exactly as designed. Option 3 is rejected (it would mint a supersession whose "current preflight" can never admit a successful attempt). Option 4 is rejected (a permanently `capture_started`-shaped journal left in an active-looking path with no ledger closure is a stranded attempt).

**Ambiguity C: Treatment 3 is binding** — the archive entry's reason code is `STAGE_SEMANTIC_VALIDATION_FAILED`; C3's `PREFLIGHT_EXPIRED_BEFORE_DONE` vocabulary is untouched and reserved for true capture supersessions. An additive `runner_refusal_code` field is required and must carry the exact literal `sol_stage_source_mismatch`.

Question dispositions (each bound in the cited section): Q1–Q4 → §3; Q5–Q6 → §3 and §7; Q7–Q9 → §8; Q10–Q12 → §4–§5; Q13–Q16 → §5 (Q13: **no** for capture journals; Q15's alternative is fully specified; Q16: **atomic rename required**); Q17–Q18 → §4 (Q18: **yes**, via `runner_refusal_code`); Q19 → §4 (**does not** count toward the C3 two-supersession bound; counts as preparation-failure #1 under a separate bound of 2); Q20 → §5 (**not** via `supersedes`, which stays null; via new `prior_preparation_failures`); Q21 → **yes**, §5/§10; Q22 → **yes**, the certified g++-14 runnability/shrink artifacts remain reusable — the failure was consumer-side comparison logic, compilation succeeded under the certified environment with exact fingerprint; Q23 → **yes**, the historical portable bootstrap remains valid unchanged under its existing allowlist; Q24 → **no**, no re-shrink or re-runnability is required; Q25–Q26 → §8/§10; Q27 → **confirmed**, §11; Q28 → §2.

## 2. PRIOR-ADJUDICATION CLAUSES AMENDED

1. **Annex A §4(3)** (staged-definition validation in `capture_adapter.py`/runner): the previously unspecified source-vs-staged definition equality rule is now bound as `SOL_DEF_EQUIV_V1` (§3). The staged-definition *hash* computation (packaged bytes + env-contract digest) is explicitly **unchanged**.
2. **Annex A §6** ("Every SOL cell" provenance list): additively extended with the definition-equivalence attestation and the source-copy seal (§7).
3. **Annex B §1 Q2(c) and C3 scope**: clarified/amended — the two-supersession bound counts only C3 GPU-capture supersessions. Pre-GPU preparation failures are governed by the preparation-failure archive protocol with a separate per-cell bound of 2 before an escalation flag is required.
4. **Annex B C5, §7(6), and §10** ("nothing under `superseded/`"): every occurrence of the superseded-leak prohibition is extended to also cover `preparation_failures/` paths.
5. **Annex B §5** (additive schema list): extended with `prior_preparation_failures`, the preparation-failure ledger and inventories, the stage source copy, and the equivalence attestation (§7).
6. **Annex A §7 and Annex B §9 checklists**: superseded for the remainder of Round 28 by §10 of this adjudication.

**All clauses of Annex A and Annex B not named above remain binding and unmodified**, including: the environment contract and literal `g++-14` pin, the certifying fingerprint equality rule `0d7c673c…`, ambient refusal gates, the 900s admission window and C2/C4 semantics, single-preflight chunk identity, whole-chunk C3 supersession mechanics, matrix admission checks 1–6, AC-9 compression-first routing, authority counts 344/122/222 with 18 chunks and 74 runnable SOL, the shell-origin mitigation posture (§8 of Annex B), and the pinned `gpt-5.5:high` review path.

## 3. DEFINITION EQUIVALENCE CONTRACT

**Rule `SOL_DEF_EQUIV_V1`** — the binding comparison between pinned source definition bytes S and packaged staged definition bytes T:

1. **Parse.** Decode both as UTF-8 JSON via `json.loads` with (a) an `object_pairs_hook` that refuses duplicate keys (typed error `sol_definition_duplicate_key`) and (b) rejection of NaN/Infinity tokens (typed error `sol_definition_nonfinite`). Parse failure → `sol_definition_parse_error`. Both top-level values must be JSON objects; otherwise refuse.
2. **Canonicalize (top level only).** For exactly the closed field set `{hf_id, op_type, custom_inputs_entrypoint}`, on each side independently: if the field is **absent**, or present with **null**, or present with the **string** `""` (string type required — checked via `isinstance(v, str)`), set it to `null`. Use a `_MISSING` sentinel exactly as in the proposed pseudocode, with the string-type guard. **No other field and no other value is ever canonicalized.** Nested occurrences of these names are NOT canonicalized.
3. **Compare.** Type-strict recursive equality of the canonicalized objects: null≡null; bool≡bool of equal value (bool checked before int, so `True ≢ 1`); int≡int; float≡float (`1 ≢ 1.0`); strings equal post-decode; arrays equal by length and element-wise recursion; objects equal by identical key sets and per-key recursion. **Unknown fields participate in exact equality on both sides** — a field present in source and dropped by the packager (or vice versa) is a mismatch and must refuse; no silent equivalence for unknowns.
4. **Outcome.** Equal → admit. Unequal → `sol_stage_source_mismatch`, now reporting the JSON-pointer paths of differing nodes (first 20).

Binding answers: **Q1** — the rule above. **Q2** — exactly `hf_id`, `op_type`, `custom_inputs_entrypoint`, top level only. **Q3** — yes, absent ≡ `""` ≡ null for those three fields only. **Q4** — yes: `0`, `False`, `[]`, `{}`, and every other value remain distinct everywhere, including for the three named fields (a non-string falsey value there is not canonicalized and must mismatch); bool≠int and int≠float are enforced. **Q5** — yes: the exact pinned source bytes must be copied into the persistent stage as `definition.source.json`, byte-verified against the matrix-pinned source SHA. **Q6** — yes: the stage manifest seals both `source_definition_sha256` and `staged_definition_sha256`, plus the equivalence attestation `{rule_id: "SOL_DEF_EQUIV_V1", per-field source_state/staged_state ∈ {absent, empty, null, value}, result}`. Equality applies **recursively to the entire document**; canonicalization applies **only at the top level**. The packager's serialization is not modified in Round 28; the comparator is consumer-side, single-sourced in a shared module.

Any divergence discovered (by the §8 audit or otherwise) outside these closed classes is a defect requiring immediate stop and a new escalation — Codex must not widen the rule unilaterally.

## 4. CLASSIFICATION OF THE PRESERVED ATTEMPT

**Q10:** The journal at `.staging/9b8e0e40…/sol_execbench-000-bootstrap/fd3a1868…/journals/sol_execbench__problem_070_…` is a **preparation attempt** (typed category `preparation_failure`), not a capture attempt. It passed admission and performed native compilation, but no eval driver, correctness run, warmup, candidate execution, NCU, NVBit, or control capture occurred. It is permanently non-authoritative and contributes zero rows to campaign accounting.

**Q17/Q18:** Its archive ledger entry seals `reason_code: STAGE_SEMANTIC_VALIDATION_FAILED` and the required additive field `runner_refusal_code: "sol_stage_source_mismatch"`.

**Q19:** This transition does **not** count toward Annex B C3's two-supersession bound (that bound exists to prevent GPU thrash; the cell's C3 supersession count remains 0). It establishes `preparation_failure_count = 1` for this cell under a new, separate bound: after 2 preparation-failure archives for the same cell, the runner refuses to open a new attempt without `--override-preparation-attempt-limit` (typed error `PREPARATION_ATTEMPT_LIMIT`).

**Q11 — binding no-GPU proof contract.** The archive's `no_gpu_attestation` object must contain, each item with evidence path + SHA-256:

1. Journal state `capture_started` with no `journal_complete.json` present.
2. Prepare stdout containing only the validated preparation manifest; empty stderr; both streams hashed.
3. A **forbidden-artifact scan**: an explicit checked pattern list (eval-driver execution outputs, NCU report files, NVBit trace/decoded-record files, control-capture artifacts, accepted-launch records) with zero matches across the journal inventory.
4. The runner refusal record: typed error `sol_stage_source_mismatch` emitted at semantic validation, with timestamp, and a phase log showing no phase after prepare/validate.
5. Static attestation that `ProblemPackager.execute()` at HEAD `9b8e0e40…` writes the driver and returns the command without executing it (file + function citation).
6. The recorded observation that no GPU process remained after refusal.

If item 3 finds any match, the tool must refuse with `PREPARATION_ARCHIVE_NOT_PREGPU` — such an attempt would not be archivable under this protocol and would require fresh escalation.

## 5. CROSS-HEAD TRANSITION PROTOCOL

**Q13:** Cross-HEAD supersession of capture journals is **not permitted**; C3 is unchanged. **Q14 is therefore moot.** **Q15/Q12/Q16** — the binding alternative:

1. **Ledger location (HEAD-independent):** `/home/jakeshea/sm86_reference_capture_r27/preparation_failures/preparation_failure_ledger.jsonl`, append-only, fsync'd (file and parent directory) per entry.
2. **Archival procedure** (new tool `tools/local_labels/archive_preparation_failure.py`, operator-invoked with explicit flag `--archive-preparation-failure`), in order: (a) verify the source journal's `journal_start.json` seal, old HEAD `9b8e0e40…`, old matrix SHA `fd3a1868…`/seal `5449383…`, old preflight `3810d2b2…`, session `f10de1cc…`, admission run id `0ff22c45…`, and full 22-file content inventory; (b) run the forbidden-artifact scan (§4); (c) verify `st_dev` equality between source and destination (`PREPARATION_ARCHIVE_CROSS_DEVICE` on failure); (d) atomically `rename()` the chunk's `journals/` parent (verify it contains exactly the one member journal first) to `/home/jakeshea/sm86_reference_capture_r27/preparation_failures/<UTC>Z__head-9b8e0e40__pf-3810d2b2__sol_execbench-000-bootstrap/`; (e) re-verify the inventory post-move; write `inventory.json` inside; on re-verify failure, refuse with `PREPARATION_ARCHIVE_INVENTORY_MISMATCH`, append no ledger entry, never roll back silently; (f) **copy** (not move) chunk-level admission/run-journal context into `chunk_context/` inside the archive, hashed, originals untouched; (g) append one ledger entry.
3. **Ledger entry fields (exact):** `schema_version: 1`; `entry_id` (uuid4); `entry_type: "preparation_failure_archive"`; `created_at_utc`; `rlcr_session_id`; `rlcr_dir`; `campaign_round: 28`; `chunk_id`; `cell_id`; `reason_code: STAGE_SEMANTIC_VALIDATION_FAILED`; `runner_refusal_code: sol_stage_source_mismatch`; `original_head`; `original_matrix_file_sha256`; `original_matrix_seal_sha256`; `original_preflight_sha256`; `original_admission_run_id`; `original_session_id`; `replacement_head` (the fix commit, known at append time); `source_journal_path`; `archive_path`; `inventory_file` + `inventory_sha256`; `no_gpu_attestation` (§4); `pinned_source_definition_sha256: 3e0aceb3…`; `staged_definition_sha256: e7526755…`; `staged_module_sha256: 017ec9c8…`; `defect_summary` (one line: comparator lacked absent≡null for optional fields); `adjudication_ref` (path in §7 item 5); `operator_flag` (exact CLI string).
4. **Q16:** Atomic rename is **required**; in-place retention is rejected because leak enforcement keys off the `preparation_failures/` namespace and an in-place `capture_started` journal remains an active-looking, ledger-unclosed attempt.
5. **Discovery and back-reference (Q20):** the replacement attempt's journal keeps `supersedes: null` and must carry the new field `prior_preparation_failures: [{ledger_entry_id, ledger_path, archive_path}]` listing exactly the ledger entries for its cell. The runner consults the ledger at attempt open to populate this and enforce the bound; unparseable ledger → `PREPARATION_LEDGER_INCONSISTENT`.
6. **Verifier/accounting closure:** DONE semantic verification asserts, per member cell: the ledger's entries for that cell are complete and inventory-verified; the active journal's `prior_preparation_failures` equals that entry set exactly; no published artifact path resolves under `preparation_failures/` or `superseded/` (`PREPARATION_ARCHIVE_LEAK`); dangling references in either direction → `PREPARATION_LEDGER_INCONSISTENT`. Accounting extends §7(6) of Annex B identically.
7. **Q21 — sequencing:** the archival transition must be fully sealed (move + inventory + ledger fsync) **after** the fix commit exists (so `replacement_head` is known) and **before** matrix regeneration, the new certifying preflight, and the bootstrap. The old journal remains byte-identical in place until the archival step.
8. The archived staged module `017ec9c8…` and every other archived artifact are permanently non-authoritative and must never be spliced into any new stage.

## 6. REQUIRED CODE CHANGES — IN ORDER

1. **New shared module `tools/local_labels/sol_definition_equivalence.py`:** implements `SOL_DEF_EQUIV_V1` exactly per §3 (strict parse, closed canonicalization with `_MISSING` sentinel and string-type guard, type-strict recursive comparator, JSON-pointer diff reporting). Single-source the rule id string and the closed field set here.
2. **`tools/local_labels/sm86_capture_chunk.py`:** replace `normalized_definition`-based comparison with the shared rule; register new typed errors (`sol_definition_parse_error`, `sol_definition_duplicate_key`, `sol_definition_nonfinite`, `sol_stage_source_copy_missing`, `sol_stage_source_copy_mismatch`); add preparation-ledger consultation at attempt open (populate `prior_preparation_failures`, enforce the bound of 2, `PREPARATION_ATTEMPT_LIMIT`).
3. **Stage evidence producer** (`capture_adapter.py` / the prepare-manifest composer): write `definition.source.json` as an exact byte copy of the pinned source; verify it against the matrix-pinned source SHA; seal both SHAs and the equivalence attestation into the stage manifest. The staged-definition hash computation remains byte-for-byte as at `9b8e0e40…`.
4. **New tool `tools/local_labels/archive_preparation_failure.py`:** implements §5 with all typed errors (`PREPARATION_ARCHIVE_SOURCE_MISSING`, `PREPARATION_ARCHIVE_NOT_PREGPU`, `PREPARATION_ARCHIVE_CROSS_DEVICE`, `PREPARATION_ARCHIVE_INVENTORY_MISMATCH`).
5. **`campaign_done_manifest.py` + DONE schema:** additive required per-cell field `prior_preparation_failures` (default `[]`); DONE records per-cell preparation ledger entry ids; verifier implements §5(6). The historical portable bootstrap allowlist is unchanged.
6. **`campaign_accounting.py`:** extend the join-stage checks with preparation-ledger consistency and `preparation_failures/` leak detection; both produce distinct typed errors; matrix admission checks 1–5 remain strictly first.
7. **Schemas:** journal/cell-result additive `prior_preparation_failures`; stage-manifest additive source-copy + attestation fields; new preparation-failure ledger entry schema file.
8. **Tests** per §8/§9, including the 74-definition audit and tombstone comments citing this adjudication wherever old comparator behavior is inverted.
9. **Docs/protocol registry:** register every new typed error code; document `SOL_DEF_EQUIV_V1` and the preparation-failure protocol. `capture_budget_route.py` is untouched; AC-9 semantics unchanged — state so in the round journal.

## 7. REQUIRED SCHEMA AND PROVENANCE

1. **Stage (every future SOL cell):** `definition.source.json` byte copy; stage manifest seals `source_definition_sha256`, `staged_definition_sha256`, and the equivalence attestation (rule id, per-field states, result). For the retried problem-070 stage, expected values are `3e0aceb3…` (source), `e7526755…` (staged, packager unchanged), attestation showing `op_type: source_state=absent, staged_state=null`.
2. **Cell journal / result:** `prior_preparation_failures` (required, default empty); `supersedes` unchanged.
3. **DONE:** per-cell preparation ledger entry ids; verifier closure per §5(6); `preflight_sha256` semantics unchanged — the new bootstrap DONE stamps the **new** preflight; the old preflight SHA appears only inside the archive ledger.
4. **New files:** `preparation_failures/preparation_failure_ledger.jsonl` (append-only) and per-archive `inventory.json` + `chunk_context/`.
5. **RLCR records:** this adjudication saved at `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_stage_equivalence.md`; the 74-definition audit artifact at `.humanize/rlcr/2026-07-08_11-39-13/sol_definition_equivalence_audit_r28.json` (per definition: problem id, source SHA, round-trip staged SHA, canonicalized fields, verdict).
6. **Untouched and read-only:** the certified fingerprint JSON, runnability/shrink artifacts, the historical portable bootstrap, the pinned source definitions, and the archived old attempt.

## 8. REQUIRED POSITIVE TESTS

1. **Q7/Q8 integration test (binding):** run the real `ProblemPackager` staging path on the real in-tree problem-070 definition (assert its SHA equals `3e0aceb3…` as a drift tripwire, and that `op_type` is absent), producing an actual staged `definition.json`; feed real output through the real `_validate_sol_prepare_manifest` path; assert admission. Only the compile subprocess may be stubbed, and only if the packager cannot stage without nvcc; the serialization path must be the real code.
2. **Q9 audit (binding, pre-retry):** for all 74 runnable SOL definitions, in-process parse→serialize round-trip compared under `SOL_DEF_EQUIV_V1`; zero divergences outside the closed classes; committed as a test and recorded as the RLCR artifact. Any divergence → nonzero exit listing offending JSON pointers; stop and escalate.
3. Parametrized 9-case equivalence: each of the three fields × {absent, `""`, null} in source vs packaged null → admit; byte hashes sealed.
4. Comparator self-tests: `True ≢ 1`; `1 ≢ 1.0`; `"" ≢ null` for a non-listed field; nested `op_type` not canonicalized; identical nested documents equal; duplicate-key and NaN/Infinity refusals fire.
5. Source-copy round-trip: staged dir contains byte-identical `definition.source.json`; manifest seals both SHAs; attestation `result: equal`; determinism across two invocations.
6. Archival happy path (fixture old work root, `capture_started` journal, clean forbidden-artifact scan): atomic move, post-move inventory verification, complete ledger entry, `chunk_context/` copies; then a new-HEAD attempt whose journal carries the correct `prior_preparation_failures`; DONE verifier and accounting accept with zero leaks.
7. Regression: Annex B fast-resume, finalization carve-out, and admission-window suites remain green (no behavioral drift from the comparator or ledger changes).

## 9. REQUIRED NEGATIVE TESTS

Each must refuse with the exact typed error and perform no GPU work:

1. Staged `op_type: "foo"` vs source absent → `sol_stage_source_mismatch` with correct JSON pointer.
2. Unknown top-level field in source, dropped in staged → `sol_stage_source_mismatch`.
3. Nested absent-vs-null for a listed field name → refuse (top-level-only canonicalization).
4. `hf_id: 0` vs null → refuse (non-string never canonicalized); `True` vs `1` and `1` vs `1.0` → refuse.
5. Missing or byte-divergent `definition.source.json` → `sol_stage_source_copy_missing` / `sol_stage_source_copy_mismatch`.
6. Duplicate keys / NaN token in either document → `sol_definition_duplicate_key` / `sol_definition_nonfinite`.
7. Archive tool on a journal containing an NCU/NVBit/eval artifact → `PREPARATION_ARCHIVE_NOT_PREGPU`.
8. Cross-device destination → `PREPARATION_ARCHIVE_CROSS_DEVICE`; post-move inventory mismatch → `PREPARATION_ARCHIVE_INVENTORY_MISMATCH` with no ledger append.
9. Archive dir without ledger entry, ledger entry without archive dir, or journal back-reference set ≠ ledger set → `PREPARATION_LEDGER_INCONSISTENT` at DONE verification and at accounting.
10. Published artifact path resolving under `preparation_failures/` → `PREPARATION_ARCHIVE_LEAK`.
11. Third preparation attempt for one cell without the escalation flag → `PREPARATION_ATTEMPT_LIMIT`.
12. Ordinary-shell preflight with ambient `NVCC_CCBIN=/usr/bin/g++-15` → `SOL_ENV_AMBIENT_CONFLICT` (tripwire re-armed evidence, since the runner file changed).

## 10. EXACT RECOVERY AND BOOTSTRAP CHECKLIST

1. Verify the preserved journal is still byte-identical (inventory hash check); no captures started; repo clean at `9b8e0e40…`.
2. Implement §6 changes 1–9. Run: full `tools/local_labels` suite, SOL adapter/runtime, NVBit Python and host tests, DONE/accounting/runner adversarial suites, focused DONE suite, all §8/§9 tests, the 74-definition audit, Python compilation, shell syntax, `git diff --check` — all green (this is the complete Q25 evidence set; no GPU required).
3. Commit; record the new HEAD **H2**; working tree clean. The `9b8e0e40…` matrix and preflight are thereby stale and must not be reused.
4. Run `archive_preparation_failure.py` per §5 with `replacement_head = H2`; verify ledger entry, `inventory.json`, `no_gpu_attestation`, and that the old work root no longer contains `journals/`.
5. Regenerate authority + matrix at H2 into `/home/jakeshea/sm86_reference_capture_r27/inputs/r28_<H2-8>/`. Verify: universe 344, runnable 122, excluded 222, budgets 122, blocked 0, chunks 18, runnable SOL 74; fingerprint pin present and equal to `0d7c673c…`. Any count drift → stop and escalate.
6. Run accounting matrix admission (checks 1–5): expect accounted 1 / missing 121 / duplicates 0 / leaks 0 (the allowlisted historical bootstrap only).
7. Verify wrapper SHA still `77a96b4e…`; record one ordinary-shell negative preflight (`SOL_ENV_AMBIENT_CONFLICT`) per §9(12).
8. In the dedicated shell: generate a fresh certifying preflight; admit within 900s; fingerprint exactly `0d7c673c…`, degraded false, missing `[]`, injection ok, HEAD H2, matrix SHA exact.
9. Execute the one-cell SOL bootstrap. Verify: staged definition SHA `e7526755…`, source copy SHA `3e0aceb3…`, attestation `equal`; exactly one accepted launch; no inactive-launch trace records; module identity sealed (SHA-256 + path-at-capture); complete phase evidence; AC-9 compression-first on any breach; sealed env components matching the certified hash; `prior_preparation_failures` correctly populated; DONE stamps the new preflight; semantic verification passes.
10. Run accounting: exactly 2 accounted / 120 missing / 0 duplicates / 0 unexpected / 0 leaks; integrity true.
11. Submit bootstrap evidence through pinned `gpt-5.5:high`; on any additional harness safety check, wait and retry the identical path.
12. Only after acceptance: proceed to chunked batch capture under Annex B C2–C4, per-chunk fingerprint re-probes, shared monotonic deadlines, AC-9 routing, toward exactly 122 terminal rows.

## 11. ACCOUNTING AND REVIEW GATES

Q27 is confirmed in full: no other chunk may start until the repaired one-cell SOL bootstrap seals, its DONE passes semantic verification (including §5(6) closure), accounting reads exactly 2 accounted / 120 missing with zero duplicates, unexpected, stale, superseded leaks, and preparation-archive leaks, and the pinned `gpt-5.5:high` review accepts the bootstrap evidence. The archived preparation attempt never appears as an accounting row; `complete=true` semantics, matrix-admission ordering, and the 122-row terminal contract are unchanged.

## 12. FORBIDDEN SHORTCUTS

- Editing any pinned `definition.json` (e.g., inserting `"op_type": null` into source) to satisfy the comparator — input rewriting.
- Overwriting the staged definition with source bytes, re-hashing staged artifacts post-hoc, or any in-place stage mutation.
- Widening `SOL_DEF_EQUIV_V1` beyond the three fields and three states, adding recursive canonicalization, or equivalencing unknown fields — any new divergence class requires a fresh escalation.
- Hand-moving, hand-editing, or deleting the preserved journal; only the archival tool may relocate it, and only per §5.
- Classifying this attempt as a C3 supersession, consuming C3 counts for it, or misusing `PREFLIGHT_EXPIRED_BEFORE_DONE`.
- Reusing any artifact from the archived attempt — including the compiled module `017ec9c8…` — in the new stage, however identical it would be.
- Starting the bootstrap before the archive is sealed, capturing under the stale `3810d2b2…` preflight, reusing the `fd3a1868…` matrix at H2, or capturing with a dirty tree.
- Skipping the 74-definition audit, or typed-disposing/excluding SOL cells to route around definition mismatches.
- Publishing or counting anything under `preparation_failures/` or `superseded/`; backdating or synthesizing timestamps.
- Switching or downgrading the pinned review path; on any harness safety check, wait and retry the identical `gpt-5.5:high` path.

As with the fingerprint tripwire in Annex A, this refusal was the provenance system working: it caught a representation divergence before any GPU evidence existed. The closed equivalence rule plus the preparation-failure archive converts both the defect and the interrupted attempt into permanently sealed, non-authoritative history with full bidirectional closure. This adjudication is binding and self-contained; Codex may implement it without further interpretation.
