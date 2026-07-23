# FABLE-5 BINDING SUPPLEMENTAL CORRECTION — Round 28 Post-Move Failure-State Contract

**Session:** `b41e0614-a371-4041-b2ed-0d0aecfda548` · **Round:** 28
**Supplements and narrowly interprets:** `ANNEX_C_AMEND_R28_PREP070_RECON_V2` as corrected by `R28_PREP070_RECON_V2_CORR1` (which remain the live amendment and correction IDs — no V3 is minted, no live schema const changes)
**Correction ID:** `R28_PREP070_RECON_V2_CORR2`
**Save verbatim to:** `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_post_move_failure_state_correction.md`, mode `0444`, before implementation resumes; compute its SHA-256 and append the Section 5 session-log entry immediately after saving.

---

## 1. DECISION

1. **The inherited clause remains normative.** The re-enacted negative-test requirement — "Post-move verification failure → no rollback occurs, dedicated failure state recorded (Annex C no-silent-rollback preserved)" — was re-enacted by V2 §3/§9 and CORR1 §1.8 as a non-defective clause and is binding. It is not struck, weakened, or reworded.
2. **Option A is binding, in a refined and fully specified form.** The dedicated failure state is **the arrested structural state itself**: the atomically renamed destination directory present under `preparation_failures/`, the absent original `journals/` parent, the absent (or row-less) ledger, the absent `preparation_failure_archived` session-log event, and the typed refusal emitted at event time. "Recorded" is satisfied **by construction**: the state is durable, deterministic, machine-recognizable, and typed — and, decisively, it is a record that **cannot fail to be written**, because it is constituted by the absence of completions rather than the presence of a new artifact. A failure *of* the recording machinery cannot be trustworthily documented *by* that same machinery at failure time; the only honest record is the one requiring no successful post-failure write. This is the same honesty principle that grounded V2's withdrawal of the false certifying premise and its timestamp rules.
3. **Option B is rejected.** A `POST_MOVE_FAILURE.json` (or any marker) written after a detected integrity failure is untrustworthy by its own premise; it may itself fail to write (recreating this exact ambiguity recursively); it contaminates the archive tree — `verify_preparation_archive` requires the destination to contain **exactly one** regular journal directory besides `inventory.json` and `chunk_context`, so a root-level marker breaks every future adjudicated verification; and its schema, timing truth, and inventory membership are all unadjudicated. No marker file of any name may ever be created.
4. **Option C is rejected.** A separate failure ledger in the preparation root violates the loader's closure invariant ("unexpected preparation-failure root entry"); a `session_log.jsonl` failure event breaks CORR1 §8's closed required-entry ordering, which `_verify_session_log_prefix` enforces by exact equality; and both create a new record class whose own append failure reproduces the original problem.
5. **Option D is rejected as framing, absorbed as substance.** The clause is not "non-normative shorthand" to be removed; it is given its precise meaning. Option D's useful residue — explicit positive/negative tests pinning the structural state — is adopted in Sections 9–10.
6. **No new refusal codes, files, schemas, or ledger fields are minted for this state.** The existing typed codes are bound to it exhaustively in Section 3.
7. Everything not amended here remains binding verbatim: Annexes A–D; V2 in full (authority rebinding to `SOL_DEF_LEGACY_COMPARE_9B8E0E40`, the literal `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`, conjunctive dual-basis); CORR1 in full; frozen `SOL_DEF_EQUIV_V1` and both immutable audits; H2 `823f2d60c9bc5c5e78d21ff0c74f3e48ceaa1abc`; the H3 subject `fix(local_labels): rebind prep070 refusal authority to legacy 9b8e0e40 comparator`; all cardinalities (344/122/222/0; budgets 122; chunks 18; runnable SOL 74; bootstrap target 2/120/0/0/0/integrity true; final exactly 122); AC-9; and the pinned `gpt-5.5:high` review path with wait-and-retry and no model substitution.

## 2. TRIGGER BOUNDARY

The contract covers **every failure after a successful `_rename_noreplace` return**, in both `contemporaneous` and `reconstructed` context modes, partitioned at one boundary — the durable ledger append:

**FS-PRE (the dedicated post-move failure state):** any failure strictly after the atomic rename and strictly before the ledger row's fsynced append completes. This includes, exhaustively for the stopped implementation: the post-move `content_inventory` re-read; the `moved_rows != rows` / inventory-domain-SHA comparison; the `inventory.json` create-exclusive write; `_copy_context` (including the `reconstructed/` context copies); `verify_preparation_archive`; `validate_preparation_failure_entry(..., verify_archive=False)`; `_verify_reconstructed_archive_entry(..., require_archive_log_event=False)`; and the pre-append `_verify_session_log_prefix`. Signature: destination present and unledgered; original absent.

**FS-POST (post-seal inconsistency states):** failures at or after the durable append: a torn/partial ledger line; a validated row present but the reconstructed-mode `preparation_failure_archived` session-log event absent; or a failure of the final `load_preparation_failure_ledger(..., verify_archives=True)` closure pass.

Both partitions share the identical contract: **no rollback, no deletion, no hand-editing, no marker, typed refusal, permanent stop, mandatory fresh Fable-5 adjudication.** They differ only in structural signature and rediscovery code (Section 3).

## 3. EXACT FAILURE-STATE REPRESENTATION

**FS-PRE structural signature (all five conditions):**

1. The `journals/` parent is absent at its staging path (live: `/home/jakeshea/sm86_reference_capture_r27/.staging/9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a/sol_execbench-000-bootstrap/fd3a1868…/journals`).
2. A destination directory exists directly under `preparation_failures/`, named by the tool's pre-rename convention `<UTC>Z__head-9b8e0e40__pf-3810d2b2__sol_execbench-000-bootstrap`.
3. No validated ledger row's `archive_path` resolves to that directory — either `preparation_failure_ledger.jsonl` is absent entirely, or it lacks the row.
4. `session_log.jsonl` contains no `preparation_failure_archived` event for it.
5. The moved journal bytes are inside the destination, possibly accompanied by legitimate post-rename residue: `inventory.json` and/or a partial `chunk_context/`, depending on the failure point. Residue is **non-authoritative** and must never be parsed for any decision except the re-adjudication's own forensics.

**Contemporaneous (event-time) representation:** the typed refusal actually raised, unchanged from the stopped tree — for the first post-rename re-verification, exactly `PREPARATION_ARCHIVE_INVENTORY_MISMATCH` with detail literal `post-move inventory drift`, printed by the CLI as `archive_preparation_failure: REFUSED [PREPARATION_ARCHIVE_INVENTORY_MISMATCH] post-move inventory drift`. Other FS-PRE sites raise their existing codes (`PREPARATION_ARCHIVE_INVENTORY_MISMATCH` variants, `PREPARATION_LEDGER_INCONSISTENT`, `PREP_ARCHIVE_RECON_SCHEMA_INVALID`, `RLCR_SESSION_LOG_INCONSISTENT`), and a raw `OSError` surfaces via the CLI's existing wrapper as `REFUSED [PREPARATION_ARCHIVE_SOURCE_MISSING] <type>: <detail>`. These event-time codes are **operational only**; the dedicated recognition does not depend on which one fired, or on whether the process survived to print it.

**Rediscovery (durable) recognition:**

- FS-PRE with no ledger file → `PREPARATION_LEDGER_INCONSISTENT`, detail `archive evidence exists without ledger: [...]` naming the exact unledgered path(s).
- FS-PRE with a ledger file → `PREPARATION_LEDGER_INCONSISTENT`, detail `ledger/archive set mismatch: unledgered=[...]; missing=[...]`.
- FS-POST torn line → `PREPARATION_LEDGER_INCONSISTENT` at the `{path}:{line_number}` parse site.
- FS-POST row-present/log-event-absent (reconstructed) → `RLCR_SESSION_LOG_INCONSISTENT`, detail `archived prefix mismatch: rows=<n>`.

These existing details **are** the dedicated identification: they name the state and the exact paths. The state must never be reported as `PREPARATION_ARCHIVE_LEAK` (reserved exclusively for published artifact paths resolving under `preparation_failures/` or `superseded/`) and never as an untyped error.

**Quarantine rule:** after FS-PRE/FS-POST entry, the destination's contents are non-citable. No certifying record may cite the moved journal until a fresh adjudication re-verifies it against the sealed pins (22 files / 3,855,797 bytes / `ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`).

## 4. DURABILITY AND ATOMICITY

The atomic `renameat2(RENAME_NOREPLACE)` is the sole pre-append commitment point; the tool's immediately following `_fsync_directory(prep_root)` and `_fsync_directory(journals.parent)` make the move crash-durable. Therefore FS-PRE requires **no additional durability action**: its persistence is exactly the durability of the rename already performed. If a crash lands between the rename and the directory fsyncs and the filesystem loses the rename, the residue is the *original* pre-archive state (journal intact in place, nothing in `preparation_failures/`) — which is not a failure state and retries cleanly. Both possible residues are closed; there is no third, ambiguous residue. `inventory.json` (create-exclusive + fsync) and the ledger append (fsync of file and parent) retain their existing semantics. The `.capture.lock` exclusive flock continues to serialize all writers; it releases automatically on process death, so a crashed archiver leaves the durable structural state and a free lock. No new write, rename-back, or cleanup of any kind is permitted after the failure is detected.

## 5. LEDGER / SESSION-LOG / INVENTORY SEMANTICS

1. **No ledger row is appended for the failure state** — neither an authoritative row nor any distinguishable "failure row." Ledger rows exist only for completed, verified archives.
2. **No `session_log.jsonl` event is appended for the failure state.** CORR1 §8's required-entry ordering is closed; failure events would make the exact-prefix verification state-dependent and would themselves be subject to append failure.
3. **`inventory.json` never carries a marker.** In FS-PRE it is either absent (its absence is part of the signature) or non-authoritative residue.
4. **This correction itself is saved and logged.** Save this document verbatim, mode `0444`, at the header path; compute its SHA-256; append to `.humanize/rlcr/2026-07-08_11-39-13/session_log.jsonl` (fsynced, `sort_keys=True`, separators `(",", ":")`) the new required entry 2, immediately after CORR1's entry 1 and before the two V2 §10-step-5 entries:

```json
{"event":"adjudication_saved","artifact":"adjudication_r28_post_move_failure_state_correction.md","sha256":"<computed>","status":"binding_supplement","supplements_amendment_id":"ANNEX_C_AMEND_R28_PREP070_RECON_V2","correction_id":"R28_PREP070_RECON_V2_CORR2"}
```

CORR1 §8's required sequence is amended to: entry 1 (CORR1 save, already present), **entry 2 (this CORR2 save)**, entries 3–4 (superseded-V1 and binding-V2 saves, after H3), entries 5–11 (the seven `reconstruction_record_created` events), entry 12 (`preparation_failure_archived`). The ledger Arm-B `authority` object is **unchanged** (six fields, CORR1 consts); CORR2 is cited only through this log entry and the new module constant.

## 6. CONSUMER AND ACCOUNTING CLOSURE

1. `load_preparation_failure_ledger(..., verify_archives=True)` recognizes FS-PRE/FS-POST exactly as in Section 3 and refuses. This strict mode is mandatory at every certifying gate: the archive tool (both modes, already implemented), the DONE verifier's Annex C §5(6) closure, and accounting's join-stage checks. During implementation Codex must confirm each of these call sites uses `verify_archives=True`; any lax call site is a conformance defect against already-bound Annex C §5(5)–(6) and must be switched to strict — this is not new semantics. The runner's attempt-open consultation retains its Annex C behavior unchanged.
2. Any re-invocation of the archive tool refuses before mutation: contemporaneous mode via the missing-`journals/` check (`PREPARATION_ARCHIVE_SOURCE_MISSING`); reconstructed mode likewise after its read-only preconditions; and any archive attempt for **any** cell sharing the preparation root refuses at its strict ledger load. The state fail-closes the entire preparation root — this is intended.
3. The failure state contributes zero campaign rows, zero `preparation_failure_count` (the two-per-cell bound counts validated ledger rows only), consumes no Annex-B supersession attempts, and **cannot** satisfy `prior_preparation_failures` (`preparation_failure_backrefs` derives exclusively from validated rows).
4. DONE and accounting must surface it as `PREPARATION_LEDGER_INCONSISTENT` (or `RLCR_SESSION_LOG_INCONSISTENT` for the FS-POST log-gap variant) — never as a leak, never as a countable or publishable object.

## 7. RECOVERY / ONE-SHOT CONTRACT

1. **Recovery is permanent stop plus mandatory fresh Fable-5 adjudication.** No dedicated resume/finalize command is minted. Silent rollback, rename-back, deletion of the destination, hand-created ledger rows or log events, and any edit inside the destination remain forbidden.
2. The escalation must include: the exact stderr refusal line; a listing of `preparation_failures/`; a read-only rehash of the destination journal against the pins 22 / 3,855,797 / `ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`; and current git state. Only the returned adjudication may authorize either a verified completion path (with newly bound semantics) or a damage classification.
3. **One-shot/race behavior (Question 11):** the state cannot fail to be recorded. If the process dies before printing anything, the structural state persists and the next strict load refuses. There is no marker write to race, duplicate, or lose. The one-shot reconstructed-incident guard is preserved: after FS-PRE, no ledger row exists, so nothing is consumed; after a *successful* seal, the existing "already sealed" and session-log checks refuse a second invocation, exactly as the stopped tests demonstrate.

## 8. REQUIRED CODE CHANGES

The runtime failure-path behavior of `tools/local_labels/archive_preparation_failure.py` is retained **byte-for-byte in effect**; no marker writer, no new refusal code, no schema file changes, no edits to the six CORR1 schemas. Exactly these deltas are authorized, folded into the single H3 commit (subject unchanged):

1. Add module constants: `ROUND28_CORRECTION2_ID = "R28_PREP070_RECON_V2_CORR2"`; `ROUND28_CORRECTION2_ADJUDICATION = ROUND28_RLCR_DIR / "adjudication_r28_post_move_failure_state_correction.md"`; `ROUND28_CORRECTION2_SHA256 = "<computed at save time>"`.
2. Extend `_adjudication_hashes` with a `"correction2"` ref (regular file, mode `0444`, SHA equal to `ROUND28_CORRECTION2_SHA256`, all four SHAs pairwise distinct; refusal `PREP_ARCHIVE_RECON_SCHEMA_INVALID` as for the others). Extend `ArchiveRequest` with `correction2_adjudication_ref: Path = ROUND28_CORRECTION2_ADJUDICATION`, the CLI with `--correction2-adjudication-ref`, and `_verify_reconstructed_allowlist`'s `path_expected` accordingly.
3. Insert the Section 5 entry-2 object at index 1 of both `_required_session_prefix` and `_verify_reconstructed_session_log`'s expected list (the latter reads `ROUND28_CORRECTION2_SHA256` from the constant, since the ledger `authority` object is unchanged).
4. Add the Section 9–10 tests; update the reconstructed test fixture to create the CORR2 file (live bytes, `0444`) and monkeypatch the new constants, mirroring its existing CORR1 handling.
5. Rerun the complete no-GPU battery; all suites green (the focused four-file suite grows beyond 91; no count is pinned).

## 9. REQUIRED POSITIVE TESTS

1. **Retain and extend** `test_post_move_inventory_mismatch_appends_no_ledger`: additionally assert the destination directory exists under the preparation root containing exactly the moved journal (this injection point), with no `inventory.json`, no `chunk_context/`, and no file named `POST_MOVE_FAILURE.json` (or any `*FAILURE*` marker) anywhere under the root; that a subsequent `load_preparation_failure_ledger(..., verify_archives=True)` raises `PREPARATION_LEDGER_INCONSISTENT` with detail containing `archive evidence exists without ledger`; and that a second `archive_preparation_failure(request)` raises `PREPARATION_ARCHIVE_SOURCE_MISSING` while leaving the preparation-root tree byte-identical.
2. **New** `test_reconstructed_post_move_failure_state_is_structural_and_fail_closed`: reconstructed fixture; inject post-move inventory drift; assert `PREPARATION_ARCHIVE_INVENTORY_MISMATCH`; journals root gone; destination unledgered; construction-site originals intact; `session_log.jsonl` rows exactly equal to the pre-archive required prefix (no failure event appended); strict load refuses; re-invocation refuses typed without mutation.
3. **New** `test_corr2_artifact_is_verified_and_logged`: builder succeeds only with the CORR2 file present, mode `0444`, SHA matching, and log entry 2 in place.

## 10. REQUIRED NEGATIVE TESTS

1. Parameterized FS-PRE family over injection points (post-move `content_inventory` failure; `inventory.json` write failure; `_copy_context` failure; `verify_preparation_archive` failure): each yields its typed code, appends no ledger row, appends no session-log event, creates no marker file, and leaves a state the strict loader refuses as `PREPARATION_LEDGER_INCONSISTENT`.
2. Torn ledger append (partial final line injected via a failing `_append_jsonl_fsync`) → strict load `PREPARATION_LEDGER_INCONSISTENT`; re-assert the existing truncation test.
3. Reconstructed FS-POST: ledger append succeeds, session-log `preparation_failure_archived` append fails → archive raises; strict load refuses `RLCR_SESSION_LOG_INCONSISTENT`; the appended row is never edited or removed.
4. Static guard: the module defines no identifier and no filename literal matching `POST_MOVE_FAILURE`; no refusal code beginning `PREP_ARCHIVE_POST_MOVE` exists.
5. CORR2 controls: log missing entry 2 → `RLCR_SESSION_LOG_INCONSISTENT`; CORR2 file mode `0644` or SHA drift → `PREP_ARCHIVE_RECON_SCHEMA_INVALID`.
6. Re-assert unchanged: `test_bidirectional_closure_rejects_archive_without_ledger`, `test_bidirectional_closure_rejects_extra_archive_directory`, `test_ledger_loader_rejects_truncation_and_archive_drift`, and the leak tests for `preparation_failures/` / `superseded/` publication.

## 11. ANSWERS TO QUESTIONS 1–14

1. **Yes**, normative; this correction binds its meaning rather than striking it (Section 1).
2. **Option A**, refined per Sections 2–7; B and C rejected; D rejected as framing, its test residue absorbed.
3. **Every failure after the rename**, partitioned at the durable ledger append: FS-PRE (dedicated structural state) before it; FS-POST (torn line; row-without-log-event; failed closure pass) after it — identical recovery contract (Section 2).
4. **No file is required, and every marker file is forbidden** — path, schema, serialization, permissions, and ordering are all N/A by prohibition.
5. **No timestamp is minted.** The stderr refusal is merely operational; the structural state is durable and **non-certifying** (it certifies nothing historical; it only gates). Directory-name timestamps and mtimes must never be read as the failure instant.
6. **Not in `inventory.json`.** Its absence is part of the FS-PRE signature; when present it is non-authoritative residue (Section 3).
7. **No ledger row of any kind.** Zero campaign rows; no `preparation_failure_count` consumption; cannot satisfy `prior_preparation_failures` (Section 6.3).
8. Per Sections 3 and 6: strict loads refuse `PREPARATION_LEDGER_INCONSISTENT` with the two bound detail literals (`RLCR_SESSION_LOG_INCONSISTENT` for the FS-POST log gap); certifying gates must use `verify_archives=True`; never reported as `PREPARATION_ARCHIVE_LEAK` or untyped.
9. **No session-log event for the failure state.** The only new log requirement is this correction's own save event, exact shape in Section 5.
10. **Permanent stop and re-adjudication** with the Section 7 evidence bundle; no resume/finalize command; rollback, deletion, and hand-editing remain forbidden.
11. Recording cannot fail: the state is constituted by absences plus the already-durable rename; crash residues are closed in both directions (Sections 4, 7.3).
12. Sections 9–10, with the refusal codes and detail literals of Section 3.
13. **Retain H2/stopped-tree runtime behavior unchanged** for this path; add only the Section 8 CORR2-logging deltas and the Section 9–10 tests before H3, inside the single H3 commit with the bound subject.
14. **Confirmed.** All other CORR1/V2/Annex A–D clauses, all cardinalities (344/122/222/0; 122 budgets; 18 chunks; 74 runnable SOL; 2/120/0/0/0/true; final 122), the archive sequencing of CORR1 §10 (with step 0′ = save-and-log this correction), and the pinned `gpt-5.5:high` review path with wait-and-retry remain unchanged.

## 12. FORBIDDEN SHORTCUTS

All V2 §12 and CORR1 §12 prohibitions remain in force verbatim, and additionally: do not create `POST_MOVE_FAILURE.json` or any failure marker, flag file, or sentinel under any name, in the destination, the preparation root, the journal, or the RLCR directory; do not mint any new refusal code for this state; do not roll back, rename back, delete, or edit the destination or its residue; do not hand-write a ledger row, session-log event, or inventory describing the failure; do not parse FS-state residue (`inventory.json`, partial `chunk_context/`) as authoritative; do not interpret directory-name timestamps or mtimes as failure event times; do not add a resume/finalize command or auto-repair path; do not report the state as `PREPARATION_ARCHIVE_LEAK`, count it, or publish anything under `preparation_failures/` or `superseded/`; do not weaken any strict-load call to `verify_archives=False` at a certifying gate; do not split the Section 8 deltas into a second commit or alter the bound H3 subject; do not treat the moved journal as citable before re-adjudicated re-verification; and do not switch, downgrade, or substitute the pinned `gpt-5.5:high` review path — on any safety or capacity check, wait and retry the identical path.

**End of binding supplemental correction.** Codex may resume at Section 5's save-and-log step, then apply Sections 8–10 within the CORR1 §10 ordering, and must stop and re-escalate under `AGENTS.md` on any further material ambiguity rather than guessing or weakening a check.
